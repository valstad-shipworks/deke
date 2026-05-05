use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

use hafgufa::{Options, Problem, VariableArena, subject_to};

use deke_types::{DekeError, DekeResult};

use crate::Topp3Tcp6Constraints;
use crate::boundary::ProjectedBoundary;
use crate::diagnostic::SolveStatus;
use crate::path_derivatives::PathDerivatives;

/// Numeric output of the NLP solve — everything downstream uses this POD struct so that the
/// `VariableArena` can be dropped before the next pipeline stage.
#[derive(Debug, Clone)]
pub struct Solution {
    pub sd: Vec<f64>,
    pub sdd: Vec<f64>,
    pub sddd: Vec<f64>,
    pub dt: Vec<Duration>,
    pub status: SolveStatus,
    pub iterations: i32,
    pub solve_time: Duration,
}

pub fn build_and_solve<const N: usize>(
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
    start: ProjectedBoundary,
    end: ProjectedBoundary,
) -> DekeResult<Solution> {
    let m = deriv.num_waypoints();
    let seg = deriv.num_segments();
    if m < 2 || seg == 0 {
        return Err(DekeError::PathTooShort(m));
    }
    let lock = constraints.locked_prefix.min(N);

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let sd: Vec<_> = (0..m).map(|_| problem.decision_variable()).collect();
    let sdd: Vec<_> = (0..m).map(|_| problem.decision_variable()).collect();
    let sddd: Vec<_> = (0..seg).map(|_| problem.decision_variable()).collect();
    let dt: Vec<_> = (0..seg).map(|_| problem.decision_variable()).collect();

    let tcp_active = deriv.has_tcp() && !constraints.tcp.is_disabled();

    for k in 0..m {
        let sd_k = sd[k];
        subject_to!(problem, sd_k >= 0.0);

        if tcp_active && constraints.tcp.v_max.is_finite() && constraints.tcp.v_max > 0.0 {
            let pp = &deriv.pp[k];
            let pp_norm_sq = pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2];
            if pp_norm_sq > 1e-18 {
                let upper = constraints.tcp.v_max / pp_norm_sq.sqrt();
                subject_to!(problem, sd_k <= upper);
            }
        }

        for j in lock..N {
            let qp_j = deriv.qp[k][j];
            if qp_j.abs() < 1e-12 {
                continue;
            }
            let v_max = constraints.joint.v_max.0[j];
            if v_max.is_finite() && v_max > 0.0 {
                let upper = v_max / qp_j.abs();
                subject_to!(problem, sd_k <= upper);
            }
        }
    }

    for k in 0..seg {
        let dt_k = dt[k];
        subject_to!(problem, dt_k >= 1e-6);
    }

    for k in 0..seg {
        let sd_k = sd[k];
        let sd_k1 = sd[k + 1];
        let sdd_k = sdd[k];
        let sdd_k1 = sdd[k + 1];
        let sddd_k = sddd[k];
        let dt_k = dt[k];
        let ds_k = deriv.ds[k];

        let rhs_sd = sd_k + sdd_k * dt_k + 0.5 * sddd_k * dt_k * dt_k;
        subject_to!(problem, sd_k1 == rhs_sd);

        let rhs_sdd = sdd_k + sddd_k * dt_k;
        subject_to!(problem, sdd_k1 == rhs_sdd);

        let rhs_ds = sd_k * dt_k
            + 0.5 * sdd_k * dt_k * dt_k
            + (1.0 / 6.0) * sddd_k * dt_k * dt_k * dt_k;
        subject_to!(problem, rhs_ds == ds_k);
    }

    {
        let sd_0 = sd[0];
        let sdd_0 = sdd[0];
        let sd_f = sd[m - 1];
        let sdd_f = sdd[m - 1];
        subject_to!(problem, sd_0 == start.sd);
        subject_to!(problem, sdd_0 == start.sdd);
        subject_to!(problem, sd_f == end.sd);
        subject_to!(problem, sdd_f == end.sdd);
    }

    for k in 0..m {
        let sd_k = sd[k];
        let sdd_k = sdd[k];
        let seg_idx = if k < seg { k } else { seg - 1 };
        let sddd_k = sddd[seg_idx];

        for j in lock..N {
            let qp_j = deriv.qp[k][j];
            let qpp_j = deriv.qpp[k][j];
            let qppp_j = deriv.qppp[k][j];
            let v_max = constraints.joint.v_max.0[j];
            let a_max = constraints.joint.a_max.0[j];
            let j_max = constraints.joint.j_max.0[j];

            if v_max.is_finite() && v_max > 0.0 && qp_j.abs() > 1e-12 {
                let expr = qp_j * sd_k;
                subject_to!(problem, expr <= v_max);
                let neg = -qp_j * sd_k;
                subject_to!(problem, neg <= v_max);
            }

            if a_max.is_finite() && a_max > 0.0 {
                let expr = qpp_j * sd_k * sd_k + qp_j * sdd_k;
                subject_to!(problem, expr <= a_max);
                let neg = -qpp_j * sd_k * sd_k - qp_j * sdd_k;
                subject_to!(problem, neg <= a_max);
            }

            if j_max.is_finite() && j_max > 0.0 {
                let expr = qppp_j * sd_k * sd_k * sd_k
                    + 3.0 * qpp_j * sd_k * sdd_k
                    + qp_j * sddd_k;
                subject_to!(problem, expr <= j_max);
                let neg = -qppp_j * sd_k * sd_k * sd_k
                    - 3.0 * qpp_j * sd_k * sdd_k
                    - qp_j * sddd_k;
                subject_to!(problem, neg <= j_max);
            }
        }

        if tcp_active && constraints.tcp.a_max.is_finite() && constraints.tcp.a_max > 0.0 {
            let a_bound_sq = constraints.tcp.a_max * constraints.tcp.a_max;
            let ppp = &deriv.ppp[k];
            let pp = &deriv.pp[k];
            let c0 = ppp[0] * sd_k * sd_k + pp[0] * sdd_k;
            let c1 = ppp[1] * sd_k * sd_k + pp[1] * sdd_k;
            let c2 = ppp[2] * sd_k * sd_k + pp[2] * sdd_k;
            let sum_sq = c0 * c0 + c1 * c1 + c2 * c2;
            subject_to!(problem, sum_sq <= a_bound_sq);
        }

        if tcp_active && constraints.tcp.j_max.is_finite() && constraints.tcp.j_max > 0.0 {
            let j_bound_sq = constraints.tcp.j_max * constraints.tcp.j_max;
            let pppp = &deriv.pppp[k];
            let ppp = &deriv.ppp[k];
            let pp = &deriv.pp[k];
            let c0 = pppp[0] * sd_k * sd_k * sd_k
                + 3.0 * ppp[0] * sd_k * sdd_k
                + pp[0] * sddd_k;
            let c1 = pppp[1] * sd_k * sd_k * sd_k
                + 3.0 * ppp[1] * sd_k * sdd_k
                + pp[1] * sddd_k;
            let c2 = pppp[2] * sd_k * sd_k * sd_k
                + 3.0 * ppp[2] * sd_k * sdd_k
                + pp[2] * sddd_k;
            let sum_sq = c0 * c0 + c1 * c1 + c2 * c2;
            subject_to!(problem, sum_sq <= j_bound_sq);
        }
    }

    let mut total = dt[0];
    for k in 1..seg {
        total = total + dt[k];
    }
    problem.minimize(total);

    apply_initial_guess(&sd, &sdd, &sddd, &dt, deriv, constraints, start, end);

    let iter_counter = Arc::new(AtomicI32::new(0));
    let ic = iter_counter.clone();
    problem.add_callback(move |_info| {
        ic.fetch_add(1, Ordering::Relaxed);
        false
    });

    let mut options = Options::default()
        .tolerance(constraints.solver.tolerance)
        .max_iterations(constraints.solver.max_iterations)
        .diagnostics(constraints.solver.diagnostics);
    if let Some(t) = constraints.solver.timeout {
        options = options.timeout(t);
    }

    let t0 = Instant::now();
    let status_raw = problem.solve_status(options);
    let solve_time = t0.elapsed();
    let status = SolveStatus::from(status_raw);
    let iterations = iter_counter.load(Ordering::Relaxed);

    let sd_vals: Vec<f64> = sd.iter().map(|v| v.value()).collect();
    let sdd_vals: Vec<f64> = sdd.iter().map(|v| v.value()).collect();
    let sddd_vals: Vec<f64> = sddd.iter().map(|v| v.value()).collect();
    let dt_vals: Vec<Duration> = dt
        .iter()
        .map(|v| Duration::from_secs_f64(v.value().max(0.0)))
        .collect();

    Ok(Solution {
        sd: sd_vals,
        sdd: sdd_vals,
        sddd: sddd_vals,
        dt: dt_vals,
        status,
        iterations,
        solve_time,
    })
}

fn apply_initial_guess<'a, const N: usize>(
    sd: &[hafgufa::Variable<'a>],
    sdd: &[hafgufa::Variable<'a>],
    sddd: &[hafgufa::Variable<'a>],
    dt: &[hafgufa::Variable<'a>],
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
    start: ProjectedBoundary,
    end: ProjectedBoundary,
) {
    let m = deriv.num_waypoints();
    let seg = deriv.num_segments();
    let lock = constraints.locked_prefix.min(N);

    let mut sd_guess = vec![0.0_f64; m];
    for k in 0..m {
        let mut cap = f64::INFINITY;
        for j in lock..N {
            let q = deriv.qp[k][j].abs();
            if q > 1e-9 {
                let v = constraints.joint.v_max.0[j];
                if v.is_finite() && v > 0.0 {
                    let bound = v / q;
                    if bound < cap {
                        cap = bound;
                    }
                }
            }
        }
        if deriv.has_tcp()
            && constraints.tcp.v_max.is_finite()
            && constraints.tcp.v_max > 0.0
        {
            let pp = &deriv.pp[k];
            let pn_sq = pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2];
            if pn_sq > 1e-18 {
                let bound = constraints.tcp.v_max / pn_sq.sqrt();
                if bound < cap {
                    cap = bound;
                }
            }
        }
        if !cap.is_finite() {
            cap = 1.0;
        }
        sd_guess[k] = (cap * 0.7).max(1e-3);
    }
    sd_guess[0] = start.sd.max(1e-6);
    sd_guess[m - 1] = end.sd.max(1e-6);

    let mut sdd_guess = vec![0.0_f64; m];
    sdd_guess[0] = start.sdd;
    sdd_guess[m - 1] = end.sdd;

    let sddd_guess = vec![0.0_f64; seg];

    let mut dt_guess = vec![0.0_f64; seg];
    for k in 0..seg {
        let sd_avg = 0.5 * (sd_guess[k] + sd_guess[k + 1]);
        dt_guess[k] = (deriv.ds[k] / sd_avg.max(1e-6)).max(1e-5);
    }

    for k in 0..m {
        sd[k].set_value(sd_guess[k]);
        sdd[k].set_value(sdd_guess[k]);
    }
    for k in 0..seg {
        sddd[k].set_value(sddd_guess[k]);
        dt[k].set_value(dt_guess[k]);
    }
}
