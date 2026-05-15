//! Offline jerk-limited path-to-trajectory solver and its [`Retimer`] impl.

use core::fmt;
use std::time::Duration;

use deke_types::{
    DekeError, DekeResult, FKChain, FKScalar, Retimer, SRobotPath, SRobotQ, SRobotTraj, Validator,
};

use crate::solver::Solver;
use crate::spec::MotionSpec;
use crate::status::StepStatus;

/// Stateless offline solver. Holds only the sampling interval used to
/// resample the analytical trajectory into a discrete [`SRobotTraj`].
#[derive(Debug, Clone)]
pub struct ToppSolver<const N: usize, F: FKScalar = f32> {
    pub dt: Duration,
    _marker: core::marker::PhantomData<fn() -> F>,
}

impl<const N: usize, F: FKScalar> ToppSolver<N, F> {
    /// Create a solver that samples the produced trajectory at the given
    /// control-cycle interval.
    pub fn new(dt: Duration) -> Self {
        Self {
            dt,
            _marker: core::marker::PhantomData,
        }
    }
}

/// Diagnostic returned alongside the trajectory from [`Retimer::retime`].
#[derive(Debug, Clone)]
pub struct SolveDiagnostic {
    pub status: StepStatus,
    pub solve_micros: f64,
    pub solve_interrupted: bool,
    /// When [`MotionSpec::max_tcp_speed`] is set, the time-scaling factor
    /// applied to the trajectory to keep TCP speed under the limit. `1.0`
    /// when no scaling was needed; `>1.0` when the trajectory was slowed.
    /// `None` when no TCP limit was requested.
    pub tcp_speed_scale: Option<f64>,
    /// Peak ‖J_v·q̇‖ observed across the sampled trajectory *before*
    /// scaling. `None` when no TCP limit was requested.
    pub tcp_peak_speed: Option<f64>,
}

impl fmt::Display for SolveDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ToppSolver: status={:?}, solve_micros={:.1}, interrupted={}",
            self.status, self.solve_micros, self.solve_interrupted
        )?;
        if let (Some(scale), Some(peak)) = (self.tcp_speed_scale, self.tcp_peak_speed) {
            write!(f, ", tcp_peak={:.4}, tcp_scale={:.4}", peak, scale)?;
        }
        Ok(())
    }
}

impl<const N: usize, F: FKScalar> Retimer<N, F, ()> for ToppSolver<N, F> {
    type Diagnostic = SolveDiagnostic;
    type Constraints = MotionSpec<N, F>;

    fn retime<V: Validator<N, (), F>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, F>,
        fk: &impl FKChain<N, F>,
        _validator: &V,
        _ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, F>>, Self::Diagnostic) {
        let mut diag = SolveDiagnostic {
            status: StepStatus::Failure,
            solve_micros: 0.0,
            solve_interrupted: false,
            tcp_speed_scale: None,
            tcp_peak_speed: None,
        };

        if path.len() < 2 {
            return (Err(DekeError::PathTooShort(path.len())), diag);
        }

        // Translate the path into a `MotionSpec`-with-waypoints layout:
        //   path.first() -> current_pose
        //   path[1..n-1] -> intermediate waypoints
        //   path.last()  -> goal_pose
        let mut spec = constraints.clone();
        spec.current_pose = *path.first();
        spec.goal_pose = *path.last();
        spec.waypoint_poses.clear();
        for i in 1..(path.len() - 1) {
            if let Some(p) = path.get(i) {
                spec.waypoint_poses.push(*p);
            }
        }

        // Stack-local scratch. The allocations (`Solver`, `Plan`,
        // resample-samples buffer) are made fresh per `retime` call —
        // simpler and correct for any `<N, F>` combination on any thread,
        // at the cost of one solver allocation per call (microseconds on
        // typical paths). Callers wanting allocation-free re-runs can use
        // the [`crate::Pursuer`] which holds its own state.
        let t_solve_start = std::time::Instant::now();
        let mut solver = Solver::<N, F>::new();
        let mut plan = crate::plan::Plan::<N, F>::empty();
        let mut samples: Vec<SRobotQ<N, F>> = Vec::new();
        let status = solver.solve(&spec, &mut plan);
        diag.status = status;
        diag.solve_micros = t_solve_start.elapsed().as_secs_f64() * 1e6;

        if status.is_err() {
            return (
                Err(DekeError::RetimerFailed(format!(
                    "solver returned {:?}",
                    status
                ))),
                diag,
            );
        }

        let plan_ref: &mut crate::plan::Plan<N, F> = &mut plan;
        let samples_ref: &mut Vec<SRobotQ<N, F>> = &mut samples;

        if let Some(max_tcp_speed) = spec.max_tcp_speed {
            let solver_ref: &mut Solver<N, F> = &mut solver;
            match enforce_tcp_speed_limit_per_section(
                solver_ref,
                plan_ref,
                &mut spec,
                max_tcp_speed,
                fk,
            ) {
                Ok(TcpOutcome {
                    baseline_peak,
                    duration_scale,
                    resolve_status,
                }) => {
                    diag.tcp_peak_speed = baseline_peak.to_f64();
                    diag.tcp_speed_scale = duration_scale.to_f64();
                    // If we re-solved, surface the second solve's status.
                    if let Some(rs) = resolve_status {
                        diag.status = rs;
                        if rs.is_err() {
                            return (
                                Err(DekeError::RetimerFailed(format!(
                                    "TCP re-solve returned {:?}",
                                    rs
                                ))),
                                diag,
                            );
                        }
                    }
                }
                Err(e) => return (Err(e), diag),
            }
        }

        let traj = match resample_plan(plan_ref, self.dt, samples_ref) {
            Ok(t) => t,
            Err(e) => return (Err(e), diag),
        };

        (Ok(traj), diag)
    }
}

/// Outcome of the per-section TCP-limit enforcement pass.
struct TcpOutcome<F> {
    /// Maximum ‖J_v·q̇‖ observed across the *baseline* (pre-rescale)
    /// trajectory. Useful for verifying that the post-processor saw the
    /// excess it was meant to address.
    baseline_peak: F,
    /// Final trajectory duration divided by the baseline trajectory
    /// duration. `1.0` when no scaling was applied; `>1.0` when the
    /// trajectory was slowed (either via per-section caps or via the
    /// safety-net global scale).
    duration_scale: F,
    /// If a re-solve was triggered, its status. `None` if no section
    /// breached the limit and the baseline plan was kept verbatim.
    resolve_status: Option<crate::status::StepStatus>,
}

/// Per-section TCP-speed enforcement.
///
/// Algorithm:
/// 1. Sample the baseline plan section-by-section, recording the peak
///    Cartesian TCP speed ‖J_v(q)·q̇‖ within each section.
/// 2. For sections whose peak exceeds `max_tcp_speed`, derive a section
///    scale factor `k = peak / limit` and reduce that section's `v/a/j`
///    caps by `(1/k, 1/k², 1/k³)`. Sections under the limit get `k = 1`,
///    i.e. their effective caps are unchanged.
/// 3. Re-solve with the new per-section caps installed on the spec. The
///    waypoint solver redistributes boundary velocities so the resulting
///    trajectory is time-optimal under the tightened section caps.
/// 4. As a safety net (a single re-solve cannot in general be guaranteed
///    to hit the TCP limit exactly when boundary states shift), re-sample
///    the post-resolve plan and apply a small global time-scale if any
///    section still overshoots.
///
/// The per-joint `max_vel`/`max_accel`/`max_jerk` ceilings on `spec` are
/// upheld at every stage: the per-section caps are strictly tighter than
/// (or equal to) the global ceilings.
fn enforce_tcp_speed_limit_per_section<const N: usize, F: FKScalar, FK: FKChain<N, F>>(
    solver: &mut Solver<N, F>,
    plan: &mut crate::plan::Plan<N, F>,
    spec: &mut MotionSpec<N, F>,
    max_tcp_speed: F,
    fk: &FK,
) -> DekeResult<TcpOutcome<F>> {
    let zero = F::zero();
    let one = F::one();
    if max_tcp_speed <= zero || !max_tcp_speed.is_finite() {
        return Ok(TcpOutcome {
            baseline_peak: zero,
            duration_scale: one,
            resolve_status: None,
        });
    }
    let baseline_duration = plan.duration();
    if baseline_duration <= zero {
        return Ok(TcpOutcome {
            baseline_peak: zero,
            duration_scale: one,
            resolve_status: None,
        });
    }

    // --- Step 1: per-section peak sampling. -------------------------------
    let n_sections = plan.profiles.len();
    let mut peaks: Vec<F> = Vec::with_capacity(n_sections);
    sample_section_tcp_peaks(plan, fk, &mut peaks)?;
    let baseline_peak = peaks.iter().copied().fold(zero, |a, b| if b > a { b } else { a });

    // --- Step 2: per-section scale factors. -------------------------------
    let mut needs_resolve = false;
    let mut k_factors: Vec<F> = Vec::with_capacity(n_sections);
    for &p in peaks.iter() {
        let k = if p > max_tcp_speed {
            needs_resolve = true;
            p / max_tcp_speed
        } else {
            one
        };
        k_factors.push(k);
    }

    if !needs_resolve {
        // Plan is already within the TCP limit. Leave it untouched so we
        // don't pay a re-solve, and report duration_scale = 1.0.
        return Ok(TcpOutcome {
            baseline_peak,
            duration_scale: one,
            resolve_status: None,
        });
    }

    // --- Step 2b: do per-section caps actually help over a global scale? --
    // Uniform global scaling multiplies the whole plan by `baseline_peak /
    // limit`. Per-section caps + re-solve scale each section by its own
    // `k_i`. Per-section yields a shorter trajectory iff the sections have
    // meaningfully different `k_i` values; when peaks are roughly uniform
    // (e.g. the waypoint solver smeared a high-TCP joint across all
    // sections by choosing non-zero boundary velocities), the re-solve adds
    // cost without shortening anything — and the small post-resolve safety
    // overshoot can even make it slightly worse than global. Predict both
    // and only pay the re-solve when per-section is estimated to be ≥10 %
    // faster than global.
    let global_scale = baseline_peak / max_tcp_speed;
    let global_predicted = baseline_duration * global_scale;
    let per_section_predicted = {
        let mut sum = zero;
        for sec_idx in 0..n_sections {
            let start_t = if sec_idx == 0 {
                zero
            } else {
                plan.intermediate_durations[sec_idx - 1]
            };
            let end_t = plan.intermediate_durations[sec_idx];
            let section_baseline = end_t - start_t;
            sum = sum + section_baseline * k_factors[sec_idx];
        }
        sum
    };
    let per_section_worthwhile_threshold = F::from(0.9_f64).unwrap_or(one);
    if per_section_predicted >= global_predicted * per_section_worthwhile_threshold {
        // Sections are too uniform for per-section to pay off; fall back to
        // a single in-place global time-scale. This is exact (no overshoot,
        // no re-solve cost) and per-joint v/a/j ceilings are preserved
        // because scaling only divides magnitudes.
        plan.scale(one, global_scale);
        return Ok(TcpOutcome {
            baseline_peak,
            duration_scale: global_scale,
            resolve_status: None,
        });
    }

    // --- Step 3: install per-section caps and re-solve. -------------------
    // For each section, the effective cap entering this pass is whatever the
    // user (or a previous TCP pass) set in `per_section_max_*`, falling
    // back to the global `max_*`. We divide that effective cap by powers of
    // `k_i`, which is equivalent to instructing the solver to produce a
    // section profile time-scaled by `k_i`.
    let baseline_per_v = spec.per_section_max_vel.clone();
    let baseline_per_a = spec.per_section_max_accel.clone();
    let baseline_per_j = spec.per_section_max_jerk.clone();

    let mut new_per_v: Vec<SRobotQ<N, F>> = Vec::with_capacity(n_sections);
    let mut new_per_a: Vec<SRobotQ<N, F>> = Vec::with_capacity(n_sections);
    let mut new_per_j: Vec<SRobotQ<N, F>> = Vec::with_capacity(n_sections);
    for (i, &k) in k_factors.iter().enumerate() {
        let cur_v = baseline_per_v
            .as_ref()
            .and_then(|v| v.get(i).copied())
            .unwrap_or(spec.max_vel);
        let cur_a = baseline_per_a
            .as_ref()
            .and_then(|v| v.get(i).copied())
            .unwrap_or(spec.max_accel);
        let cur_j = baseline_per_j
            .as_ref()
            .and_then(|v| v.get(i).copied())
            .unwrap_or(spec.max_jerk);
        let k2 = k * k;
        let k3 = k2 * k;
        new_per_v.push(cur_v / k);
        new_per_a.push(cur_a / k2);
        new_per_j.push(cur_j / k3);
    }
    spec.per_section_max_vel = Some(new_per_v);
    spec.per_section_max_accel = Some(new_per_a);
    spec.per_section_max_jerk = Some(new_per_j);

    let resolve_status = solver.solve(spec, plan);
    if resolve_status.is_err() {
        return Ok(TcpOutcome {
            baseline_peak,
            duration_scale: one,
            resolve_status: Some(resolve_status),
        });
    }

    // --- Step 4: verify + (rare) safety-net global scale. -----------------
    // A single re-solve isn't guaranteed to land exactly at the limit because
    // the waypoint solver may have shifted intermediate boundary velocities
    // when it accepted the tighter caps. Re-sample the result; if any
    // section still overshoots, apply one global time-scale so the limit is
    // observed.
    let final_peak = sample_overall_tcp_peak(plan, fk, 512)?;
    let post_resolve_duration = plan.duration();
    let (final_duration, ) = if final_peak > max_tcp_speed {
        let safety_k = final_peak / max_tcp_speed;
        plan.scale(one, safety_k);
        (post_resolve_duration * safety_k,)
    } else {
        (post_resolve_duration,)
    };

    let duration_scale = if baseline_duration > zero {
        final_duration / baseline_duration
    } else {
        one
    };
    Ok(TcpOutcome {
        baseline_peak,
        duration_scale,
        resolve_status: Some(resolve_status),
    })
}

/// Sample each section of `plan` at fine resolution, computing the peak
/// Cartesian TCP speed ‖J_v(q)·q̇‖ observed within the section. The peaks
/// are appended to `out` (one entry per section, in section order).
fn sample_section_tcp_peaks<const N: usize, F: FKScalar, FK: FKChain<N, F>>(
    plan: &crate::plan::Plan<N, F>,
    fk: &FK,
    out: &mut Vec<F>,
) -> DekeResult<()> {
    let zero = F::zero();
    let one = F::one();
    let n_sections = plan.profiles.len();
    out.clear();
    out.reserve(n_sections);

    // 64 samples per section is enough to keep the discrete peak within a
    // couple of percent of the true peak for typical S-curve shapes while
    // keeping the FK Jacobian budget bounded (n_sections × 64 evaluations).
    const SAMPLES_PER_SECTION: usize = 64;
    let step_div = F::from(SAMPLES_PER_SECTION as f64).unwrap_or(one);

    for sec_idx in 0..n_sections {
        let start_t = if sec_idx == 0 {
            zero
        } else {
            plan.intermediate_durations[sec_idx - 1]
        };
        let end_t = plan.intermediate_durations[sec_idx];
        let span = end_t - start_t;
        if span <= zero {
            out.push(zero);
            continue;
        }
        let dt = span / step_div;
        let mut max_sq = zero;
        let mut t = start_t;
        for _ in 0..=SAMPLES_PER_SECTION {
            let (pose, vel, _, _, _) = plan.sample_at(t);
            let jac = fk
                .jacobian(&pose)
                .map_err(|e| -> DekeError { e.into() })?;
            let mut vx = zero;
            let mut vy = zero;
            let mut vz = zero;
            for i in 0..N {
                let qd = vel[i];
                vx = vx + jac[0][i] * qd;
                vy = vy + jac[1][i] * qd;
                vz = vz + jac[2][i] * qd;
            }
            let sq = vx * vx + vy * vy + vz * vz;
            if sq > max_sq {
                max_sq = sq;
            }
            t = t + dt;
        }
        out.push(max_sq.sqrt());
    }
    Ok(())
}

/// Sample the entire plan at uniform fine resolution and return the maximum
/// Cartesian TCP speed observed. Used post-resolve as a safety-net check.
fn sample_overall_tcp_peak<const N: usize, F: FKScalar, FK: FKChain<N, F>>(
    plan: &crate::plan::Plan<N, F>,
    fk: &FK,
    samples: usize,
) -> DekeResult<F> {
    let zero = F::zero();
    let one = F::one();
    let duration = plan.duration();
    if duration <= zero {
        return Ok(zero);
    }
    let n = samples.max(2);
    let dt = duration / F::from(n as f64).unwrap_or(one);
    let mut max_sq = zero;
    let mut t = zero;
    for _ in 0..=n {
        let (pose, vel, _, _, _) = plan.sample_at(t);
        let jac = fk
            .jacobian(&pose)
            .map_err(|e| -> DekeError { e.into() })?;
        let mut vx = zero;
        let mut vy = zero;
        let mut vz = zero;
        for i in 0..N {
            let qd = vel[i];
            vx = vx + jac[0][i] * qd;
            vy = vy + jac[1][i] * qd;
            vz = vz + jac[2][i] * qd;
        }
        let sq = vx * vx + vy * vy + vz * vz;
        if sq > max_sq {
            max_sq = sq;
        }
        t = t + dt;
    }
    Ok(max_sq.sqrt())
}

/// Resample an analytical [`Plan`] at uniform `dt` into an [`SRobotTraj`].
///
/// Samples are placed at `t = 0, dt, 2*dt, ...`, with one extra sample at the
/// exact trajectory end-time so the returned path always reaches `goal_pose`.
fn resample_plan<const N: usize, F: FKScalar>(
    plan: &crate::plan::Plan<N, F>,
    dt: Duration,
    scratch: &mut Vec<SRobotQ<N, F>>,
) -> DekeResult<SRobotTraj<N, F>> {
    let duration_f = plan.duration();
    let duration_secs = duration_f
        .to_f64()
        .ok_or_else(|| DekeError::RetimerFailed(String::from("trajectory duration not finite")))?;
    if duration_secs < 0.0 || !duration_secs.is_finite() {
        return Err(DekeError::RetimerFailed(String::from(
            "trajectory duration not finite",
        )));
    }
    let dt_secs = dt.as_secs_f64();
    if dt_secs <= 0.0 {
        return Err(DekeError::RetimerFailed(String::from(
            "control-cycle interval must be positive",
        )));
    }

    // Match the streaming control-cycle convention: emit one sample per dt
    // starting at `t=0`, advancing until and including the first cycle past
    // the trajectory's exact duration. The final sample (past `duration`) is
    // extrapolated at zero jerk from the end state, mirroring how a
    // control-loop sees the post-trajectory state.
    let n_cycles_past_end = if duration_secs > 0.0 {
        (duration_secs / dt_secs).ceil() as usize
    } else {
        0
    };
    let total = (n_cycles_past_end + 1).max(2);
    // The returned SRobotTraj owns its waypoint Vec; allocate it fresh here
    // and fill it in-place. Using `scratch` as a scratch is not a win because
    // SRobotPath requires owned data and a clone would dwarf the alloc.
    let _ = scratch;
    let mut waypoints: Vec<SRobotQ<N, F>> = vec![SRobotQ::zeros(); total];

    let dt_f = F::from(dt_secs)
        .ok_or_else(|| DekeError::RetimerFailed(String::from("scalar conversion failed")))?;
    plan.resample_positions(dt_f, total, &mut waypoints);

    Ok(SRobotTraj::new(
        dt,
        deke_types::SRobotPath::try_new(waypoints)?,
    ))
}
