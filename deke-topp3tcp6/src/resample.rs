use std::time::Duration;

use deke_types::SRobotQ;

use crate::nlp::Solution;
use crate::path_derivatives::PathDerivatives;

/// Resamples the solver output to a uniform-time grid.
///
/// Each output sample at time `t = i · dt_out` is placed on the path by (1) finding the segment
/// whose time interval contains `t`, (2) integrating `s` over the segment from its start via the
/// closed-form constant-jerk expression, and (3) linearly interpolating between the
/// segment's endpoint waypoints at the resulting parameter.
pub fn resample_to_uniform<const N: usize>(
    solution: &Solution,
    deriv: &PathDerivatives<N>,
    dt_out: Duration,
) -> (Duration, Vec<SRobotQ<N, f64>>) {
    let m = deriv.num_waypoints();
    let seg = deriv.num_segments();
    assert!(m >= 2 && seg >= 1);

    let dt_out_secs = dt_out.as_secs_f64().max(1e-9);

    let mut cum = Vec::with_capacity(m);
    cum.push(0.0_f64);
    for k in 0..seg {
        let t = *cum.last().unwrap() + solution.dt[k].as_secs_f64();
        cum.push(t);
    }
    let total_secs = *cum.last().unwrap();
    let n_samples = ((total_secs / dt_out_secs).ceil() as usize).max(1) + 1;

    let mut out = Vec::with_capacity(n_samples);
    let mut seg_idx = 0;

    for i in 0..n_samples {
        let t_sample = (i as f64 * dt_out_secs).min(total_secs);

        while seg_idx + 1 < seg && cum[seg_idx + 1] < t_sample {
            seg_idx += 1;
        }

        let tau = (t_sample - cum[seg_idx]).max(0.0);
        let sd = solution.sd[seg_idx];
        let sdd = solution.sdd[seg_idx];
        let sddd = solution.sddd[seg_idx];
        let ds = deriv.ds[seg_idx];

        let s_local = sd * tau + 0.5 * sdd * tau * tau + (1.0 / 6.0) * sddd * tau * tau * tau;
        let u = (s_local / ds).clamp(0.0, 1.0);

        let a = deriv.waypoints[seg_idx];
        let b = deriv.waypoints[seg_idx + 1];
        out.push(a.interpolate(&b, u));
    }

    if let Some(last) = out.last_mut() {
        *last = *deriv.waypoints.last().unwrap();
    }

    (Duration::from_secs_f64(total_secs.max(0.0)), out)
}
