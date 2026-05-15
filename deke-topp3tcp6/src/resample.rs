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
    // Snap-to-nearest when `total_secs / dt_out_secs` is within IPM-tolerance of an
    // integer — `discrete_dt = true` aims for an integer multiple, but the post-IPM
    // total inherits `solver.tolerance` (~1e-6 in seconds) of slack from convergence
    // plus FP noise from the per-segment rescale. That translates to a fractional
    // wobble of `tol / dt_out`, which is ~1e-4 at 125 Hz and ~1e-3 at 1 kHz. Without
    // this snap, `floor` randomly produces N or N−1 sample counts on equivalent
    // trajectories. With snap+floor the output grid is strictly uniform: every emitted
    // sample sits at `i·dt_out ≤ total_secs`. The downstream backward-FD check can
    // then evaluate every sample (no non-uniform last-step clamp to hide).
    let frac = total_secs / dt_out_secs;
    let rounded = frac.round();
    let n_count = if (frac - rounded).abs() < 1e-3 {
        rounded as usize
    } else {
        frac.floor() as usize
    };
    let n_samples = n_count.max(1) + 1;

    let mut out = Vec::with_capacity(n_samples);
    let mut seg_idx = 0;

    for i in 0..n_samples {
        // After snap+floor every sampled `i·dt_out` is `≤ total_secs` (with the snap
        // path putting the last sample at the snapped integer multiple). The `min`
        // is a defensive clamp for the snapped case where `total_secs` may sit a few
        // ULPs below the integer multiple due to FP — without it we'd read `τ` past
        // the last segment's `dt[seg-1]` and extrapolate the cubic.
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

    (Duration::from_secs_f64(total_secs.max(0.0)), out)
}
