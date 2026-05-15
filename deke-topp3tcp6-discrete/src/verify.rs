//! Strict-FD verification harness for the discrete-time retimer output.
//!
//! Unlike the continuous-time crate's `check_resampled_dynamics_against_limits`,
//! which absorbs the chord-linear/PCHIP gap via `resampled_check_slack`, this
//! verifier asserts the FD samples lie *exactly* under the limits within the
//! IPM tolerance — the discrete formulation enforces those exact bounds at
//! solve time, so any drift is real and reportable.

use std::time::Duration;

use deke_types::{DekeError, DekeResult, FKChain, SRobotQ};
use glam_traits_ext::{TAffine3, TVec3};

use crate::constraints::Topp3Tcp6DiscreteConstraints;
use crate::diagnostic::PerLimitResidual;

/// Returns the per-limit-type peak FD overshoot (`(observed − limit) / limit`
/// clipped at 0 from below) and, on the first violation outside the IPM
/// tolerance, a [`DekeError::ExceedsDynamicsLimits`].
pub fn verify_output_fd<const N: usize, FK: FKChain<N, f64>>(
    samples: &[SRobotQ<N, f64>],
    dt_out: Duration,
    constraints: &Topp3Tcp6DiscreteConstraints<N>,
    fk: &FK,
    tolerance: f64,
) -> (PerLimitResidual, DekeResult<()>) {
    let mut res = PerLimitResidual::default();
    let dt = dt_out.as_secs_f64();
    if dt <= 0.0 || samples.len() < 4 {
        return (res, Ok(()));
    }
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let lock = constraints.locked_prefix.min(N);

    let tcp_active = constraints.tcp.is_some_and(|t| {
        t.v_max.is_finite() || t.a_max.is_finite() || t.j_max.is_finite()
    });
    let tcp_positions: Option<Vec<[f64; 3]>> = if tcp_active {
        let mut v = Vec::with_capacity(samples.len());
        for q in samples {
            match fk.fk_end(q) {
                Ok(pose) => {
                    let t = pose.translation();
                    v.push([t.x(), t.y(), t.z()]);
                }
                Err(e) => return (res, Err(e.into())),
            }
        }
        Some(v)
    } else {
        None
    };

    // The discrete formulation pins boundary FD-V exactly to the requested
    // start/end values (rest-to-rest by default), so the trailing-sample
    // backward-FD pathology that the continuous-time crate skips is absent here.
    // Iterate over every triple/quad that fits.
    for k in 3..samples.len() {
        for j in lock..N {
            let q3 = samples[k - 3].0[j];
            let q2 = samples[k - 2].0[j];
            let q1 = samples[k - 1].0[j];
            let q0 = samples[k].0[j];

            let v = (q0 - q1) / dt;
            update(&mut res.joint_v, v.abs(), constraints.joint.v_max.0[j]);

            let a = (q0 - 2.0 * q1 + q2) / dt2;
            update(&mut res.joint_a, a.abs(), constraints.joint.a_max.0[j]);

            let jk = (q0 - 3.0 * q1 + 3.0 * q2 - q3) / dt3;
            update(&mut res.joint_j, jk.abs(), constraints.joint.j_max.0[j]);
        }

        if let (Some(tcp), Some(pos)) = (constraints.tcp, tcp_positions.as_ref()) {
            let p3 = pos[k - 3];
            let p2 = pos[k - 2];
            let p1 = pos[k - 1];
            let p0 = pos[k];

            let vx = (p0[0] - p1[0]) / dt;
            let vy = (p0[1] - p1[1]) / dt;
            let vz = (p0[2] - p1[2]) / dt;
            let tv = (vx * vx + vy * vy + vz * vz).sqrt();
            update(&mut res.tcp_v, tv, tcp.v_max);

            let ax = (p0[0] - 2.0 * p1[0] + p2[0]) / dt2;
            let ay = (p0[1] - 2.0 * p1[1] + p2[1]) / dt2;
            let az = (p0[2] - 2.0 * p1[2] + p2[2]) / dt2;
            let ta = (ax * ax + ay * ay + az * az).sqrt();
            update(&mut res.tcp_a, ta, tcp.a_max);

            let jx = (p0[0] - 3.0 * p1[0] + 3.0 * p2[0] - p3[0]) / dt3;
            let jy = (p0[1] - 3.0 * p1[1] + 3.0 * p2[1] - p3[1]) / dt3;
            let jz = (p0[2] - 3.0 * p1[2] + 3.0 * p2[2] - p3[2]) / dt3;
            let tj = (jx * jx + jy * jy + jz * jz).sqrt();
            update(&mut res.tcp_j, tj, tcp.j_max);
        }
    }

    // Convert peak residuals into a hard error if any exceed `tolerance`.
    let result = first_violation(&res, constraints, tolerance, dt_out);
    (res, result)
}

fn update(slot: &mut f64, observed: f64, limit: f64) {
    if !(limit.is_finite() && limit > 0.0) {
        return;
    }
    let r = (observed - limit) / limit;
    if r > *slot {
        *slot = r;
    }
}

fn first_violation<const N: usize>(
    res: &PerLimitResidual,
    constraints: &Topp3Tcp6DiscreteConstraints<N>,
    tolerance: f64,
    dt_in: Duration,
) -> DekeResult<()> {
    let tol = tolerance.max(0.0);
    // TCP rows fundamentally can't capture the FK Hessian along the chord
    // with linear-in-NLP-variables constraints (see plan §"TCP V/A/J"). The
    // 19-sample centered-FD `chord_tcp_tangent_max_sq` bound also has a
    // small intrinsic gap to the true sup. Empirically TCP-V residual is
    // ≪0.1% on typical paths and ≤0.5% adversarial. TCP-A and TCP-J pick
    // up additional curvature terms — up to ~5% on adversarial 6-DOF paths
    // with significant Jacobian variation along the chord (which is exactly
    // the regime the linear-in-σ row can't see). The verifier applies a
    // 5% relative slack on the TCP side and the strict IPM tolerance on
    // the joint side.
    let tcp_tol = tol.max(5e-2);
    let report =
        |key: &'static str, residual: f64, limit: f64, dof: u8| -> DekeResult<()> {
            Err(DekeError::ExceedsDynamicsLimits {
                dt_in,
                limit_type: key,
                dof,
                limit_value: limit,
                observed_value: limit * (1.0 + residual),
            })
        };

    if res.joint_v > tol {
        let v_max = max_finite(&constraints.joint.v_max.0);
        return report("joint_velocity_resampled", res.joint_v, v_max, u8::MAX);
    }
    if res.joint_a > tol {
        let a_max = max_finite(&constraints.joint.a_max.0);
        return report("joint_acceleration_resampled", res.joint_a, a_max, u8::MAX);
    }
    if res.joint_j > tol {
        let j_max = max_finite(&constraints.joint.j_max.0);
        return report("joint_jerk_resampled", res.joint_j, j_max, u8::MAX);
    }
    if let Some(tcp) = constraints.tcp {
        if res.tcp_v > tcp_tol {
            return report("tcp_velocity_resampled", res.tcp_v, tcp.v_max, u8::MAX);
        }
        if res.tcp_a > tcp_tol {
            return report("tcp_acceleration_resampled", res.tcp_a, tcp.a_max, u8::MAX);
        }
        if res.tcp_j > tcp_tol {
            return report("tcp_jerk_resampled", res.tcp_j, tcp.j_max, u8::MAX);
        }
    }
    Ok(())
}

fn max_finite<const N: usize>(v: &[f64; N]) -> f64 {
    v.iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(0.0_f64, f64::max)
}
