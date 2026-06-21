//! Construction-time kinematics: DH→HP, forward kinematics, chain reversal,
//! fixed-axis locking, normal-vector helper. The chain is stored as
//! `Vec<DVec3>` here because `partial_joint_parametrization` shrinks it at
//! runtime; the IK hot path then copies into fixed-size arrays.

use glam::{DMat3, DMat4, DVec3};

use crate::FixedAxis;
use crate::ik_geo::subproblems::auxiliary::rot as axis_angle;

/// DH → (H, P, R_dh).
/// `H[i]` is the joint axis; `P` is the offset chain (length n+1), `P[0]` is
/// the base offset (always 0 by construction here).
///
/// Feeds the internal [`crate::Robot`] DH constructors, which are exercised by
/// the engine's test suite.
#[cfg_attr(not(test), allow(dead_code))]
pub fn dh_to_hp(alpha: &[f64], a: &[f64], d: &[f64]) -> (Vec<DVec3>, Vec<DVec3>, DMat3) {
    let n = a.len();
    let mut h = Vec::with_capacity(n);
    let mut p = Vec::with_capacity(n + 1);
    p.push(DVec3::ZERO);
    let mut r = DMat3::IDENTITY;

    for i in 0..n {
        let ca = alpha[i].cos();
        let sa = alpha[i].sin();
        // Rotation about X by alpha_i, column-major.
        let r_loc = DMat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, ca, sa, 0.0, -sa, ca]);

        let z = DVec3::Z;
        h.push(r * z);
        let off = DVec3::new(a[i], 0.0, d[i]);
        p.push(r * off);

        r *= r_loc;
    }

    (h, p, r)
}

/// Forward kinematics over runtime-sized H, P.
#[cfg_attr(not(test), allow(dead_code))]
pub fn fwdkin(h: &[DVec3], p: &[DVec3], q: &[f64]) -> DMat4 {
    debug_assert_eq!(q.len(), h.len());
    debug_assert_eq!(p.len(), h.len() + 1);

    let mut r = DMat3::IDENTITY;
    let mut pos = p[0];
    for i in 0..q.len() {
        r *= axis_angle(&h[i], q[i]);
        pos += r * p[i + 1];
    }
    let mut pose = DMat4::from_mat3(r);
    pose.w_axis.x = pos.x;
    pose.w_axis.y = pos.y;
    pose.w_axis.z = pos.z;
    pose.w_axis.w = 1.0;
    pose
}

/// Right-multiply the rotation block by `r6t`.
#[cfg_attr(not(test), allow(dead_code))]
pub fn apply_r6t(pose: DMat4, r6t: &DMat3) -> DMat4 {
    let r = DMat3::from_cols(
        pose.x_axis.truncate(),
        pose.y_axis.truncate(),
        pose.z_axis.truncate(),
    ) * *r6t;
    let mut out = DMat4::from_mat3(r);
    out.w_axis = pose.w_axis;
    out
}

/// `T⁻¹` for a homogeneous transform (rotation transpose + back-translate).
pub fn inverse_homogeneous(t: &DMat4) -> DMat4 {
    let r = DMat3::from_cols(
        t.x_axis.truncate(),
        t.y_axis.truncate(),
        t.z_axis.truncate(),
    );
    let p = DVec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z);
    let rt = r.transpose();
    let p_new = -(rt * p);
    let mut out = DMat4::from_mat3(rt);
    out.w_axis.x = p_new.x;
    out.w_axis.y = p_new.y;
    out.w_axis.z = p_new.z;
    out.w_axis.w = 1.0;
    out
}

/// Reverse a kinematic chain (used by 4R/5R/6R "reversed" classes). Angle
/// parametrisation also flips, so callers reverse each solution vector.
pub fn reverse_chain(h: &[DVec3], p: &[DVec3]) -> (Vec<DVec3>, Vec<DVec3>) {
    let h_rev = h.iter().rev().map(|v| -*v).collect();
    let p_rev = p.iter().rev().map(|v| -*v).collect();
    (h_rev, p_rev)
}

/// Unit vector numerically guaranteed to be non-parallel to `v`.
pub fn create_normal_vector(v: &DVec3) -> DVec3 {
    const THRESH: f64 = 1e-10;
    let x = DVec3::X;
    let mut hn = v.cross(x);
    if hn.length_squared() < THRESH * THRESH {
        let y = DVec3::Y;
        hn = v.cross(y);
    }
    hn.normalize()
}

/// Process locked joints by absorbing them into surrounding offsets. Walks in
/// *descending* `joint` order so later indices don't shift.
pub fn partial_joint_parametrization(
    h: &[DVec3],
    p: &[DVec3],
    fixed_axes: &[FixedAxis],
    r6t: &DMat3,
) -> (Vec<DVec3>, Vec<DVec3>, DMat3) {
    let mut h_new = h.to_vec();
    let mut p_new = p.to_vec();
    let mut r6t_new = *r6t;

    let mut sorted: Vec<FixedAxis> = fixed_axes.to_vec();
    sorted.sort_by_key(|fa| std::cmp::Reverse(fa.joint));

    for fa in sorted {
        let axis_idx = fa.joint;
        let r_fixed = axis_angle(&h_new[axis_idx], fa.angle);

        p_new[axis_idx] = p_new[axis_idx] + r_fixed * p_new[axis_idx + 1];

        let p_cols = p_new.len();
        for c in (axis_idx + 1)..(p_cols - 1) {
            p_new[c] = r_fixed * p_new[c + 1];
        }

        if axis_idx + 1 < h_new.len() {
            h_new[axis_idx] = r_fixed * h_new[axis_idx + 1];
            let h_cols = h_new.len();
            for c in (axis_idx + 1)..(h_cols - 1) {
                h_new[c] = r_fixed * h_new[c + 1];
            }
        }

        h_new.pop();
        p_new.pop();
        r6t_new = r_fixed * r6t_new;
    }

    (h_new, p_new, r6t_new)
}
