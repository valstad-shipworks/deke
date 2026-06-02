//! Minimum-norm perturbations to H/P that snap a *nearly*-classified chain to
//! an *exactly*-classified one. Currently covers the spherical-wrist case:
//! force `h[i] ∩ h[i+1]` to be exact and force `h[i+2]` through that point by
//! adjusting only `p[i+1]` and `p[i+2]`.
//!
//! Each primitive is a projection onto a single scalar linear constraint, so
//! the perturbation is the minimum-L2 change to the corresponding P vector.

use glam::DVec3;

/// Per-link perturbations applied by a snap. Magnitudes are absolute lengths
/// (metres). The sum upper-bounds the FK position error introduced, since
/// each P shift propagates rigidly along the remainder of the chain.
#[derive(Clone, Copy, Debug, Default)]
pub struct SnapReport {
    /// `|Δp[i+1]|` — the snap that forces `h[i]` and `h[i+1]` to intersect.
    pub delta_intersect: f64,
    /// `|Δp[i+2]|` — the snap that forces `h[i+2]` through the intersection.
    pub delta_through: f64,
}

impl SnapReport {
    /// Conservative bound on the FK position error (metres) introduced by the
    /// snap, summed over both link perturbations.
    pub fn estimated_error(&self) -> f64 {
        self.delta_intersect + self.delta_through
    }
}

/// Snap `p[i+1]` so the lines `h[i]` and `h[i+1]` intersect exactly. Returns
/// the displacement applied (zero if axes are parallel). The perturbation is
/// `-((h_i × h_{i+1}) · p_{i+1} / |h_i × h_{i+1}|²) · (h_i × h_{i+1})`.
pub fn snap_pair_intersect(
    h: &[DVec3],
    p: &mut [DVec3],
    i: usize,
    zero_threshold: f64,
) -> DVec3 {
    let cross = h[i].cross(h[i + 1]);
    let n2 = cross.length_squared();
    if n2 <= zero_threshold * zero_threshold {
        return DVec3::ZERO;
    }
    let delta = -(cross.dot(p[i + 1]) / n2) * cross;
    p[i + 1] += delta;
    delta
}

/// Snap `p[k]` perpendicular to `h[k]` so the line `h[k]` (passing through
/// `joint_origin`) instead passes through `target`. Returns the displacement
/// applied.
pub fn snap_axis_through_point(
    h: &[DVec3],
    p: &mut [DVec3],
    k: usize,
    joint_origin: DVec3,
    target: DVec3,
) -> DVec3 {
    let h_k = h[k].normalize();
    let offset = joint_origin - target;
    let perp = offset - h_k * offset.dot(h_k);
    let delta = -perp;
    p[k] += delta;
    delta
}

/// Compute the closest point on `h[j]` to `h[k]` assuming the two axes
/// intersect. `p0j` is a point on `h[j]`; `pkj` is the offset from that point
/// to a point on `h[k]`.
fn intersection(hj: DVec3, hk: DVec3, p0j: DVec3, pkj: DVec3) -> DVec3 {
    let cross = hj.cross(hk);
    let n2 = cross.length_squared();
    let m = glam::DMat3::from_cols(pkj, hk, cross);
    p0j + hj * (m.determinant() / n2)
}

/// Apply the two-step minimum-norm snap that turns axes `h[i], h[i+1], h[i+2]`
/// into a concurrent (spherical-wrist) triple by perturbing `p[i+1]` and
/// `p[i+2]`. Returns `None` if the wrist is degenerate (any adjacent pair is
/// parallel within `zero_threshold`).
///
/// `h` and the **partial** prefix sum `p[0] + … + p[i]` define the world-frame
/// position of joint `i`'s axis origin; the snap is invariant to the choice of
/// that prefix beyond knowing the joint-origin location.
pub fn min_snap_spherical_wrist(
    h: &[DVec3],
    p: &mut [DVec3],
    i: usize,
    zero_threshold: f64,
) -> Option<SnapReport> {
    debug_assert!(i + 2 < h.len());
    debug_assert_eq!(p.len(), h.len() + 1);

    let cross_01 = h[i].cross(h[i + 1]);
    let cross_12 = h[i + 1].cross(h[i + 2]);
    if cross_01.length_squared() <= zero_threshold * zero_threshold
        || cross_12.length_squared() <= zero_threshold * zero_threshold
    {
        return None;
    }

    let delta_p_i1 = snap_pair_intersect(h, p, i, zero_threshold);

    let mut p0_i = DVec3::ZERO;
    for k in 0..=i {
        p0_i += p[k];
    }
    let p0_i1 = p0_i + p[i + 1];
    let isect = intersection(h[i], h[i + 1], p0_i, p[i + 1]);

    let joint_origin_k = p0_i1 + p[i + 2];
    let delta_p_i2 = snap_axis_through_point(h, p, i + 2, joint_origin_k, isect);

    Some(SnapReport {
        delta_intersect: delta_p_i1.length(),
        delta_through: delta_p_i2.length(),
    })
}
