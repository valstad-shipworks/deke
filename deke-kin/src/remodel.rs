//! Kinematic remodeling — collapse adjacent intersecting axes by re-routing
//! P vectors through their intersection point.

use glam::{DMat3, DVec3};

/// `true` if `h1` and `h2` are non-parallel and their axes intersect.
pub fn do_axes_intersect(
    h1: &DVec3,
    h2: &DVec3,
    p12: &DVec3,
    zero_threshold: f64,
    axis_intersect_threshold: f64,
) -> bool {
    let cross = h1.cross(*h2);
    cross.dot(*p12).abs() < axis_intersect_threshold && cross.length() > zero_threshold
}

/// `true` if point `p` lies on the axis defined by direction `h` and origin `p0h`.
pub fn is_point_on_axis(
    h: &DVec3,
    p0h: &DVec3,
    p: &DVec3,
    axis_intersect_threshold: f64,
) -> bool {
    let p_h = *p - *p0h;
    let proj = *h * p_h.dot(*h);
    (proj - p_h).length() < axis_intersect_threshold
}

/// Closest point on `hj` to `hk`, assuming the two axes intersect.
pub fn calc_intersection(
    hj: &DVec3,
    hk: &DVec3,
    p0j: &DVec3,
    pkj: &DVec3,
    zero_threshold: f64,
) -> DVec3 {
    let cross = hj.cross(*hk);
    let cross_sq = cross.length_squared();
    debug_assert!(
        cross_sq > zero_threshold,
        "calc_intersection called on parallel axes"
    );
    let mat = DMat3::from_cols(*pkj, *hk, cross);
    let lambda_1 = mat.determinant() / cross_sq;
    *p0j + *hj * lambda_1
}

/// Reroute P vectors through intersection points of adjacent intersecting
/// axes. Returns the new P; H is unchanged.
pub fn remodel_kinematics(
    h: &[DVec3],
    p: &[DVec3],
    zero_threshold: f64,
    axis_intersect_threshold: f64,
) -> Vec<DVec3> {
    let mut p_new = p.to_vec();
    let n_h = h.len();

    if p.len() == 7 {
        let mut p0 = DVec3::ZERO;

        let mut i = 0usize;
        while i < n_h.saturating_sub(4) {
            p0 = p0 + p_new[i];
            if do_axes_intersect(&h[i], &h[i + 1], &p[i + 1], zero_threshold, axis_intersect_threshold) {
                let p0_i1 = p0;
                let intersection = calc_intersection(&h[i], &h[i + 1], &p0_i1, &p[i + 1], zero_threshold);
                p_new[i + 1] = DVec3::ZERO;

                let mut j = i + 2;
                let mut p0j = p0_i1 + p[j - 1] + p[j];
                while j < n_h - 3
                    && is_point_on_axis(&h[j], &p0j, &intersection, axis_intersect_threshold)
                {
                    p_new[j] = DVec3::ZERO;
                    j += 1;
                    p0j = p0j + p[j];
                }
                let p_new_i = p_new[i];
                p_new[i] = intersection - (p0_i1 - p_new_i);
                p_new[j] = p0j - intersection;
                i = j - 1;
                p0 = p0j - p[j];
            }
            i += 1;
        }

        p0 = p0 + p[n_h - 4];

        let mut i = n_h - 3;
        while i < n_h - 1 {
            p0 = p0 + p_new[i];
            if do_axes_intersect(&h[i], &h[i + 1], &p[i + 1], zero_threshold, axis_intersect_threshold) {
                let p0_i1 = p0;
                let intersection = calc_intersection(
                    &h[i],
                    &h[i + 1],
                    &p0_i1,
                    &p_new[i + 1],
                    axis_intersect_threshold,
                );

                let mut j = i + 2;
                let mut p0_j_plus_1 = p0_i1 + p_new[i + 1] + p_new[j];
                p_new[i + 1] = DVec3::ZERO;
                while j < n_h
                    && is_point_on_axis(&h[j], &p0_j_plus_1, &intersection, axis_intersect_threshold)
                {
                    p_new[j] = DVec3::ZERO;
                    j += 1;
                    p0_j_plus_1 = p0_j_plus_1 + p[j];
                }

                let p_new_i = p_new[i];
                p_new[i] = intersection - (p0_i1 - p_new_i);
                p_new[j] = p0_j_plus_1 - intersection;
                i = j - 2;
                p0 = p0_j_plus_1 - p[j] - p[j - 1];
            }
            i += 1;
        }
    } else {
        let mut p0 = DVec3::ZERO;
        let mut i = 0usize;
        while i + 1 < n_h {
            p0 = p0 + p_new[i];
            if do_axes_intersect(&h[i], &h[i + 1], &p[i + 1], zero_threshold, axis_intersect_threshold) {
                let p0_i1 = p0;
                let intersection = calc_intersection(
                    &h[i],
                    &h[i + 1],
                    &p0_i1,
                    &p_new[i + 1],
                    axis_intersect_threshold,
                );

                let mut j = i + 2;
                let pi_plus_1 = p_new[i + 1];
                let pj_now = if j < p_new.len() { p_new[j] } else { DVec3::ZERO };
                let mut p0_j_plus_1 = p0_i1 + pi_plus_1 + pj_now;

                p_new[i + 1] = DVec3::ZERO;
                while j < n_h
                    && is_point_on_axis(&h[j], &p0_j_plus_1, &intersection, axis_intersect_threshold)
                {
                    p_new[j] = DVec3::ZERO;
                    j += 1;
                    if j < p.len() {
                        p0_j_plus_1 = p0_j_plus_1 + p[j];
                    }
                }

                let p_new_i = p_new[i];
                p_new[i] = intersection - (p0_i1 - p_new_i);
                if j < p_new.len() {
                    p_new[j] = p0_j_plus_1 - intersection;
                }
                i = j.saturating_sub(2);
                if j < p.len() && j >= 1 {
                    p0 = p0_j_plus_1 - p[j] - p[j - 1];
                } else {
                    p0 = p0_j_plus_1;
                }
            }
            i += 1;
        }
    }

    p_new
}

/// Normalise every entry of `h` in place.
pub fn normalise_axes(h: &mut [DVec3]) {
    for v in h.iter_mut() {
        let n = v.length();
        if n > 0.0 {
            *v = *v / n;
        }
    }
}
