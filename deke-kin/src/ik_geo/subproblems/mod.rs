//! IK-geo subproblems 1–6.

pub(crate) mod auxiliary;

use glam::{DMat2, DMat3, DVec2, DVec3, DVec4};
use num_complex::Complex64;

use crate::ik_geo::math::Mat3x2;
use crate::ik_geo::solutionset::{SolutionSet2, SolutionSet4};

use auxiliary::{
    cone_polynomials, solve_quartic_roots, solve_two_ellipse_numeric, vec_convolve_3,
    vec_self_convolve_2, vec_self_convolve_3,
};

/// Length-7 joint vector type alias retained for callers of the old ik-geo
/// surface (e.g. a future `kuka_r800_fixed_q3`). reaik-internal callers should
/// prefer [`deke_types::SRobotQ<7, f64>`].
pub type Vector7<T> = [T; 7];

/// Solve for `theta` such that `rot(k, theta) * p1 = p2`. Returns `None` when
/// no exact solution exists (i.e. the rotation cannot map `p1` onto `p2`).
pub fn subproblem1(p1: &DVec3, p2: &DVec3, k: &DVec3) -> Option<f64> {
    if (p1.length() - p2.length()).abs() > 1e-6 || (k.dot(*p1) - k.dot(*p2)).abs() > 1e-6 {
        return None;
    }
    let a = Mat3x2::perp_basis(*k, *p1);
    let x = a.transpose_mul_vec3(*p2);
    Some(x.x.atan2(x.y))
}

/// Solve for `(theta1, theta2)` such that `rot(k1, theta1) * p1 = rot(k2, theta2) * p2`.
/// May return 0, 1, or 2 solutions; returns `Zero` when no exact solution exists.
pub fn subproblem2(
    p1: &DVec3,
    p2: &DVec3,
    k1: &DVec3,
    k2: &DVec3,
) -> SolutionSet2<(f64, f64)> {
    if (p1.length() - p2.length()).abs() > 1e-8 {
        return SolutionSet2::Zero;
    }
    let p1_norm = p1.normalize();
    let p2_norm = p2.normalize();

    let theta1 = subproblem4(k2, &p1_norm, k1, k2.dot(p2_norm));
    let theta2 = subproblem4(k1, &p2_norm, k2, k1.dot(p1_norm));

    if theta1.size() == 0 || theta2.size() == 0 {
        return SolutionSet2::Zero;
    }

    if theta1.size() > 1 || theta2.size() > 1 {
        let (t1a, t1b) = theta1.duplicated().expect_two();
        let (t2a, t2b) = theta2.duplicated().expect_two();
        SolutionSet2::Two((t1a, t2b), (t1b, t2a))
    } else {
        SolutionSet2::One((theta1.expect_one(), theta2.expect_one()))
    }
}

/// `p0 + rot(k1, theta1) * p1 = rot(k2, theta2) * p2`, assuming a unique solution.
pub fn subproblem2extended(
    p0: &DVec3,
    p1: &DVec3,
    p2: &DVec3,
    k1: &DVec3,
    k2: &DVec3,
) -> (f64, f64) {
    let a1 = Mat3x2::perp_basis(*k1, *p1);
    let a2 = Mat3x2::perp_basis(*k2, *p2);
    let a2_neg = -a2;
    let kxp1 = a1.c0;
    let kxp2 = a2.c0;
    // a (Mat3x4) is [a1.c0, a1.c1, -a2.c0, -a2.c1]. We compute
    // x_ls = a.T * (aat_inv * p) ∈ ℝ⁴ directly via four dot-products.
    let p = -*k1 * k1.dot(*p1) + *k2 * k2.dot(*p2) - *p0;

    let radius1_sq = kxp1.length_squared();
    let radius2_sq = kxp2.length_squared();
    let alpha = radius1_sq / (radius1_sq + radius2_sq);
    let beta = radius2_sq / (radius1_sq + radius2_sq);
    let m_inv = DMat3::IDENTITY + outer(*k1, *k1) * (alpha / (1.0 - alpha));
    let minv_k2 = m_inv * *k2;
    let k2t_minv_k2 = k2.dot(minv_k2);
    let aat_inv = (m_inv + outer(minv_k2, minv_k2) * (beta / (1.0 - k2t_minv_k2 * beta)))
        * (1.0 / (radius1_sq + radius2_sq));

    let aat_inv_p = aat_inv * p;
    let x_ls = DVec4::new(
        a1.c0.dot(aat_inv_p),
        a1.c1.dot(aat_inv_p),
        a2_neg.c0.dot(aat_inv_p),
        a2_neg.c1.dot(aat_inv_p),
    );

    // a_perp_tilde = (4×3 stacked pinv_a1 rows / r1, pinv_a2 rows / r2) * n_sym.
    let n_sym = k1.cross(*k2);
    let inv_r1 = 1.0 / radius1_sq;
    let inv_r2 = 1.0 / radius2_sq;
    let a_perp_tilde = DVec4::new(
        a1.c0.dot(n_sym) * inv_r1,
        a1.c1.dot(n_sym) * inv_r1,
        a2.c0.dot(n_sym) * inv_r2,
        a2.c1.dot(n_sym) * inv_r2,
    );

    let xls_a = DVec2::new(x_ls.x, x_ls.y);
    let xls_b = DVec2::new(x_ls.z, x_ls.w);
    let apt_a = DVec2::new(a_perp_tilde.x, a_perp_tilde.y);
    let apt_b = DVec2::new(a_perp_tilde.z, a_perp_tilde.w);

    let num = (xls_b.length_squared() - 1.0) * apt_a.length_squared()
        - (xls_a.length_squared() - 1.0) * apt_b.length_squared();
    let den = 2.0
        * (xls_a.dot(apt_a) * apt_b.length_squared()
            - xls_b.dot(apt_b) * apt_a.length_squared());
    let xi = num / den;

    let sc = x_ls + a_perp_tilde * xi;
    (sc.x.atan2(sc.y), sc.z.atan2(sc.w))
}

/// `|| rot(k, theta) * p1 - p2 || = d`. Returns up to two `theta`s, or `Zero`
/// when no exact solution exists.
pub fn subproblem3(p1: &DVec3, p2: &DVec3, k: &DVec3, d: f64) -> SolutionSet2<f64> {
    let a_1 = Mat3x2::perp_basis(*k, *p1);
    let kxp = a_1.c0;
    let proj = *p1 - a_1.c1; // = k * (k·p1) since a_1.c1 = p1 - k(k·p1)
    let a = a_1.transpose_mul_vec3(-2.0 * *p2);
    let norm_a_sq = a.length_squared();
    let norm_a = a.length();

    let b = d * d - (*p2 - proj).length_squared() - kxp.length_squared();

    let x_ls = a_1.transpose_mul_vec3(-2.0 * *p2 * b / norm_a_sq);

    if x_ls.length_squared() > 1.0 {
        return SolutionSet2::Zero;
    }

    let xi = (1.0 - b * b / norm_a_sq).sqrt();
    let a_perp_tilde = DVec2::new(a.y, -a.x);
    let a_perp = a_perp_tilde / norm_a;
    let sc_1 = x_ls + a_perp * xi;
    let sc_2 = x_ls - a_perp * xi;
    SolutionSet2::Two(sc_1.x.atan2(sc_1.y), sc_2.x.atan2(sc_2.y))
}

/// `h.T * rot(k, theta) * p = d`. Up to two solutions, or `Zero` when no exact
/// solution exists.
pub fn subproblem4(h: &DVec3, p: &DVec3, k: &DVec3, d: f64) -> SolutionSet2<f64> {
    let a_1 = Mat3x2::perp_basis(*k, *p);
    let a = a_1.transpose_mul_vec3(*h);

    let b = d - h.dot(*k) * k.dot(*p);
    let norm_a_2 = a.length_squared();
    let x_ls = a * b;

    if norm_a_2 > b * b {
        let xi = (norm_a_2 - b * b).sqrt();
        let a_perp_tilde = DVec2::new(a.y, -a.x);
        let sc_1 = x_ls + a_perp_tilde * xi;
        let sc_2 = x_ls - a_perp_tilde * xi;
        SolutionSet2::Two(sc_1.x.atan2(sc_1.y), sc_2.x.atan2(sc_2.y))
    } else {
        SolutionSet2::Zero
    }
}

/// `p0 + rot(k1, t1) * p1 = rot(k2, t2) * (p2 + rot(k3, t3) * p3)`. Up to 4 solutions.
pub fn subproblem5(
    p0: &DVec3,
    p1: &DVec3,
    p2: &DVec3,
    p3: &DVec3,
    k1: &DVec3,
    k2: &DVec3,
    k3: &DVec3,
) -> SolutionSet4<(f64, f64, f64)> {
    const EPSILON: f64 = 1e-6;

    let mut theta = Vec::with_capacity(8);

    let p1_s = *p0 + *k1 * k1.dot(*p1);
    let p3_s = *p2 + *k3 * k3.dot(*p3);

    let delta1 = k2.dot(p1_s);
    let delta3 = k2.dot(p3_s);

    let (p_1, r_1) = cone_polynomials(p0, k1, p1, &p1_s, k2);
    let (p_3, r_3) = cone_polynomials(p2, k3, p3, &p3_s, k2);

    let p_13 = [p_1[0] - p_3[0], p_1[1] - p_3[1]];
    let p_13_sq = vec_self_convolve_2(&p_13);

    let rhs = [
        r_3[0] - r_1[0] - p_13_sq[0],
        r_3[1] - r_1[1] - p_13_sq[1],
        r_3[2] - r_1[2] - p_13_sq[2],
    ];

    let conv1 = vec_self_convolve_3(&rhs);
    let conv2 = vec_convolve_3(&p_13_sq, &r_1);
    let eqn_real = [
        conv1[0] - 4.0 * conv2[0],
        conv1[1] - 4.0 * conv2[1],
        conv1[2] - 4.0 * conv2[2],
        conv1[3] - 4.0 * conv2[3],
        conv1[4] - 4.0 * conv2[4],
    ];

    let all_roots = solve_quartic_roots(&[
        Complex64::new(eqn_real[0], 0.0),
        Complex64::new(eqn_real[1], 0.0),
        Complex64::new(eqn_real[2], 0.0),
        Complex64::new(eqn_real[3], 0.0),
        Complex64::new(eqn_real[4], 0.0),
    ]);

    let h_vec: Vec<f64> = all_roots
        .iter()
        .filter(|c| c.im.abs() < EPSILON)
        .map(|c| c.re)
        .collect();

    let a_1 = Mat3x2::perp_basis(*k1, *p1);
    let a_3 = Mat3x2::perp_basis(*k3, *p3);

    let signs_1 = [1.0, 1.0, -1.0, -1.0];
    let signs_3 = [1.0, -1.0, 1.0, -1.0];

    for &h in h_vec.iter() {
        let a1t_k2 = a_1.transpose_mul_vec3(*k2);
        let a3t_k2 = a_3.transpose_mul_vec3(*k2);

        let const_1 = a1t_k2 * (h - delta1);
        let const_3 = a3t_k2 * (h - delta3);

        let hd1 = h - delta1;
        let hd3 = h - delta3;

        let sq1 = a1t_k2.length_squared() - hd1 * hd1;
        if sq1 < 0.0 {
            continue;
        }
        let sq3 = a3t_k2.length_squared() - hd3 * hd3;
        if sq3 < 0.0 {
            continue;
        }

        // 90° rotation: J·v = (v.y, -v.x).
        let pm_1 = DVec2::new(a1t_k2.y, -a1t_k2.x) * sq1.sqrt();
        let pm_3 = DVec2::new(a3t_k2.y, -a3t_k2.x) * sq3.sqrt();

        let a1t_k2_norm_sq = a1t_k2.length_squared();
        let a3t_k2_norm_sq = a3t_k2.length_squared();

        for (&sign_1, &sign_3) in signs_1.iter().zip(signs_3.iter()) {
            let sc1 = (const_1 + pm_1 * sign_1) / a1t_k2_norm_sq;
            let sc3 = (const_3 + pm_3 * sign_3) / a3t_k2_norm_sq;

            let v1 = a_1.mul_vec2(sc1) + p1_s;
            let v3 = a_3.mul_vec2(sc3) + p3_s;

            if ((v1 - *k2 * h).length() - (v3 - *k2 * h).length()).abs() < 1e-6 {
                if let Some(theta2_value) = subproblem1(&v3, &v1, k2) {
                    theta.push((sc1.x.atan2(sc1.y), theta2_value, sc3.x.atan2(sc3.y)));
                }
            }
        }
    }

    reduced_solutionset(theta)
}

/// Trim a >4-solution set down to the 4 most "spread" by ordering each pick by
/// gap to its predecessor; preserves the original ik-geo selection heuristic.
fn reduced_solutionset(mut solutions: Vec<(f64, f64, f64)>) -> SolutionSet4<(f64, f64, f64)> {
    if solutions.len() <= 4 {
        return SolutionSet4::from_vec(&solutions);
    }

    solutions.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut ranked = Vec::with_capacity(solutions.len());
    let mut last = f64::INFINITY;
    for s in solutions {
        let delta = s.0 - last;
        let ordering = 1.0 / (delta * delta);
        ranked.push((ordering, s));
        last = s.0;
    }
    ranked.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    SolutionSet4::Four(ranked[0].1, ranked[1].1, ranked[2].1, ranked[3].1)
}

/// `h_i.T * rot(k_i, t_a) * p_i + h_{i+1}.T * rot(k_{i+1}, t_b) * p_{i+1} = d_*`
/// for two equations (i=0 → d1, i=2 → d2). Up to 4 `(t_a, t_b)` solutions.
pub fn subproblem6(
    h: &[DVec3; 4],
    k: &[DVec3; 4],
    p: &[DVec3; 4],
    d1: f64,
    d2: f64,
) -> SolutionSet4<(f64, f64)> {
    let a_arr = [
        Mat3x2::perp_basis(k[0], p[0]),
        Mat3x2::perp_basis(k[1], p[1]),
        Mat3x2::perp_basis(k[2], p[2]),
        Mat3x2::perp_basis(k[3], p[3]),
    ];

    let h0_a0 = a_arr[0].transpose_mul_vec3(h[0]);
    let h1_a1 = a_arr[1].transpose_mul_vec3(h[1]);
    let h2_a2 = a_arr[2].transpose_mul_vec3(h[2]);
    let h3_a3 = a_arr[3].transpose_mul_vec3(h[3]);

    // The 2×4 matrix `A`. Each row absorbs the two coefficient pairs from one equation.
    let row0 = DVec4::new(h0_a0.x, h0_a0.y, h1_a1.x, h1_a1.y);
    let row1 = DVec4::new(h2_a2.x, h2_a2.y, h3_a3.x, h3_a3.y);

    let b = DVec2::new(
        d1 - h[0].dot(k[0]) * k[0].dot(p[0]) - h[1].dot(k[1]) * k[1].dot(p[1]),
        d2 - h[2].dot(k[2]) * k[2].dot(p[2]) - h[3].dot(k[3]) * k[3].dot(p[3]),
    );

    // Min-norm solution + null-space basis via (A Aᵀ)⁻¹ instead of QR.
    let m00 = row0.dot(row0);
    let m01 = row0.dot(row1);
    let m11 = row1.dot(row1);
    let det = m00 * m11 - m01 * m01;
    let inv_det = 1.0 / det;
    let mi00 = m11 * inv_det;
    let mi01 = -m01 * inv_det;
    let mi11 = m00 * inv_det;

    let u = DVec2::new(mi00 * b.x + mi01 * b.y, mi01 * b.x + mi11 * b.y);
    let x_min = row0 * u.x + row1 * u.y;

    // Null projector: N e = e - Aᵀ (A Aᵀ)⁻¹ A e.
    let project = |e: DVec4| {
        let ae = DVec2::new(row0.dot(e), row1.dot(e));
        let u = DVec2::new(mi00 * ae.x + mi01 * ae.y, mi01 * ae.x + mi11 * ae.y);
        e - row0 * u.x - row1 * u.y
    };

    let candidates = [
        project(DVec4::X),
        project(DVec4::Y),
        project(DVec4::Z),
        project(DVec4::W),
    ];

    // Pick the largest-norm projected vector as q3, then the largest residual
    // after orthogonalising against q3 as q4. Standard Gram-Schmidt on a 2D null space.
    let mut idx_max = 0;
    let mut max_norm = candidates[0].length_squared();
    for i in 1..4 {
        let n = candidates[i].length_squared();
        if n > max_norm {
            max_norm = n;
            idx_max = i;
        }
    }
    let q3 = candidates[idx_max].normalize();

    let mut q4_candidate = DVec4::ZERO;
    let mut max_perp_norm = 0.0;
    for i in 0..4 {
        if i == idx_max {
            continue;
        }
        let perp = candidates[i] - q3 * q3.dot(candidates[i]);
        let n = perp.length_squared();
        if n > max_perp_norm {
            max_perp_norm = n;
            q4_candidate = perp;
        }
    }
    let q4 = q4_candidate.normalize();

    let x_null_1 = q3;
    let x_null_2 = q4;

    // Two 2-row blocks of [x_null_1 | x_null_2]. DMat2 is column-major, so
    // each col holds one null-space vector's component pair.
    let xn1 = DMat2::from_cols_array(&[x_null_1.x, x_null_1.y, x_null_2.x, x_null_2.y]);
    let xn2 = DMat2::from_cols_array(&[x_null_1.z, x_null_1.w, x_null_2.z, x_null_2.w]);

    let xm1 = DVec2::new(x_min.x, x_min.y);
    let xm2 = DVec2::new(x_min.z, x_min.w);

    let xi_pairs = solve_two_ellipse_numeric(xm1, &xn1, xm2, &xn2).get_all();

    let mut theta = Vec::with_capacity(xi_pairs.len());
    for (xi0, xi1) in xi_pairs {
        let x = x_min + x_null_1 * xi0 + x_null_2 * xi1;
        theta.push((x.x.atan2(x.y), x.z.atan2(x.w)));
    }
    SolutionSet4::from_vec(&theta)
}

/// Outer product `a · bᵀ` as a `DMat3`.
#[inline]
fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols_array(&[
        a.x * b.x, a.y * b.x, a.z * b.x,
        a.x * b.y, a.y * b.y, a.z * b.y,
        a.x * b.z, a.y * b.z, a.z * b.z,
    ])
}
