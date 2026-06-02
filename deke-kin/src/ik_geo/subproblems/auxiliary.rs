//! Subproblem helpers — Rodrigues rotation, polynomial roots, and the
//! supporting linear-algebra primitives the IK subproblems share.

use glam::{DMat2, DMat3, DVec2, DVec3};
use num_complex::Complex64;

use crate::ik_geo::solutionset::SolutionSet4;

/// 3×3 rotation matrix about a **unit** axis `k` by `theta` (Rodrigues).
///
/// Callers must pass a unit-length axis; this function does not re-normalise.
/// reaik guarantees this by calling [`crate::remodel::normalise_axes`] on the
/// chain at robot construction. External ik_geo callers pre-normalise their
/// axes.
#[inline]
pub fn rot(k: &DVec3, theta: f64) -> DMat3 {
    debug_assert!(
        (k.length_squared() - 1.0).abs() < 1e-6,
        "rot expects a unit axis; got |k|^2 = {}",
        k.length_squared()
    );
    let (s, c) = theta.sin_cos();
    let t = 1.0 - c;
    DMat3::from_cols_array(&[
        // col 0
        t * k.x * k.x + c,
        t * k.x * k.y + s * k.z,
        t * k.x * k.z - s * k.y,
        // col 1
        t * k.x * k.y - s * k.z,
        t * k.y * k.y + c,
        t * k.y * k.z + s * k.x,
        // col 2
        t * k.x * k.z + s * k.y,
        t * k.y * k.z - s * k.x,
        t * k.z * k.z + c,
    ])
}

/// Rotate vector `v` about unit axis `k` by `theta` (Rodrigues, vector form).
/// Equivalent to `rot(k, theta) * v` but doesn't materialise the 3×3 matrix —
/// use when the rotation is only applied to one vector.
#[inline]
pub fn rot_vec(k: &DVec3, theta: f64, v: DVec3) -> DVec3 {
    debug_assert!(
        (k.length_squared() - 1.0).abs() < 1e-6,
        "rot_vec expects a unit axis; got |k|^2 = {}",
        k.length_squared()
    );
    let (s, c) = theta.sin_cos();
    v * c + k.cross(v) * s + *k * (k.dot(v) * (1.0 - c))
}

/// Returns `(p, r)` where `p ∈ ℝ²` (a, b coefficients) and `r ∈ ℝ³`
/// (a, b, c coefficients) of the cone-intersection polynomials.
pub fn cone_polynomials(
    p0_i: &DVec3,
    k_i: &DVec3,
    p_i: &DVec3,
    p_i_s: &DVec3,
    k2: &DVec3,
) -> ([f64; 2], [f64; 3]) {
    let ki_x_k2 = k_i.cross(*k2);
    let ki_x_ki_x_k2 = k_i.cross(ki_x_k2);
    let norm_ki_x_k2_sq = ki_x_k2.dot(ki_x_k2);

    let ki_x_pi = k_i.cross(*p_i);
    let norm_ki_x_pi_sq = ki_x_pi.dot(ki_x_pi);

    let alpha = p0_i.dot(ki_x_ki_x_k2) / norm_ki_x_k2_sq;
    let delta = k2.dot(*p_i_s);
    let beta = p0_i.dot(ki_x_k2) / norm_ki_x_k2_sq;

    let p_const = norm_ki_x_pi_sq + p_i_s.length_squared() + 2.0 * alpha * delta;
    let p = [-2.0 * alpha, p_const];

    let r_coeff = (2.0 * beta).powi(2);
    let r = [
        -1.0 * r_coeff,
        2.0 * delta * r_coeff,
        (-delta * delta + norm_ki_x_pi_sq * norm_ki_x_k2_sq) * r_coeff,
    ];

    (p, r)
}

/// Convolution of a length-2 polynomial with itself. Coefficients are in
/// descending order of power (highest first).
#[inline]
pub fn vec_self_convolve_2(v: &[f64; 2]) -> [f64; 3] {
    let (a, b) = (v[0], v[1]);
    [a * a, 2.0 * a * b, b * b]
}

/// Convolution of a length-3 polynomial with itself.
#[inline]
pub fn vec_self_convolve_3(v: &[f64; 3]) -> [f64; 5] {
    let (a, b, c) = (v[0], v[1], v[2]);
    [a * a, 2.0 * a * b, 2.0 * a * c + b * b, 2.0 * b * c, c * c]
}

/// Convolution of two length-3 polynomials.
#[inline]
pub fn vec_convolve_3(v1: &[f64; 3], v2: &[f64; 3]) -> [f64; 5] {
    let (a, b, c) = (v1[0], v1[1], v1[2]);
    let (x, y, z) = (v2[0], v2[1], v2[2]);
    [
        a * x,
        b * x + a * y,
        a * z + b * y + c * x,
        b * z + c * y,
        c * z,
    ]
}

const POLY_EPS: f64 = 1e-12;

#[inline]
fn cabs(z: Complex64) -> f64 {
    (z.re * z.re + z.im * z.im).sqrt()
}

/// Principal complex cube root.
#[inline]
fn ccbrt(z: Complex64) -> Complex64 {
    let r = cabs(z);
    let theta = z.im.atan2(z.re);
    let r3 = r.cbrt();
    let t3 = theta / 3.0;
    Complex64::new(r3 * t3.cos(), r3 * t3.sin())
}

/// Solve `p[0] x⁴ + p[1] x³ + p[2] x² + p[3] x + p[4] = 0`. Degenerate
/// leading coefficients delegate to the cubic / quadratic solver.
pub fn solve_quartic_roots(p: &[Complex64; 5]) -> Vec<Complex64> {
    let a = p[0];
    let b = p[1];
    let c = p[2];
    let d = p[3];
    let e = p[4];

    if cabs(a) < POLY_EPS {
        return solve_cubic_roots(&[p[1], p[2], p[3], p[4]]);
    }

    let p1 = 2.0 * c * c * c - 9.0 * b * c * d + 27.0 * a * d * d + 27.0 * b * b * e
        - 72.0 * a * c * e;
    let q1 = c * c - 3.0 * b * d + 12.0 * a * e;
    let p2 = p1 + (-4.0 * q1 * q1 * q1 + p1 * p1).sqrt();
    let q2 = ccbrt(p2 / 2.0);
    let p3 = q1 / (3.0 * a * q2) + q2 / (3.0 * a);
    let p4 = ((b * b) / (4.0 * a * a) - (2.0 * c) / (3.0 * a) + p3).sqrt();
    let p5 = (b * b) / (2.0 * a * a) - (4.0 * c) / (3.0 * a) - p3;
    let p6 = (-(b * b * b) / (a * a * a) + (4.0 * b * c) / (a * a) - (8.0 * d) / a) / (4.0 * p4);

    vec![
        -b / (4.0 * a) - p4 / 2.0 - (p5 - p6).sqrt() / 2.0,
        -b / (4.0 * a) - p4 / 2.0 + (p5 - p6).sqrt() / 2.0,
        -b / (4.0 * a) + p4 / 2.0 - (p5 + p6).sqrt() / 2.0,
        -b / (4.0 * a) + p4 / 2.0 + (p5 + p6).sqrt() / 2.0,
    ]
}

/// Solve `p[0] x³ + p[1] x² + p[2] x + p[3] = 0`.
pub fn solve_cubic_roots(p: &[Complex64; 4]) -> Vec<Complex64> {
    let a = p[0];
    let b = p[1];
    let c = p[2];
    let d = p[3];

    if cabs(a) < POLY_EPS {
        return solve_quadratic_roots(&[p[1], p[2], p[3]]);
    }

    // z = (-1 + sqrt(-3)) / 2 — a primitive cube root of unity.
    let z = (Complex64::new(-1.0, 0.0) + Complex64::new(-3.0, 0.0).sqrt()) / 2.0;

    let p1 = b * b - 3.0 * a * c;
    let p2 = 2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d;

    let q1 = ccbrt((p2 + (p2 * p2 - 4.0 * p1 * p1 * p1).sqrt()) / 2.0);
    let q2 = ccbrt((p2 + (p2 * p2 + 4.0 * p1 * p1 * p1).sqrt()) / 2.0);

    if cabs(q1) > POLY_EPS {
        let e0 = q1;
        let e1 = e0 * z;
        let e2 = e1 * z;
        vec![
            -1.0 / (3.0 * a) * (b + e0 + p1 / e0),
            -1.0 / (3.0 * a) * (b + e1 + p1 / e1),
            -1.0 / (3.0 * a) * (b + e2 + p1 / e2),
        ]
    } else if cabs(q2) > POLY_EPS {
        let e0 = q2;
        let e1 = e0 * z;
        let e2 = e1 * z;
        vec![
            -1.0 / (3.0 * a) * (b + e0 + p1 / e0),
            -1.0 / (3.0 * a) * (b + e1 + p1 / e1),
            -1.0 / (3.0 * a) * (b + e2 + p1 / e2),
        ]
    } else {
        let r = -b / (3.0 * a);
        vec![r, r, r]
    }
}

/// Solve `p[0] x² + p[1] x + p[2] = 0`.
pub fn solve_quadratic_roots(p: &[Complex64; 3]) -> Vec<Complex64> {
    let a = p[0];
    let b = p[1];
    let c = p[2];

    if cabs(a) < POLY_EPS {
        return if cabs(b) < POLY_EPS {
            Vec::new()
        } else {
            vec![-c / b]
        };
    }

    let s = (b * b - 4.0 * a * c).sqrt();
    vec![(-b + s) / (2.0 * a), (-b - s) / (2.0 * a)]
}

/// Closed-form ellipse-ellipse intersection.
///
/// Ellipses defined by
/// ```ignore
/// xm1' * xm1 + xi' * xn1' * xn1 * xi + xm1' * xn1 * xi == 1
/// xm2' * xm2 + xi' * xn2' * xn2 * xi + xm2' * xn2 * xi == 1
/// ```
/// where `xi = (xi_1, xi_2)`. Returns up to four `(x, y)` real intersections.
pub fn solve_two_ellipse_numeric(
    xm1: DVec2,
    xn1: &DMat2,
    xm2: DVec2,
    xn2: &DMat2,
) -> SolutionSet4<(f64, f64)> {
    const EPSILON: f64 = 1e-12;

    let xn1_t_xn1 = xn1.transpose() * *xn1;
    let a = xn1_t_xn1.x_axis.x;
    let b = 2.0 * xn1_t_xn1.x_axis.y;
    let c = xn1_t_xn1.y_axis.y;
    // 2·xm1ᵀ·xn1 as a row vector: each component is 2·xm1·col_i.
    let d = 2.0 * xm1.dot(xn1.x_axis);
    let e = 2.0 * xm1.dot(xn1.y_axis);
    let f = xm1.dot(xm1) - 1.0;

    let xn2_t_xn2 = xn2.transpose() * *xn2;
    let a1 = xn2_t_xn2.x_axis.x;
    let b1 = 2.0 * xn2_t_xn2.x_axis.y;
    let c1 = xn2_t_xn2.y_axis.y;
    let d1 = 2.0 * xm2.dot(xn2.x_axis);
    let e1 = 2.0 * xm2.dot(xn2.y_axis);
    let fq = xm2.dot(xm2) - 1.0;

    let z0 = f * a * d1 * d1 + a * a * fq * fq - d * a * d1 * fq + a1 * a1 * f * f
        - 2.0 * a * fq * a1 * f
        - d * d1 * a1 * f
        + a1 * d * d * fq;
    let z1 = e1 * d * d * a1 - fq * d1 * a * b - 2.0 * a * fq * a1 * e - f * a1 * b1 * d
        + 2.0 * d1 * b1 * a * f
        + 2.0 * e1 * fq * a * a
        + d1 * d1 * a * e
        - e1 * d1 * a * d
        - 2.0 * a * e1 * a1 * f
        - f * a1 * d1 * b
        + 2.0 * f * e * a1 * a1
        - fq * b1 * a * d
        - e * a1 * d1 * d
        + 2.0 * fq * b * a1 * d;
    let z2 = e1 * e1 * a * a + 2.0 * c1 * fq * a * a - e * a1 * d1 * b + fq * a1 * b * b
        - e * a1 * b1 * d
        - fq * b1 * a * b
        - 2.0 * a * e1 * a1 * e
        + 2.0 * d1 * b1 * a * e
        - c1 * d1 * a * d
        - 2.0 * a * c1 * a1 * f
        + b1 * b1 * a * f
        + 2.0 * e1 * b * a1 * d
        + e * e * a1 * a1
        - c * a1 * d1 * d
        - e1 * b1 * a * d
        + 2.0 * f * c * a1 * a1
        - f * a1 * b1 * b
        + c1 * d * d * a1
        + d1 * d1 * a * c
        - e1 * d1 * a * b
        - 2.0 * a * fq * a1 * c;
    let z3 = -2.0 * a * a1 * c * e1 + e1 * a1 * b * b + 2.0 * c1 * b * a1 * d - c * a1 * b1 * d
        + b1 * b1 * a * e
        - e1 * b1 * a * b
        - 2.0 * a * c1 * a1 * e
        - e * a1 * b1 * b
        - c1 * b1 * a * d
        + 2.0 * e1 * c1 * a * a
        + 2.0 * e * c * a1 * a1
        - c * a1 * d1 * b
        + 2.0 * d1 * b1 * a * c
        - c1 * d1 * a * b;
    let z4 = a * a * c1 * c1 - 2.0 * a * c1 * a1 * c + a1 * a1 * c * c
        - b * a * b1 * c1
        - b * b1 * a1 * c
        + b * b * a1 * c1
        + c * a * b1 * b1;

    let roots = solve_quartic_roots(&[
        Complex64::new(z4, 0.0),
        Complex64::new(z3, 0.0),
        Complex64::new(z2, 0.0),
        Complex64::new(z1, 0.0),
        Complex64::new(z0, 0.0),
    ]);

    let ys: Vec<f64> = roots
        .iter()
        .filter(|z| z.im.abs() < EPSILON)
        .map(|z| z.re)
        .collect();

    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(ys.len());
    for y in &ys {
        let y_sq = y * y;
        let num = -(a * c1 * y_sq + a * fq - a1 * c * y_sq + a * e1 * y - a1 * e * y - a1 * f);
        let den = a * b1 * y + a * d1 - a1 * b * y - a1 * d;
        if den.abs() > EPSILON {
            pairs.push((num / den, *y));
        }
    }
    SolutionSet4::from_vec(&pairs)
}
