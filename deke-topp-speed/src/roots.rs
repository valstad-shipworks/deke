//! Polynomial root finding for the analytical trajectory solvers.
//!
//! Faithfully follows the public `solve_cubic` / `solve_quartic` /
//! `poly_root_newton` semantics used by the upstream analytical derivations.
//! All routines are allocation-free and built around the const-generic
//! [`PositiveSet`].

use num_traits::Float;

/// Convergence tolerance used by Newton iterations.
pub const ROOT_TOLERANCE: f64 = 1e-14;

/// Ordered set of non-negative real roots with compile-time capacity.
///
/// Sorted in ascending order on [`PositiveSet::as_sorted_slice`]. Insertion
/// silently drops negatives and overflow.
#[derive(Debug, Clone, Copy)]
pub struct PositiveSet<F: Float, const CAP: usize> {
    items: [F; CAP],
    len: usize,
}

impl<F: Float, const CAP: usize> PositiveSet<F, CAP> {
    pub fn new() -> Self {
        Self {
            items: [F::zero(); CAP],
            len: 0,
        }
    }

    pub fn insert(&mut self, value: F) {
        if value < F::zero() {
            return;
        }
        if self.len >= CAP {
            return;
        }
        self.items[self.len] = value;
        self.len += 1;
    }

    /// Returns the items as an unsorted slice.
    pub fn as_slice(&self) -> &[F] {
        &self.items[..self.len]
    }

    /// Returns the items sorted in ascending order. Sorts in place if needed.
    pub fn as_sorted_slice(&mut self) -> &[F] {
        let slice = &mut self.items[..self.len];
        slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        slice
    }
}

impl<F: Float, const CAP: usize> Default for PositiveSet<F, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

/// Signed real cube root: `sgn(x) · |x|^(1/3)`.
#[inline]
pub fn signed_cbrt<F: Float>(x: F) -> F {
    // `Float::cbrt` already preserves the sign of `x`.
    x.cbrt()
}

/// Solve `a x^3 + b x^2 + c x + d = 0` for its non-negative real roots.
pub fn solve_cubic<F: Float>(mut a: F, mut b: F, mut c: F, mut d: F) -> PositiveSet<F, 3> {
    let mut result: PositiveSet<F, 3> = PositiveSet::new();
    let zero = F::zero();
    let one = F::one();
    let two = F::from(2.0).unwrap();
    let three = F::from(3.0).unwrap();
    let four = F::from(4.0).unwrap();
    let nine = F::from(9.0).unwrap();
    let twenty_seven = F::from(27.0).unwrap();
    let fifty_four = F::from(54.0).unwrap();

    // Degenerate-d shortcut from upstream: drop a near-zero constant by extracting the
    // trivial root and degrading to a quadratic.
    if a.abs() != zero && (d / a).abs() < F::epsilon() {
        if c != zero {
            result.insert(-d / c);
        } else {
            result.insert(zero);
        }
        d = c;
        c = b;
        b = a;
        a = zero;
    }

    if a == zero {
        if b == zero {
            if c != zero {
                result.insert(-d / c);
            }
            return result;
        }
        let disc2 = c * c - four * b * d;
        if disc2 >= zero {
            let inv_2b = one / (two * b);
            let sqrt_disc2 = disc2.sqrt();
            result.insert((-c + sqrt_disc2) * inv_2b);
            result.insert((-c - sqrt_disc2) * inv_2b);
        }
        return result;
    }

    let inv_a = one / a;
    let inv_a_sq = inv_a * inv_a;
    let b_sq = b * b;
    let shift = b * inv_a / three;
    let p = (a * c - b_sq / three) * inv_a_sq;
    let q = ((two * b_sq * b - nine * a * b * c) / fifty_four + a * a * d / two) * inv_a_sq * inv_a;
    let det = p * p * p / twenty_seven + q * q;

    if det > F::epsilon() {
        let sqrt_det = det.sqrt();
        let r1 = -q + sqrt_det;
        let r2 = -q - sqrt_det;
        let larger = if r1.abs() > r2.abs() { r1 } else { r2 };
        let cube_root = signed_cbrt(larger);
        result.insert(cube_root - p / (three * cube_root) - shift);
    } else if det < -F::epsilon() {
        let neg_q = -q;
        let sqrt_neg_det = (-det).sqrt();
        let pi = F::from(core::f64::consts::PI).unwrap();
        let pi_over_two = F::from(core::f64::consts::FRAC_PI_2).unwrap();
        let theta_full;
        let mag;
        if neg_q.abs() > F::epsilon() {
            let atan = (sqrt_neg_det / neg_q).atan();
            theta_full = if neg_q > zero { atan } else { atan + pi };
            mag = (neg_q * neg_q - det).sqrt();
        } else {
            theta_full = pi_over_two;
            mag = sqrt_neg_det;
        }
        let theta = theta_full / three;
        let mag_scaled = two * signed_cbrt(mag);
        let sqrt3_half = F::from(0.866_025_403_784_438_6).unwrap();
        let cos_term = theta.cos() * mag_scaled;
        let sin_term = sqrt3_half * theta.sin() * mag_scaled;
        result.insert(cos_term - shift);
        result.insert(-cos_term / two - sin_term - shift);
        result.insert(-cos_term / two + sin_term - shift);
    } else {
        let nq = -q;
        let cr = two * signed_cbrt(nq);
        result.insert(cr - shift);
        result.insert(-cr / two - shift);
    }

    result
}

/// Solve the depressed cubic `x^3 + a x^2 + b x + c = 0` for its real roots,
/// writing them into `out`. Returns the number of distinct real roots.
#[inline]
pub fn solve_cubic_count<F: Float>(out: &mut [F; 3], a: F, b: F, c: F) -> usize {
    let two = F::from(2.0).unwrap();
    let three = F::from(3.0).unwrap();
    let a = a / three;
    let a_sq = a * a;
    let mut p = a_sq - b / three;
    let q = (a * (two * a_sq - b) + c) / two;
    let q_sq = q * q;
    let p_cube = p * p * p;

    if q_sq < p_cube {
        let sqrt_p = p.sqrt();
        let mut cos_arg = q / (p * sqrt_p);
        if cos_arg < -F::one() {
            cos_arg = -F::one();
        } else if cos_arg > F::one() {
            cos_arg = F::one();
        }
        p = -two * sqrt_p;
        let theta = cos_arg.acos() / three;
        let sqrt3_half = F::from(0.866_025_403_784_438_6).unwrap();
        let cos_term = theta.cos() * p;
        let sin_term = sqrt3_half * theta.sin() * p;
        out[0] = cos_term - a;
        out[1] = -cos_term / two - sin_term - a;
        out[2] = -cos_term / two + sin_term - a;
        3
    } else {
        let sqrt3 = F::from(1.7320508075688772).unwrap();
        let mut u = -signed_cbrt(q.abs() + (q_sq - p_cube).sqrt());
        if q < F::zero() {
            u = -u;
        }
        let v = if u == F::zero() { F::zero() } else { p / u };
        out[0] = (u + v) - a;
        out[1] = -(u + v) / two - a;
        out[2] = sqrt3 * (u - v) / two;
        if out[2].abs() < F::epsilon() {
            out[2] = out[1];
            return 2;
        }
        1
    }
}

/// Solve the monic quartic `x^4 + a x^3 + b x^2 + c x + d = 0` for its
/// non-negative real roots, using Ferrari's method.
#[inline]
pub fn solve_quartic<F: Float>(a: F, b: F, c: F, d: F) -> PositiveSet<F, 4> {
    let mut result: PositiveSet<F, 4> = PositiveSet::new();
    let zero = F::zero();
    let two = F::from(2.0).unwrap();
    let four = F::from(4.0).unwrap();
    let eps16 = F::from(16.0).unwrap() * F::epsilon();

    if d == zero {
        if c == zero {
            result.insert(zero);
            let disc = a * a - four * b;
            if disc == zero {
                result.insert(-a / two);
            } else if disc > zero {
                let sqrt_disc = disc.sqrt();
                result.insert((-a - sqrt_disc) / two);
                result.insert((-a + sqrt_disc) / two);
            }
            return result;
        }
        if a == zero && b == zero {
            result.insert(zero);
            result.insert(-signed_cbrt(c));
            return result;
        }
    }

    let cb_b = -b;
    let cb_c = a * c - four * d;
    let cb_d = -a * a * d - c * c + four * b * d;
    let mut cubic_roots = [zero; 3];
    let n_real = solve_cubic_count(&mut cubic_roots, cb_b, cb_c, cb_d);
    let mut y = cubic_roots[0];
    if n_real != 1 {
        if cubic_roots[1].abs() > y.abs() {
            y = cubic_roots[1];
        }
        if cubic_roots[2].abs() > y.abs() {
            y = cubic_roots[2];
        }
    }

    let t1;
    let t2;
    let s1;
    let s2;
    let mut delta = y * y - four * d;
    if delta <= zero {
        t1 = y / two;
        t2 = y / two;
        delta = a * a - four * (b - y);
        if delta <= zero {
            s1 = a / two;
            s2 = a / two;
        } else {
            let sqrt_delta = delta.sqrt();
            s1 = (a + sqrt_delta) / two;
            s2 = (a - sqrt_delta) / two;
        }
    } else {
        let sqrt_delta2 = delta.sqrt();
        t1 = (y + sqrt_delta2) / two;
        t2 = (y - sqrt_delta2) / two;
        s1 = (a * t1 - c) / (t1 - t2);
        s2 = (c - a * t2) / (t1 - t2);
    }

    delta = s1 * s1 - four * t1;
    if delta.abs() < eps16 {
        result.insert(-s1 / two);
    } else if delta > zero {
        let sqrt_d1 = delta.sqrt();
        result.insert((-s1 - sqrt_d1) / two);
        result.insert((-s1 + sqrt_d1) / two);
    }

    delta = s2 * s2 - four * t2;
    if delta.abs() < eps16 {
        result.insert(-s2 / two);
    } else if delta > zero {
        let sqrt_d2 = delta.sqrt();
        result.insert((-s2 - sqrt_d2) / two);
        result.insert((-s2 + sqrt_d2) / two);
    }

    result
}

/// Same as [`solve_quartic`] but takes a 4-array of coefficients.
#[inline]
pub fn solve_quartic_arr<F: Float>(coeffs: &[F; 4]) -> PositiveSet<F, 4> {
    solve_quartic(coeffs[0], coeffs[1], coeffs[2], coeffs[3])
}

/// Evaluate the polynomial `coeffs[0] x^(N-1) + coeffs[1] x^(N-2) + ... + coeffs[N-1]` at `x`.
#[inline]
pub fn poly_eval<F: Float>(coeffs: &[F], x: F) -> F {
    let n = coeffs.len();
    let mut result = F::zero();
    if n == 0 {
        return result;
    }
    if x == F::zero() {
        return coeffs[n - 1];
    }
    if x == F::one() {
        for &c in coeffs {
            result = result + c;
        }
        return result;
    }
    let mut x_pow = F::one();
    for i in (0..n).rev() {
        result = result + coeffs[i] * x_pow;
        x_pow = x_pow * x;
    }
    result
}

/// Derivative coefficients of a polynomial (non-monic input form).
///
/// For `f(x) = c[0] x^(N-1) + c[1] x^(N-2) + ... + c[N-1]`, returns the
/// coefficients of `f'` in the same non-monic form.
pub fn poly_derivative<F: Float, const N: usize, const NM1: usize>(coeffs: &[F; N]) -> [F; NM1] {
    debug_assert!(NM1 + 1 == N);
    let mut out = [F::zero(); NM1];
    for i in 0..NM1 {
        out[i] = F::from(N - 1 - i).unwrap() * coeffs[i];
    }
    out
}

/// Monic-form derivative of a polynomial.
pub fn poly_monic_derivative<F: Float, const N: usize, const NM1: usize>(
    coeffs: &[F; N],
) -> [F; NM1] {
    debug_assert!(NM1 + 1 == N);
    let mut out = [F::zero(); NM1];
    out[0] = F::one();
    let denom = F::from(N - 1).unwrap();
    for i in 1..NM1 {
        out[i] = F::from(N - 1 - i).unwrap() * coeffs[i] / denom;
    }
    out
}

/// Bracketed Newton iteration. `coeffs` is in non-monic form (see
/// [`poly_eval`]). The bracket `[lower, upper]` must straddle a sign change.
///
/// Returns the refined root.
pub fn poly_root_newton<F: Float, const N: usize, const NM1: usize>(
    coeffs: &[F; N],
    mut lower: F,
    mut upper: F,
) -> F {
    debug_assert!(NM1 + 1 == N);
    let zero = F::zero();
    let two = F::from(2.0).unwrap();
    let f_lower = poly_eval(coeffs, lower);
    let f_upper = poly_eval(coeffs, upper);
    if f_lower == zero {
        return lower;
    }
    if f_upper == zero {
        return upper;
    }
    if f_lower > zero {
        core::mem::swap(&mut lower, &mut upper);
    }
    let mut x = (lower + upper) / two;
    let mut prev_step = (upper - lower).abs();
    let mut step = prev_step;
    let deriv: [F; NM1] = poly_derivative::<F, N, NM1>(coeffs);
    let mut fx = poly_eval(coeffs, x);
    let mut dfx = poly_eval(&deriv, x);
    let tol = F::from(ROOT_TOLERANCE).unwrap();
    for _ in 0..64 {
        let bisect_cond_a = ((x - upper) * dfx - fx) * ((x - lower) * dfx - fx) > zero;
        let bisect_cond_b = (two * fx).abs() > (prev_step * dfx).abs();
        if bisect_cond_a || bisect_cond_b {
            prev_step = step;
            step = (upper - lower) / two;
            let new_x = lower + step;
            if lower == new_x {
                x = new_x;
                break;
            }
            x = new_x;
        } else {
            prev_step = step;
            step = fx / dfx;
            let prev_x = x;
            x = x - step;
            if prev_x == x {
                break;
            }
        }
        if step.abs() < tol {
            break;
        }
        fx = poly_eval(coeffs, x);
        dfx = poly_eval(&deriv, x);
        if fx < zero {
            lower = x;
        } else {
            upper = x;
        }
    }
    x
}
