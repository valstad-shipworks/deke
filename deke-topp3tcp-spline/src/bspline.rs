//! Scalar B-spline interpolation with derivative boundary conditions.

/// Find the knot span index `i` such that `knots[i] <= x < knots[i+1]`.
/// For the right endpoint, returns the last valid span.
pub(crate) fn find_span(knots: &[f64], k: usize, n_coeffs: usize, x: f64) -> usize {
    let last = n_coeffs;
    if x >= knots[last] {
        let mut i = last - 1;
        while i > k && knots[i] == knots[i + 1] {
            i -= 1;
        }
        return i;
    }
    if x <= knots[k] {
        return k;
    }
    let mut lo = k;
    let mut hi = last;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if knots[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Compute non-zero basis functions and their derivatives up to order `nd`
/// at parameter `x` in knot span `i`.
///
/// Returns `ders[d][j]` = d-th derivative of `N_{i-k+j, k}(x)` for
/// `d = 0..=nd`, `j = 0..=k`. Algorithm A2.3 from *The NURBS Book*.
pub(crate) fn ders_basis_funs(
    knots: &[f64],
    i: usize,
    x: f64,
    k: usize,
    nd: usize,
) -> Vec<Vec<f64>> {
    let mut ndu = vec![vec![0.0f64; k + 1]; k + 1];
    ndu[0][0] = 1.0;
    let mut left = vec![0.0f64; k + 1];
    let mut right = vec![0.0f64; k + 1];

    for j in 1..=k {
        left[j] = x - knots[i + 1 - j];
        right[j] = knots[i + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    let nd = nd.min(k);
    let mut ders = vec![vec![0.0f64; k + 1]; nd + 1];
    for j in 0..=k {
        ders[0][j] = ndu[j][k];
    }

    let mut a = vec![vec![0.0f64; k + 1]; 2];
    for r in 0..=k {
        let mut s1 = 0usize;
        let mut s2 = 1usize;
        a[0][0] = 1.0;

        for kk in 1..=nd {
            let mut d = 0.0f64;
            let rk = r as isize - kk as isize;
            let pk = k as isize - kk as isize;
            if rk >= 0 {
                let denom = ndu[pk as usize + 1][rk as usize];
                if denom.abs() > 0.0 {
                    a[s2][0] = a[s1][0] / denom;
                    d = a[s2][0] * ndu[rk as usize][pk as usize];
                } else {
                    a[s2][0] = 0.0;
                }
            }
            let j1 = if rk >= -1 { 1usize } else { (-rk) as usize };
            let j2 = if (r as isize - 1) <= pk {
                kk - 1
            } else {
                (pk + 1 - rk) as usize
            };
            let j2 = j2.min(k);
            for j in j1..=j2 {
                let idx = (rk + j as isize) as usize;
                let denom = ndu[pk as usize + 1][idx];
                if denom.abs() > 0.0 {
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / denom;
                    d += a[s2][j] * ndu[idx][pk as usize];
                } else {
                    a[s2][j] = 0.0;
                }
            }
            if (r as isize) <= pk {
                let denom = ndu[pk as usize + 1][r];
                if denom.abs() > 0.0 {
                    a[s2][kk] = -a[s1][kk - 1] / denom;
                    d += a[s2][kk] * ndu[r][pk as usize];
                } else {
                    a[s2][kk] = 0.0;
                }
            }
            ders[kk][r] = d;
            std::mem::swap(&mut s1, &mut s2);
        }
    }

    let mut fac = k as f64;
    for kk in 1..=nd {
        for j in 0..=k {
            ders[kk][j] *= fac;
        }
        fac *= (k - kk) as f64;
    }
    ders
}

/// Solve a dense `n × n` system `A x = b` via Gaussian elimination with
/// partial pivoting. On return `b` contains the solution.
pub(crate) fn solve_dense(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) {
    let n = b.len();
    for col in 0..n {
        let mut best = col;
        let mut best_val = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > best_val {
                best = row;
                best_val = v;
            }
        }
        if best != col {
            a.swap(col, best);
            b.swap(col, best);
        }
        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for c in (col + 1)..n {
                a[row][c] -= factor * a[col][c];
            }
            a[row][col] = 0.0;
            b[row] -= factor * b[col];
        }
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * b[j];
        }
        b[i] = sum / a[i][i];
    }
}

/// Boundary condition: `(derivative_order, value)`.
pub type BcEntry = (usize, f64);

/// Scalar B-spline of degree `k` with clamped knot vector.
#[derive(Clone, Debug)]
pub struct BSpline {
    pub(crate) knots: Vec<f64>,
    pub(crate) coeffs: Vec<f64>,
    pub(crate) degree: usize,
}

impl BSpline {
    /// Build an interpolating B-spline of the given `degree` that passes
    /// through `(sites[i], values[i])` with optional derivative BCs at the
    /// endpoints.
    pub fn interpolate(
        sites: &[f64],
        values: &[f64],
        degree: usize,
        bc_left: Option<&[BcEntry]>,
        bc_right: Option<&[BcEntry]>,
    ) -> Self {
        let n_data = sites.len();
        assert!(n_data >= 2, "need at least 2 data points");
        assert_eq!(n_data, values.len());

        let nl = bc_left.map_or(0, |v| v.len());
        let nr = bc_right.map_or(0, |v| v.len());
        let m = n_data + nl + nr;

        let n_interior = n_data.saturating_sub(degree + 1);
        let mut knots = Vec::with_capacity(m + degree + 1);
        let x0 = sites[0];
        let x_last = sites[n_data - 1];
        for _ in 0..(degree + 1 + nl) {
            knots.push(x0);
        }
        for j in 0..n_interior {
            let start = j + 1;
            let end = (j + degree).min(n_data - 1);
            let avg: f64 =
                (start..=end).map(|i| sites[i]).sum::<f64>() / (end - start + 1) as f64;
            knots.push(avg);
        }
        for _ in 0..(degree + 1 + nr) {
            knots.push(x_last);
        }
        debug_assert_eq!(knots.len(), m + degree + 1);

        let mut mat = vec![vec![0.0f64; m]; m];
        let mut rhs = vec![0.0f64; m];
        let mut row = 0usize;

        if let Some(bcs) = bc_left {
            for &(deriv_order, val) in bcs {
                let span = find_span(&knots, degree, m, x0);
                let ders = ders_basis_funs(&knots, span, x0, degree, deriv_order);
                for j in 0..=degree {
                    let col = span - degree + j;
                    if col < m {
                        mat[row][col] = ders[deriv_order][j];
                    }
                }
                rhs[row] = val;
                row += 1;
            }
        }

        for i in 0..n_data {
            let span = find_span(&knots, degree, m, sites[i]);
            let ders = ders_basis_funs(&knots, span, sites[i], degree, 0);
            for j in 0..=degree {
                let col = span - degree + j;
                if col < m {
                    mat[row][col] = ders[0][j];
                }
            }
            rhs[row] = values[i];
            row += 1;
        }

        if let Some(bcs) = bc_right {
            for &(deriv_order, val) in bcs {
                let span = find_span(&knots, degree, m, x_last);
                let ders = ders_basis_funs(&knots, span, x_last, degree, deriv_order);
                for j in 0..=degree {
                    let col = span - degree + j;
                    if col < m {
                        mat[row][col] = ders[deriv_order][j];
                    }
                }
                rhs[row] = val;
                row += 1;
            }
        }

        debug_assert_eq!(row, m);

        solve_dense(&mut mat, &mut rhs);

        BSpline {
            knots,
            coeffs: rhs,
            degree,
        }
    }

    #[inline]
    pub fn eval(&self, x: f64) -> f64 {
        self.eval_deriv(x, 0)
    }

    pub fn eval_deriv(&self, x: f64, nu: usize) -> f64 {
        if nu > self.degree {
            return 0.0;
        }
        if nu == 0 {
            let span = find_span(&self.knots, self.degree, self.coeffs.len(), x);
            let ders = ders_basis_funs(&self.knots, span, x, self.degree, 0);
            let mut val = 0.0;
            for j in 0..=self.degree {
                let col = span - self.degree + j;
                if col < self.coeffs.len() {
                    val += ders[0][j] * self.coeffs[col];
                }
            }
            return val;
        }
        // Derivative by reducing degree: B'(x) is a degree-(k-1) spline with
        //   d_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
        let mut coeffs = self.coeffs.clone();
        let mut knots = self.knots.clone();
        let mut deg = self.degree;
        for _ in 0..nu {
            let n = coeffs.len();
            if n < 2 || deg == 0 {
                return 0.0;
            }
            let mut new_coeffs = Vec::with_capacity(n - 1);
            for i in 0..(n - 1) {
                let dt = knots[i + deg + 1] - knots[i + 1];
                if dt.abs() < 1e-30 {
                    new_coeffs.push(0.0);
                } else {
                    new_coeffs.push(deg as f64 * (coeffs[i + 1] - coeffs[i]) / dt);
                }
            }
            knots = knots[1..knots.len() - 1].to_vec();
            coeffs = new_coeffs;
            deg -= 1;
        }
        let n_coeffs = coeffs.len();
        if n_coeffs == 0 {
            return 0.0;
        }
        let span = find_span(&knots, deg, n_coeffs, x);
        let ders = ders_basis_funs(&knots, span, x, deg, 0);
        let mut val = 0.0;
        for j in 0..=deg {
            let col = span - deg + j;
            if col < n_coeffs {
                val += ders[0][j] * coeffs[col];
            }
        }
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_interpolation() {
        let sites = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let values = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let sp = BSpline::interpolate(&sites, &values, 3, None, None);
        for (&x, &y) in sites.iter().zip(values.iter()) {
            assert!((sp.eval(x) - y).abs() < 1e-10);
        }
    }

    #[test]
    fn derivative_of_linear_function() {
        let sites = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![1.0, 3.0, 5.0, 7.0];
        let sp = BSpline::interpolate(&sites, &values, 3, None, None);
        for &x in &[0.5, 1.0, 1.5, 2.5] {
            assert!((sp.eval_deriv(x, 1) - 2.0).abs() < 1e-8);
            assert!(sp.eval_deriv(x, 2).abs() < 1e-8);
        }
    }
}
