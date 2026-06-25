//! The discrete σ-LP. The decision variable is `σ[k]` (arc length at tick `k`);
//! the objective maximises total progress so the profile runs at the limits
//! wherever it can, subject to rest-to-rest boundary, monotone advance, and the
//! per-joint finite-difference v/a/j bounds.
//!
//! Within one chord segment the output joint `j` at tick `k` is affine in `σ`:
//! `q[k]ⱼ = c[k]ⱼ + m[k]ⱼ·σ[k]`, with slope `m[k]ⱼ = secant[bin]ⱼ` and offset
//! `c[k]ⱼ = knot[bin]ⱼ − secant[bin]ⱼ·s[bin]`. So each finite difference of `q` is
//! an *exact* linear combination of the `σ`s — even where a stencil straddles a
//! corner and the slopes differ. Writing the v/a/j limits as those exact rows
//! (rather than a single `secant·Δᵐσ` cap on a nominal segment) bounds the
//! quantity the controller actually differences, including across sharp joint
//! kinks, with no cross-bin leak. Jerk stays linear because `dt` is fixed.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use deke_types::SRobotQ;

/// Solve `maximise Σσ` subject to `σ[0]=0`, `σ[n-1]=total`, rest (v=a=0) at both
/// ends, monotone advance, an optional per-tick TCP `σ̇` cap, and the exact
/// per-joint first/second/third finite-difference bounds (planned at
/// `margin·limit`; the caller's FD verify against the true limit is the
/// backstop). Returns the σ profile, or `None` if the program is infeasible.
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::field_reassign_with_default
)]
pub(crate) fn solve_sigma<const N: usize>(
    n: usize,
    total: f64,
    m: &[[f64; N]],
    c: &[[f64; N]],
    v_max: &SRobotQ<N, f64>,
    a_max: &SRobotQ<N, f64>,
    j_max: &SRobotQ<N, f64>,
    tcp_dsigma_max: Option<&[f64]>,
    seg_lo: Option<&[f64]>,
    seg_hi: Option<&[f64]>,
    bins: &[usize],
    margin: f64,
    dt: f64,
) -> Option<Vec<f64>> {
    let mut t: Vec<(usize, usize, f64)> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut row = 0usize;

    // Solve in σ̃ = σ/total ∈ [0,1] so the variables are O(1) and the IPM is well
    // conditioned; every cap/offset is scaled by `inv_total` and the result is
    // rescaled back to absolute σ on return.
    let inv_total = 1.0 / total;

    let eq = |t: &mut Vec<(usize, usize, f64)>,
              b: &mut Vec<f64>,
              row: &mut usize,
              e: &[(usize, f64)],
              rhs: f64| {
        for &(col, v) in e {
            t.push((*row, col, v));
        }
        b.push(rhs);
        *row += 1;
    };
    eq(&mut t, &mut b, &mut row, &[(0, 1.0)], 0.0);
    eq(&mut t, &mut b, &mut row, &[(n - 1, 1.0)], 1.0);
    eq(&mut t, &mut b, &mut row, &[(1, 1.0), (0, -1.0)], 0.0);
    eq(&mut t, &mut b, &mut row, &[(2, 1.0), (1, -1.0)], 0.0);
    eq(
        &mut t,
        &mut b,
        &mut row,
        &[(n - 1, 1.0), (n - 2, -1.0)],
        0.0,
    );
    eq(
        &mut t,
        &mut b,
        &mut row,
        &[(n - 2, 1.0), (n - 3, -1.0)],
        0.0,
    );
    let n_eq = row;

    // One inequality `a·σ ≤ rhs`, row-scaled by `1/denom` (the limit·dtᵏ part) so
    // every cap normalises to O(1) and the IPM hits the tight ramp rows fast.
    // Near-zero coefficients are dropped to keep the matrix banded-sparse.
    // Row-scale each inequality by `1/denom` (the limit·dtᵏ part) so its (tiny) cap
    // RHS normalises to O(1); without this the jerk caps (~1e-10) are looser than
    // the solver's tolerance and the returned solution violates them. Near-zero
    // coefficients are dropped to keep the matrix banded-sparse.
    let ineq = |t: &mut Vec<(usize, usize, f64)>,
                b: &mut Vec<f64>,
                row: &mut usize,
                e: &[(usize, f64)],
                rhs: f64,
                denom: f64| {
        let inv = 1.0 / denom;
        for &(col, coef) in e {
            let v = coef * inv;
            if v.abs() > 1e-15 {
                t.push((*row, col, v));
            }
        }
        b.push(rhs * inv);
        *row += 1;
    };

    for k in 1..n {
        ineq(
            &mut t,
            &mut b,
            &mut row,
            &[(k - 1, 1.0), (k, -1.0)],
            0.0,
            1.0,
        );
    }

    if let Some(tc) = tcp_dsigma_max {
        for k in 1..n {
            if tc[k].is_finite() {
                let cap = tc[k] * inv_total;
                ineq(
                    &mut t,
                    &mut b,
                    &mut row,
                    &[(k, 1.0), (k - 1, -1.0)],
                    cap,
                    cap.max(1e-12),
                );
            }
        }
    }

    // Box each tick into its assigned segment so `bin_of(σ)` matches the segment
    // whose secant the FD rows were built from — the cross-bin leak vanishes.
    if let Some(hi) = seg_hi {
        for k in 0..n {
            ineq(
                &mut t,
                &mut b,
                &mut row,
                &[(k, 1.0)],
                hi[k] * inv_total,
                1.0,
            );
        }
    }
    if let Some(lo) = seg_lo {
        for k in 0..n {
            ineq(
                &mut t,
                &mut b,
                &mut row,
                &[(k, -1.0)],
                -lo[k] * inv_total,
                1.0,
            );
        }
    }

    // `min_j limit_j / |secant_j|` on a segment — the projected scalar limit. Where
    // every sample of a stencil shares one segment the per-joint rows collapse to
    // this single two-sided row (exact, since the secant is constant); only stencils
    // that straddle a corner need the full per-joint rows with their offset terms.
    let proj = |mv: &[f64; N], lim: &SRobotQ<N, f64>| -> f64 {
        (0..N)
            .map(|j| lim.0[j] / mv[j].abs().max(1e-12))
            .fold(f64::INFINITY, f64::min)
    };

    for k in 1..n {
        if bins[k] == bins[k - 1] {
            let cap = margin * proj(&m[k], v_max) * dt * inv_total;
            let denom = cap.max(1e-12);
            ineq(
                &mut t,
                &mut b,
                &mut row,
                &[(k, 1.0), (k - 1, -1.0)],
                cap,
                denom,
            );
            ineq(
                &mut t,
                &mut b,
                &mut row,
                &[(k, -1.0), (k - 1, 1.0)],
                cap,
                denom,
            );
        } else {
            for j in 0..N {
                let (mk, mk1) = (m[k][j], m[k - 1][j]);
                let off = (c[k][j] - c[k - 1][j]) * inv_total;
                let lim = margin * v_max.0[j] * dt * inv_total;
                let denom = (v_max.0[j] * dt * inv_total).max(1e-12);
                ineq(
                    &mut t,
                    &mut b,
                    &mut row,
                    &[(k, mk), (k - 1, -mk1)],
                    lim - off,
                    denom,
                );
                ineq(
                    &mut t,
                    &mut b,
                    &mut row,
                    &[(k, -mk), (k - 1, mk1)],
                    lim + off,
                    denom,
                );
            }
        }
    }
    for k in 2..n {
        if bins[k] == bins[k - 1] && bins[k] == bins[k - 2] {
            let cap = margin * proj(&m[k], a_max) * dt * dt * inv_total;
            let denom = cap.max(1e-12);
            ineq(
                &mut t,
                &mut b,
                &mut row,
                &[(k, 1.0), (k - 1, -2.0), (k - 2, 1.0)],
                cap,
                denom,
            );
            ineq(
                &mut t,
                &mut b,
                &mut row,
                &[(k, -1.0), (k - 1, 2.0), (k - 2, -1.0)],
                cap,
                denom,
            );
        } else {
            for j in 0..N {
                let (mk, mk1, mk2) = (m[k][j], m[k - 1][j], m[k - 2][j]);
                let off = (c[k][j] - 2.0 * c[k - 1][j] + c[k - 2][j]) * inv_total;
                let lim = margin * a_max.0[j] * dt * dt * inv_total;
                let denom = (a_max.0[j] * dt * dt * inv_total).max(1e-12);
                ineq(
                    &mut t,
                    &mut b,
                    &mut row,
                    &[(k, mk), (k - 1, -2.0 * mk1), (k - 2, mk2)],
                    lim - off,
                    denom,
                );
                ineq(
                    &mut t,
                    &mut b,
                    &mut row,
                    &[(k, -mk), (k - 1, 2.0 * mk1), (k - 2, -mk2)],
                    lim + off,
                    denom,
                );
            }
        }
    }
    for k in 3..n {
        if bins[k] == bins[k - 1] && bins[k] == bins[k - 2] && bins[k] == bins[k - 3] {
            let cap = margin * proj(&m[k], j_max) * dt * dt * dt * inv_total;
            let denom = cap.max(1e-12);
            ineq(
                &mut t,
                &mut b,
                &mut row,
                &[(k, 1.0), (k - 1, -3.0), (k - 2, 3.0), (k - 3, -1.0)],
                cap,
                denom,
            );
            ineq(
                &mut t,
                &mut b,
                &mut row,
                &[(k, -1.0), (k - 1, 3.0), (k - 2, -3.0), (k - 3, 1.0)],
                cap,
                denom,
            );
        } else {
            for j in 0..N {
                let (mk, mk1, mk2, mk3) = (m[k][j], m[k - 1][j], m[k - 2][j], m[k - 3][j]);
                let off =
                    (c[k][j] - 3.0 * c[k - 1][j] + 3.0 * c[k - 2][j] - c[k - 3][j]) * inv_total;
                let lim = margin * j_max.0[j] * dt * dt * dt * inv_total;
                let denom = (j_max.0[j] * dt * dt * dt * inv_total).max(1e-12);
                ineq(
                    &mut t,
                    &mut b,
                    &mut row,
                    &[
                        (k, mk),
                        (k - 1, -3.0 * mk1),
                        (k - 2, 3.0 * mk2),
                        (k - 3, -mk3),
                    ],
                    lim - off,
                    denom,
                );
                ineq(
                    &mut t,
                    &mut b,
                    &mut row,
                    &[
                        (k, -mk),
                        (k - 1, 3.0 * mk1),
                        (k - 2, -3.0 * mk2),
                        (k - 3, mk3),
                    ],
                    lim + off,
                    denom,
                );
            }
        }
    }
    let m_rows = row;

    let a = triplets_to_csc(m_rows, n, t);
    let p = CscMatrix::new(n, n, vec![0; n + 1], vec![], vec![]);
    let q = vec![-1.0f64; n];
    let cones = [
        SupportedConeT::ZeroConeT(n_eq),
        SupportedConeT::NonnegativeConeT(m_rows - n_eq),
    ];
    let mut set = DefaultSettings::default();
    set.verbose = false;
    set.max_iter = 200;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, set).ok()?;
    solver.solve();
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            Some(solver.solution.x.iter().map(|x| x * total).collect())
        }
        _ => None,
    }
}

/// Build a column-compressed sparse matrix from `(row, col, val)` triplets.
fn triplets_to_csc(m: usize, n: usize, mut t: Vec<(usize, usize, f64)>) -> CscMatrix<f64> {
    t.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut colptr = vec![0usize; n + 1];
    let mut rowval = Vec::with_capacity(t.len());
    let mut nzval = Vec::with_capacity(t.len());
    for &(r, _c, v) in &t {
        rowval.push(r);
        nzval.push(v);
    }
    for &(_r, c, _v) in &t {
        colptr[c + 1] += 1;
    }
    for c in 0..n {
        colptr[c + 1] += colptr[c];
    }
    CscMatrix::new(m, n, colptr, rowval, nzval)
}
