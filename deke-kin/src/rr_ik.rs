//! General-6R inverse kinematics by Raghavan–Roth resultant elimination, solved
//! through the Manocha–Canny eigenvalue formulation. This is the *complete,
//! deterministic* solver for arbitrary 6R chains that have no closed-form
//! decomposition: it returns every isolated solution (up to 16) as the
//! eigenvalues of a matrix pencil, with no seeds and no iteration.
//!
//! # Method and provenance
//!
//! The chain is expressed in classic Denavit–Hartenberg parameters with
//! `A_i = RotZ(θ_i)·TransZ(d_i)·TransX(a_i)·RotX(α_i)` (Tsai, *Robot Analysis*,
//! Appendix C). The kinematic loop is split so that θ1,θ2,θ6 sit on the
//! right-hand side, yielding 6 scalar equations free of θ6; the products
//! `aᵀa, aᵀb, a×b, (aᵀa)b − 2(aᵀb)a` give 8 more, for **14 equations** linear in
//! the 9 power products of (θ4,θ5) and the 8 of (θ1,θ2):
//!
//! ```text
//!     P(θ3) · m45(θ4,θ5)  =  Q · m12(θ1,θ2)
//! ```
//!
//! with `P` (14×9) affine in sin θ3, cos θ3 and `Q` (14×8) constant. Because the
//! system is *bilinear* in those monomials, `P` and `Q` are recovered by
//! evaluating the 14 equations at a fixed set of 17 sample angles and applying a
//! constant inverse — this avoids transcribing the enormous symbolic matrix
//! entries by hand (the error-prone step) while remaining exact.
//!
//! θ1,θ2 are eliminated via the left null space of `Q` (a θ3-independent 6×14
//! matrix `N`, so `E = N·P`). The half-angle substitution
//! `tᵢ = tan(θᵢ/2)` for i=4,5 and dialytic elimination give a 12×12 matrix
//! quadratic in `x3 = tan(θ3/2)`:
//!
//! ```text
//!     Σ(x3) = A·x3² + B·x3 + C        (12×12)
//! ```
//!
//! whose `det Σ = 0` is the degree-16 characteristic polynomial. Following
//! Manocha–Canny, the roots are obtained as the eigenvalues of the 24×24
//! first-companion linearisation of `Σ` (solved as a generalized eigenproblem),
//! never by expanding the determinant. For each real `x3`, (θ4,θ5) follow from
//! the per-root 6×9 system, θ1,θ2 from a linear solve, and θ6 from the last
//! link; every candidate is FK-verified before being returned.
//!
//! See the repository `CITATIONS.md` (deke-kin) for the full citation list. Key
//! sources:
//! - M. Raghavan, B. Roth, "Inverse Kinematics of the General 6R Manipulator…",
//!   ASME J. Mech. Design 115:502–508, 1993.
//! - L.-W. Tsai, *Robot Analysis*, Appendix C (verbatim RR equations C.1–C.15).
//! - D. Manocha, J. F. Canny, "Efficient Inverse Kinematics for General 6R
//!   Manipulators", IEEE T-RA 10(5):648–657, 1994 (eigenvalue formulation).
//! - Reference (read, not copied; unlicensed): haijunsu-osu/IK_6R_RR_1993
//!   (`phase3/ik_6r_general.py`) and Manocha's C (`reduce.c`).

use arrayvec::ArrayVec;
use faer::Mat;

/// Classic DH parameters for one joint: link length `a`, twist `alpha`,
/// offset `d`. The joint variable θ is supplied separately.
#[derive(Debug, Clone, Copy)]
pub struct DhJoint {
    pub a: f64,
    pub alpha: f64,
    pub d: f64,
}

/// Tunable tolerances for the general-6R solver.
#[derive(Debug, Clone, Copy)]
pub struct RrConfig {
    /// FK pose-residual (Frobenius norm of 4×4 difference) below which a
    /// candidate is accepted.
    pub residual_tol: f64,
    /// |imag/real| threshold for treating an eigenvalue as a real root.
    pub imag_tol: f64,
    /// |x3| bound; eigenvalues beyond this are treated as the spurious
    /// roots-at-infinity introduced by the `(1+x3²)` clearing.
    pub root_bound: f64,
    /// Joint-space distance below which two solutions are deduplicated.
    pub dedup_tol: f64,
}

impl Default for RrConfig {
    fn default() -> Self {
        Self {
            residual_tol: 1e-6,
            imag_tol: 1e-6,
            root_bound: 1e6,
            dedup_tol: 1e-5,
        }
    }
}

const PI: f64 = std::f64::consts::PI;
const TAU: f64 = std::f64::consts::TAU;

// ----- minimal row-major 4×4 homogeneous-transform helpers -----------------

type M4 = [[f64; 4]; 4];

fn m4_identity() -> M4 {
    let mut m = [[0.0; 4]; 4];
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        m[i][i] = 1.0;
    }
    m
}

fn m4_mul(a: &M4, b: &M4) -> M4 {
    let mut o = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            o[i][j] = s;
        }
    }
    o
}

/// Inverse of a homogeneous SE(3) transform: Rᵀ and −Rᵀ p.
fn m4_inv_se3(t: &M4) -> M4 {
    let mut o = m4_identity();
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = t[j][i];
        }
    }
    for i in 0..3 {
        let mut s = 0.0;
        for row in t.iter().take(3) {
            s += row[i] * row[3];
        }
        o[i][3] = -s;
    }
    o
}

fn aiv(theta: f64) -> M4 {
    let (s, c) = theta.sin_cos();
    [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn ais(j: &DhJoint) -> M4 {
    let (sa, ca) = j.alpha.sin_cos();
    [
        [1.0, 0.0, 0.0, j.a],
        [0.0, ca, -sa, 0.0],
        [0.0, sa, ca, j.d],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// One link transform in the *screw* form the RR elimination requires:
/// `A_i = RotZ(θ_i) · C_i`, where `C_i` is the constant per-joint transform.
/// DH is the special case `C_i = ais(joint)`; a general [`crate::rr_ik`] chain
/// derives `C_i` from a `KinSpec` (see [`screw_from_kinspec`]).
fn a_link(c: &M4, theta: f64) -> M4 {
    m4_mul(&aiv(theta), c)
}

/// Forward kinematics of a screw chain: `∏ RotZ(θ_i)·C_i`.
fn fk_screw(c: &[M4; 6], q: &[f64; 6]) -> M4 {
    let mut t = m4_identity();
    for i in 0..6 {
        t = m4_mul(&t, &a_link(&c[i], q[i]));
    }
    t
}

/// Forward kinematics of the DH chain: `∏ A_i`.
pub fn fk_dh(dh: &[DhJoint; 6], q: &[f64; 6]) -> M4 {
    let c: [M4; 6] = std::array::from_fn(|i| ais(&dh[i]));
    fk_screw(&c, q)
}

// ----- small dense linear algebra (row-major Vec) --------------------------

#[derive(Clone)]
struct DMat {
    r: usize,
    c: usize,
    d: Vec<f64>,
}

impl DMat {
    fn zeros(r: usize, c: usize) -> Self {
        Self {
            r,
            c,
            d: vec![0.0; r * c],
        }
    }
    #[inline]
    fn at(&self, i: usize, j: usize) -> f64 {
        self.d[i * self.c + j]
    }
    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.d[i * self.c + j] = v;
    }
}

/// Fixed-shape, stack-allocated dense matrix (row-major `[[f64; C]; R]`). The
/// general-6R pipeline's constant-dimension linear algebra runs entirely on
/// these; only the runtime-sized companion/Gram matrices stay on [`DMat`].
#[derive(Clone, Debug)]
struct SMat<const R: usize, const C: usize> {
    d: [[f64; C]; R],
}

impl<const R: usize, const C: usize> SMat<R, C> {
    fn zeros() -> Self {
        Self { d: [[0.0; C]; R] }
    }
    #[inline]
    fn at(&self, i: usize, j: usize) -> f64 {
        self.d[i][j]
    }
    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.d[i][j] = v;
    }
    fn transpose(&self) -> SMat<C, R> {
        let mut out = SMat::<C, R>::zeros();
        for i in 0..R {
            for j in 0..C {
                out.d[j][i] = self.d[i][j];
            }
        }
        out
    }
    fn matmul<const C2: usize>(&self, o: &SMat<C, C2>) -> SMat<R, C2> {
        let mut out = SMat::<R, C2>::zeros();
        for i in 0..R {
            for k in 0..C {
                let a = self.d[i][k];
                if a == 0.0 {
                    continue;
                }
                for j in 0..C2 {
                    out.d[i][j] += a * o.d[k][j];
                }
            }
        }
        out
    }
}

/// Solve a square `N×N` system `A x = b` (`M` RHS columns) by Gaussian
/// elimination with partial pivoting. `a` is consumed. Returns `None` if
/// singular.
fn gauss_solve<const N: usize, const M: usize>(
    mut a: SMat<N, N>,
    mut b: SMat<N, M>,
) -> Option<SMat<N, M>> {
    for col in 0..N {
        let mut piv = col;
        let mut best = a.d[col][col].abs();
        for r in (col + 1)..N {
            let v = a.d[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            a.d.swap(col, piv);
            b.d.swap(col, piv);
        }
        let d = a.d[col][col];
        for r in (col + 1)..N {
            let f = a.d[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for j in col..N {
                a.d[r][j] -= f * a.d[col][j];
            }
            for j in 0..M {
                b.d[r][j] -= f * b.d[col][j];
            }
        }
    }
    let mut x = SMat::<N, M>::zeros();
    for col in (0..N).rev() {
        for j in 0..M {
            let mut s = b.d[col][j];
            for k in (col + 1)..N {
                s -= a.d[col][k] * x.d[k][j];
            }
            x.d[col][j] = s / a.d[col][col];
        }
    }
    Some(x)
}

/// `|det(S·Sᵀ)|` where `S` is the `k×8` submatrix of `q` whose rows are given by
/// `rows` (`k ≤ 8`). The Gram matrix and its LU factorisation are formed on the
/// stack, so the pivot-row search never allocates. Used as a conditioning proxy
/// for row independence.
fn gram_det(q: &SMat<14, 8>, rows: impl Iterator<Item = usize>, k: usize) -> f64 {
    let mut s = [[0.0f64; 8]; 8];
    for (a, r) in rows.enumerate() {
        #[allow(clippy::needless_range_loop)]
        for j in 0..8 {
            s[a][j] = q.at(r, j);
        }
    }
    let mut g = [[0.0f64; 8]; 8];
    for a in 0..k {
        for b in 0..k {
            let acc: f64 = s[a].iter().zip(s[b].iter()).map(|(x, y)| x * y).sum();
            g[a][b] = acc;
        }
    }
    // det of the leading k×k block via LU with partial pivoting (G is PSD, so the
    // magnitude is all the conditioning proxy needs).
    let mut d = 1.0;
    for col in 0..k {
        let mut piv = col;
        let mut best = g[col][col].abs();
        #[allow(clippy::needless_range_loop)]
        for r in (col + 1)..k {
            let v = g[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return 0.0;
        }
        if piv != col {
            g.swap(col, piv);
        }
        let pivot = g[col][col];
        d *= pivot;
        for r in (col + 1)..k {
            let f = g[r][col] / pivot;
            if f == 0.0 {
                continue;
            }
            #[allow(clippy::needless_range_loop)]
            for j in col..k {
                g[r][j] -= f * g[col][j];
            }
        }
    }
    d.abs()
}

// ----- angle helpers -------------------------------------------------------

fn normalize_angle(t: f64) -> f64 {
    let mut a = (t + PI).rem_euclid(TAU);
    if a < 0.0 {
        a += TAU;
    }
    a - PI
}

fn normalize_sc(s: f64, c: f64) -> (f64, f64) {
    let n = s.hypot(c);
    if n < 1e-14 {
        (0.0, 1.0)
    } else {
        (s / n, c / n)
    }
}

// ----- Raghavan–Roth equation assembly -------------------------------------

/// The 17 fixed sample angle quadruples `(θ1,θ2,θ4,θ5)` whose feature matrix
/// `[m45 ; −m12]` is well-conditioned and invertible. Reused verbatim from the
/// haijunsu reference (read-only); their only requirement is invertibility.
const SAMPLE_ANGLES: [[f64; 4]; 17] = [
    [
        1.69673823897,
        2.962965871865,
        0.551751195511,
        -0.550635933697,
    ],
    [
        2.124305683239,
        -1.35151778162,
        0.042006538206,
        -1.271978220727,
    ],
    [
        -0.916963962161,
        -2.508838046012,
        1.748576867781,
        0.377559104927,
    ],
    [
        1.471879906641,
        0.971647887754,
        -0.272341445026,
        -2.027355199883,
    ],
    [
        -2.988196582985,
        0.046459526076,
        0.150075617542,
        -1.263697979985,
    ],
    [
        -2.536061119563,
        -1.494462923611,
        0.086662269634,
        -1.376098791112,
    ],
    [
        0.831512316986,
        1.996640091182,
        -1.617969141208,
        0.029856666264,
    ],
    [
        -2.841354970612,
        -2.14209536172,
        -2.71675184462,
        2.846383108325,
    ],
    [
        -1.999165286534,
        -2.771692700411,
        -2.056266835523,
        2.318982885491,
    ],
    [
        2.089577723707,
        -0.615474169356,
        -1.261127673917,
        1.307166318385,
    ],
    [
        -3.12408776899,
        2.563595214957,
        -1.111534266641,
        -2.521343721348,
    ],
    [
        1.050822228457,
        1.782149206715,
        -3.039576922521,
        -0.956065652517,
    ],
    [
        0.484527303898,
        0.394768084706,
        2.531365903151,
        -1.732241512316,
    ],
    [
        -3.101385200507,
        1.775773952919,
        -2.942430204732,
        1.864900116697,
    ],
    [
        -1.735912964048,
        -0.09455970154,
        0.960385327619,
        -2.353825670389,
    ],
    [
        3.085871917488,
        0.496623639214,
        1.394218297829,
        2.669197573194,
    ],
    [
        1.345522300159,
        -1.400760620293,
        -2.497027865158,
        -1.501501913954,
    ],
];

fn m45(t4: f64, t5: f64) -> [f64; 9] {
    let (s4, c4) = t4.sin_cos();
    let (s5, c5) = t5.sin_cos();
    [s4 * s5, s4 * c5, c4 * s5, c4 * c5, s4, c4, s5, c5, 1.0]
}

fn m12(t1: f64, t2: f64) -> [f64; 8] {
    let (s1, c1) = t1.sin_cos();
    let (s2, c2) = t2.sin_cos();
    [s1 * s2, s1 * c2, c1 * s2, c1 * c2, s1, c1, s2, c2]
}

/// Build the left (θ3,θ4,θ5) and right (θ1,θ2,θ6) `p`/`l` vectors of the split
/// loop equation `a2s·A3·A4·A5 = aiv(θ2)⁻¹·A1⁻¹·T·A6⁻¹`. `p` is the translation
/// column, `l` the third rotation column (free of θ6 on the left).
fn pl_vectors(c: &[M4; 6], target: &M4, q: &[f64; 6]) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    let a: [M4; 6] = std::array::from_fn(|i| a_link(&c[i], q[i]));
    let a2s = c[1];
    let left = m4_mul(&m4_mul(&m4_mul(&a2s, &a[2]), &a[3]), &a[4]);
    let a2v_inv = m4_inv_se3(&aiv(q[1]));
    let a0_inv = m4_inv_se3(&a[0]);
    let a5_inv = m4_inv_se3(&a[5]);
    let right = m4_mul(&m4_mul(&m4_mul(&a2v_inv, &a0_inv), target), &a5_inv);
    let pl = [left[0][3], left[1][3], left[2][3]];
    let pr = [right[0][3], right[1][3], right[2][3]];
    let ll = [left[0][2], left[1][2], left[2][2]];
    let lr = [right[0][2], right[1][2], right[2][2]];
    (pl, pr, ll, lr)
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The 14 Raghavan–Roth scalar equations evaluated at a full `q`.
fn eqs14(c: &[M4; 6], target: &M4, q: &[f64; 6]) -> [f64; 14] {
    let (pl, pr, ll, lr) = pl_vectors(c, target, q);
    let mut e = [0.0f64; 14];
    for i in 0..3 {
        e[i] = pl[i] - pr[i]; // position vector b
        e[3 + i] = ll[i] - lr[i]; // axis vector a
    }
    e[6] = dot3(&pl, &pl) - dot3(&pr, &pr); // bᵀb
    e[7] = dot3(&pl, &ll) - dot3(&pr, &lr); // aᵀb
    let xl = cross3(&pl, &ll);
    let xr = cross3(&pr, &lr);
    for i in 0..3 {
        e[8 + i] = xl[i] - xr[i]; // a×b
    }
    let cl = {
        let pp = dot3(&pl, &pl);
        let pq = dot3(&pl, &ll);
        [
            pp * ll[0] - 2.0 * pq * pl[0],
            pp * ll[1] - 2.0 * pq * pl[1],
            pp * ll[2] - 2.0 * pq * pl[2],
        ]
    };
    let cr = {
        let pp = dot3(&pr, &pr);
        let pq = dot3(&pr, &lr);
        [
            pp * lr[0] - 2.0 * pq * pr[0],
            pp * lr[1] - 2.0 * pq * pr[1],
            pp * lr[2] - 2.0 * pq * pr[2],
        ]
    };
    for i in 0..3 {
        e[11 + i] = cl[i] - cr[i]; // (bᵀb)a − 2(aᵀb)b
    }
    e
}

/// The constant 17×17 feature matrix `F` rows `[m45(θ4,θ5) | −m12(θ1,θ2)]`.
fn feature_matrix() -> SMat<17, 17> {
    let mut f = SMat::<17, 17>::zeros();
    for (i, s) in SAMPLE_ANGLES.iter().enumerate() {
        let a = m45(s[2], s[3]);
        let b = m12(s[0], s[1]);
        for (j, &aj) in a.iter().enumerate() {
            f.set(i, j, aj);
        }
        for (j, &bj) in b.iter().enumerate() {
            f.set(i, 9 + j, -bj);
        }
    }
    f
}

/// Recover `P` (14×9) and `Q` (14×8) at a fixed `x3` by solving `F·U = Y`,
/// `Y[i] = eqs14(sample_i with θ3 = 2·atan(x3))`.
fn pq_at_x3(c: &[M4; 6], target: &M4, x3: f64, finv_solver: &FInv) -> (SMat<14, 9>, SMat<14, 8>) {
    let theta3 = 2.0 * x3.atan();
    let mut y = SMat::<17, 14>::zeros();
    for (i, s) in SAMPLE_ANGLES.iter().enumerate() {
        let q = [s[0], s[1], theta3, s[2], s[3], 0.0];
        let e = eqs14(c, target, &q);
        for (j, &ej) in e.iter().enumerate() {
            y.set(i, j, ej);
        }
    }
    let coeff = finv_solver.apply(&y); // (17×14) = F⁻¹ Y
    let mut p = SMat::<14, 9>::zeros();
    let mut q = SMat::<14, 8>::zeros();
    for k in 0..14 {
        for j in 0..9 {
            p.set(k, j, coeff.at(j, k));
        }
        for j in 0..8 {
            q.set(k, j, coeff.at(9 + j, k));
        }
    }
    (p, q)
}

/// Precomputed `F⁻¹` applied as a solver (we keep `F` factored implicitly by
/// storing its inverse, since it is constant across the whole solve).
struct FInv {
    inv: SMat<17, 17>,
}

impl FInv {
    fn new() -> Self {
        let f = feature_matrix();
        let id = {
            let mut m = SMat::<17, 17>::zeros();
            for i in 0..17 {
                m.set(i, i, 1.0);
            }
            m
        };
        let inv = gauss_solve(f, id).expect("sample-angle feature matrix is invertible");
        Self { inv }
    }
    fn apply(&self, y: &SMat<17, 14>) -> SMat<17, 14> {
        self.inv.matmul(y)
    }
}

/// sin/cos-θ3 affine model of `P` and the constant `Q`: `P(θ3) = Ps·sinθ3 +
/// Pc·cosθ3 + P1`. Sampled at three θ3 values and interpolated exactly.
struct PqModel {
    ps: SMat<14, 9>,
    pc: SMat<14, 9>,
    p1: SMat<14, 9>,
    q: SMat<14, 8>,
}

fn build_pq_model(c: &[M4; 6], target: &M4, finv: &FInv) -> PqModel {
    let grid = [-2.1f64, -0.2, 1.3];
    let basis = {
        let mut b = SMat::<3, 3>::zeros();
        for (i, &t) in grid.iter().enumerate() {
            b.set(i, 0, t.sin());
            b.set(i, 1, t.cos());
            b.set(i, 2, 1.0);
        }
        b
    };
    let mut q_acc = SMat::<14, 8>::zeros();
    let p_samples: [SMat<14, 9>; 3] = std::array::from_fn(|i| {
        let x3 = (0.5 * grid[i]).tan();
        let (p, q) = pq_at_x3(c, target, x3, finv);
        for k in 0..14 {
            for j in 0..8 {
                q_acc.d[k][j] += q.d[k][j] / 3.0;
            }
        }
        p
    });
    // Solve basis · [Ps;Pc;P1] = [P(t3_0);P(t3_1);P(t3_2)] per entry. The 14×9 P
    // is flattened row-major into the 126 columns of the RHS.
    let mut rhs = SMat::<3, { 14 * 9 }>::zeros();
    for (row, p) in p_samples.iter().enumerate() {
        for k in 0..14 {
            for j in 0..9 {
                rhs.set(row, k * 9 + j, p.at(k, j));
            }
        }
    }
    let coeff = gauss_solve(basis, rhs).expect("3-point sin/cos interpolation is nonsingular");
    let mut ps = SMat::<14, 9>::zeros();
    let mut pc = SMat::<14, 9>::zeros();
    let mut p1 = SMat::<14, 9>::zeros();
    for k in 0..14 {
        for j in 0..9 {
            ps.set(k, j, coeff.at(0, k * 9 + j));
            pc.set(k, j, coeff.at(1, k * 9 + j));
            p1.set(k, j, coeff.at(2, k * 9 + j));
        }
    }
    PqModel {
        ps,
        pc,
        p1,
        q: q_acc,
    }
}

/// Greedily choose 8 well-conditioned independent rows of `Q` (14×8).
fn select_pivot_rows(q: &SMat<14, 8>) -> Option<[usize; 8]> {
    let mut idx: ArrayVec<usize, 8> = ArrayVec::new();
    let mut remaining: ArrayVec<usize, 14> = (0..14).collect();
    while idx.len() < 8 && !remaining.is_empty() {
        let mut best_row = None;
        let mut best_score = f64::INFINITY;
        for &r in &remaining {
            // candidate = current pivot rows plus row r (no need to materialise it)
            let k = idx.len() + 1;
            let dg = gram_det(q, idx.iter().copied().chain(std::iter::once(r)), k);
            if dg <= 1e-18 {
                continue;
            }
            let score = 1.0 / dg;
            if score < best_score {
                best_score = score;
                best_row = Some(r);
            }
        }
        let br = best_row?;
        idx.push(br);
        remaining.retain(|x| *x != br);
    }
    if idx.len() != 8 {
        return None;
    }
    let mut out = [0usize; 8];
    out.copy_from_slice(&idx[..]);
    Some(out)
}

/// Left elimination matrix `N` (6×14): rows span the left null space of `Q`,
/// built from a fixed 8-row pivot block so it is identical across x3 samples.
/// In permuted order `[pivot, rest]`, `N = [−Qr·Qp⁻¹ | I₆]`.
fn left_elim_matrix(q: &SMat<14, 8>, pivot: &[usize; 8]) -> Option<SMat<6, 14>> {
    let rest: ArrayVec<usize, 6> = (0..14).filter(|r| !pivot.contains(r)).collect();
    debug_assert_eq!(rest.len(), 6);
    let mut qp = SMat::<8, 8>::zeros();
    for (a, &pr) in pivot.iter().enumerate() {
        for j in 0..8 {
            qp.set(a, j, q.at(pr, j));
        }
    }
    let mut qr = SMat::<6, 8>::zeros();
    for (a, &rr) in rest.iter().enumerate() {
        for j in 0..8 {
            qr.set(a, j, q.at(rr, j));
        }
    }
    // X = Qr · Qp⁻¹  =>  Xᵀ = (Qpᵀ)⁻¹ Qrᵀ ; solve Qpᵀ Y = Qrᵀ, X = Yᵀ.
    let x = gauss_solve(qp.transpose(), qr.transpose())?.transpose(); // 6×8
    let mut n = SMat::<6, 14>::zeros();
    for a in 0..6 {
        for (b, &pc) in pivot.iter().enumerate() {
            n.set(a, pc, -x.at(a, b));
        }
        n.set(a, rest[a], 1.0);
    }
    Some(n)
}

/// 9×9 map taking the sin/cos basis `[s4s5,s4c5,c4s5,c4c5,s4,c4,s5,c5,1]` to the
/// half-angle-cleared monomial basis `[x4²x5²,x4²x5,x4x5²,x4x5,x4²,x5²,x4,x5,1]`
/// (after multiplying each equation by `(1+x4²)(1+x5²)`).
fn halfangle_map() -> SMat<9, 9> {
    let mut t = SMat::<9, 9>::zeros();
    t.set(0, 3, 4.0);
    t.set(1, 2, -2.0);
    t.set(1, 6, 2.0);
    t.set(2, 1, -2.0);
    t.set(2, 7, 2.0);
    t.set(3, 0, 1.0);
    t.set(3, 4, -1.0);
    t.set(3, 5, -1.0);
    t.set(3, 8, 1.0);
    t.set(4, 2, 2.0);
    t.set(4, 6, 2.0);
    t.set(5, 0, -1.0);
    t.set(5, 4, -1.0);
    t.set(5, 5, 1.0);
    t.set(5, 8, 1.0);
    t.set(6, 1, 2.0);
    t.set(6, 7, 2.0);
    t.set(7, 0, -1.0);
    t.set(7, 4, 1.0);
    t.set(7, 5, -1.0);
    t.set(7, 8, 1.0);
    t.set(8, 0, 1.0);
    t.set(8, 4, 1.0);
    t.set(8, 5, 1.0);
    t.set(8, 8, 1.0);
    t
}

/// Build the 12×12 dialytic matrix from `E9` (6×9, rows are polynomials in the
/// `[x4²x5²,…,1]` monomials): each row appears unmultiplied and multiplied by
/// `x4`, expanding into the 12-monomial basis.
fn dialytic_12(e9: &SMat<6, 9>) -> SMat<12, 12> {
    let mut m = SMat::<12, 12>::zeros();
    for i in 0..6 {
        let c: [f64; 9] = std::array::from_fn(|j| e9.at(i, j));
        // unmultiplied row
        m.set(i, 3, c[0]);
        m.set(i, 4, c[1]);
        m.set(i, 6, c[2]);
        m.set(i, 7, c[3]);
        m.set(i, 5, c[4]);
        m.set(i, 9, c[5]);
        m.set(i, 8, c[6]);
        m.set(i, 10, c[7]);
        m.set(i, 11, c[8]);
        // multiplied by x4
        m.set(i + 6, 0, c[0]);
        m.set(i + 6, 1, c[1]);
        m.set(i + 6, 3, c[2]);
        m.set(i + 6, 4, c[3]);
        m.set(i + 6, 2, c[4]);
        m.set(i + 6, 6, c[5]);
        m.set(i + 6, 5, c[6]);
        m.set(i + 6, 7, c[7]);
        m.set(i + 6, 8, c[8]);
    }
    m
}

/// `E9(x3)` at a single x3 from the affine `P` model and fixed `N`.
fn e9_at_x3(model: &PqModel, n: &SMat<6, 14>, map: &SMat<9, 9>, x3: f64) -> SMat<6, 9> {
    let theta3 = 2.0 * x3.atan();
    let (s, c) = theta3.sin_cos();
    let mut p = SMat::<14, 9>::zeros();
    for k in 0..14 {
        for j in 0..9 {
            p.d[k][j] = model.ps.d[k][j] * s + model.pc.d[k][j] * c + model.p1.d[k][j];
        }
    }
    let e45 = n.matmul(&p); // 6×9
    e45.matmul(map) // 6×9
}

/// Coefficient triple `(M0, M1, M2)` of `Σ(x3) = M0 + M1 x3 + M2 x3²`, the
/// 12×12 matrix quadratic obtained after clearing `(1+x3²)`.
fn sigma_coeffs(
    model: &PqModel,
    n: &SMat<6, 14>,
    map: &SMat<9, 9>,
) -> (SMat<12, 12>, SMat<12, 12>, SMat<12, 12>) {
    // (1+x3²)P = (P1−Pc)x3² + 2Ps·x3 + (P1+Pc)
    let mut p2 = SMat::<14, 9>::zeros();
    let mut p1l = SMat::<14, 9>::zeros();
    let mut p0 = SMat::<14, 9>::zeros();
    for k in 0..14 {
        for j in 0..9 {
            p2.d[k][j] = model.p1.d[k][j] - model.pc.d[k][j];
            p1l.d[k][j] = 2.0 * model.ps.d[k][j];
            p0.d[k][j] = model.p1.d[k][j] + model.pc.d[k][j];
        }
    }
    let e2 = n.matmul(&p2).matmul(map);
    let e1 = n.matmul(&p1l).matmul(map);
    let e0 = n.matmul(&p0).matmul(map);
    (dialytic_12(&e0), dialytic_12(&e1), dialytic_12(&e2))
}

// ----- root finding --------------------------------------------------------

/// Eigenvalues (complex) of a real square matrix via faer.
fn eig_real(m: &DMat) -> Vec<(f64, f64)> {
    let n = m.r;
    let fm = Mat::<f64>::from_fn(n, n, |i, j| m.at(i, j));
    match fm.eigenvalues() {
        Ok(v) => v.iter().map(|z| (z.re, z.im)).collect(),
        Err(_) => Vec::new(),
    }
}

/// Generalized eigenvalues `λ` of pencil `(A, B)` (`det(A − λB) = 0`) via faer.
fn gen_eig(a: &DMat, b: &DMat) -> Vec<(f64, f64)> {
    let n = a.r;
    let fa = Mat::<f64>::from_fn(n, n, |i, j| a.at(i, j));
    let fb = Mat::<f64>::from_fn(n, n, |i, j| b.at(i, j));
    match fa.generalized_eigen(fb.as_ref()) {
        Ok(g) => {
            let sa = g.S_a();
            let sb = g.S_b();
            let sav = sa.column_vector();
            let sbv = sb.column_vector();
            (0..n)
                .map(|i| {
                    let na = sav[i];
                    let de = sbv[i];
                    let denom = de.re * de.re + de.im * de.im;
                    if denom < 1e-300 {
                        (f64::INFINITY, 0.0)
                    } else {
                        (
                            (na.re * de.re + na.im * de.im) / denom,
                            (na.im * de.re - na.re * de.im) / denom,
                        )
                    }
                })
                .collect()
        }
        Err(_) => Vec::new(),
    }
}

/// Real x3 roots of `det Σ(x3) = 0` via the 24×24 first-companion pencil of the
/// 12×12 quadratic `Σ = M0 + M1 x + M2 x²`.
fn x3_roots(m0: &SMat<12, 12>, m1: &SMat<12, 12>, m2: &SMat<12, 12>, cfg: &RrConfig) -> Vec<f64> {
    let n = 12;
    let big = 2 * n;
    let mut a = DMat::zeros(big, big);
    let mut b = DMat::zeros(big, big);
    // A = [[-M1, -M0], [I, 0]],  B = [[M2, 0], [0, I]]
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, -m1.at(i, j));
            a.set(i, n + j, -m0.at(i, j));
            b.set(i, j, m2.at(i, j));
        }
        a.set(n + i, i, 1.0);
        b.set(n + i, n + i, 1.0);
    }
    let mut roots: Vec<f64> = gen_eig(&a, &b)
        .into_iter()
        .filter(|(re, im)| re.is_finite() && im.abs() <= cfg.imag_tol && re.abs() <= cfg.root_bound)
        .map(|(re, _)| re)
        .collect();
    roots.sort_by(|x, y| x.partial_cmp(y).unwrap());
    roots.dedup_by(|x, y| (*x - *y).abs() <= 1e-5);
    roots
}

// ----- (x4,x5) recovery from E9 via pairwise resultants --------------------

fn poly_trim(p: &[f64]) -> Vec<f64> {
    let mut k = p.len();
    while k > 1 && p[k - 1].abs() <= 1e-13 {
        k -= 1;
    }
    p[..k].to_vec()
}

fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut o = vec![0.0; a.len() + b.len() - 1];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            o[i + j] += av * bv;
        }
    }
    poly_trim(&o)
}

fn poly_addw(a: &[f64], b: &[f64], w: f64) -> Vec<f64> {
    let n = a.len().max(b.len());
    let mut o = vec![0.0; n];
    for i in 0..a.len() {
        o[i] += a[i];
    }
    for i in 0..b.len() {
        o[i] += w * b[i];
    }
    poly_trim(&o)
}

fn poly_eval(a: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    let mut p = 1.0;
    for &c in a {
        acc += c * p;
        p *= x;
    }
    acc
}

/// Real roots of an ascending-coefficient polynomial via companion eigenvalues.
fn poly_real_roots(coeffs: &[f64], imag_tol: f64) -> Vec<f64> {
    let c = poly_trim(coeffs);
    if c.len() <= 1 {
        return vec![];
    }
    let deg = c.len() - 1;
    let lead = c[deg];
    // companion matrix (deg×deg)
    let mut comp = DMat::zeros(deg, deg);
    for i in 0..deg {
        comp.set(0, i, -c[deg - 1 - i] / lead);
    }
    for i in 1..deg {
        comp.set(i, i - 1, 1.0);
    }
    eig_real(&comp)
        .into_iter()
        .filter(|(_, im)| im.abs() <= imag_tol)
        .map(|(re, _)| re)
        .collect()
}

fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() <= 1e-12 {
        if b.abs() <= 1e-12 {
            return vec![];
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < -1e-10 {
        return vec![];
    }
    let sd = disc.max(0.0).sqrt();
    vec![(-b + sd) / (2.0 * a), (-b - sd) / (2.0 * a)]
}

/// Recover `(x4, x5)` pairs from the 6×9 `E9` system using pairwise resultants
/// in x5, then back-substitution. Verified against all six rows.
fn solve_x4x5(e9: &SMat<6, 9>) -> Vec<(f64, f64)> {
    // row poly in x5: A(x4) x5² + B(x4) x5 + C(x4), with A,B,C ascending in x4.
    let rows: [([f64; 3], [f64; 3], [f64; 3]); 6] = std::array::from_fn(|i| {
        let r: [f64; 9] = std::array::from_fn(|j| e9.at(i, j));
        let a = [r[5], r[2], r[0]];
        let b = [r[7], r[3], r[1]];
        let c = [r[8], r[6], r[4]];
        (a, b, c)
    });
    let mut cands: Vec<(f64, f64)> = Vec::new();
    let push = |x4: f64, x5: f64, cands: &mut Vec<(f64, f64)>| {
        if !x4.is_finite() || !x5.is_finite() || x4.abs() > 1e8 || x5.abs() > 1e8 {
            return;
        }
        if cands
            .iter()
            .any(|&(u, v)| (u - x4).abs() <= 1e-7 && (v - x5).abs() <= 1e-7)
        {
            return;
        }
        cands.push((x4, x5));
    };
    for i in 0..6 {
        for j in (i + 1)..6 {
            let (a1, b1, c1) = (&rows[i].0[..], &rows[i].1[..], &rows[i].2[..]);
            let (a2, b2, c2) = (&rows[j].0[..], &rows[j].1[..], &rows[j].2[..]);
            // Res = A2²C1² + A1²C2² − 2A1A2C1C2 + A1B2²C1 + A2B1²C2 − A2B1B2C1 − A1B1B2C2
            let mut res = poly_mul(&poly_mul(a2, a2), &poly_mul(c1, c1));
            res = poly_addw(&res, &poly_mul(&poly_mul(a1, a1), &poly_mul(c2, c2)), 1.0);
            res = poly_addw(&res, &poly_mul(&poly_mul(a1, a2), &poly_mul(c1, c2)), -2.0);
            res = poly_addw(&res, &poly_mul(&poly_mul(a1, c1), &poly_mul(b2, b2)), 1.0);
            res = poly_addw(&res, &poly_mul(&poly_mul(a2, c2), &poly_mul(b1, b1)), 1.0);
            res = poly_addw(&res, &poly_mul(&poly_mul(a2, c1), &poly_mul(b1, b2)), -1.0);
            res = poly_addw(&res, &poly_mul(&poly_mul(a1, c2), &poly_mul(b1, b2)), -1.0);
            let res = poly_trim(&res);
            if res.len() <= 1 {
                continue;
            }
            for x4 in poly_real_roots(&res, 1e-7) {
                let av = poly_eval(a1, x4);
                let bv = poly_eval(b1, x4);
                let cv = poly_eval(c1, x4);
                for x5 in solve_quadratic(av, bv, cv) {
                    // verify against all six rows
                    let vec = [
                        x4 * x4 * x5 * x5,
                        x4 * x4 * x5,
                        x4 * x5 * x5,
                        x4 * x5,
                        x4 * x4,
                        x5 * x5,
                        x4,
                        x5,
                        1.0,
                    ];
                    let mut worst = 0.0f64;
                    for rr in 0..6 {
                        let mut s = 0.0;
                        for (k, &vk) in vec.iter().enumerate() {
                            s += e9.at(rr, k) * vk;
                        }
                        worst = worst.max(s.abs());
                    }
                    if worst <= 1e-3 {
                        push(x4, x5, &mut cands);
                    }
                }
            }
        }
    }
    cands
}

/// θ1,θ2 from the linear system `Q·m12 = P·m45(θ4,θ5)` (least squares).
fn recover_theta12(p: &SMat<14, 9>, q: &SMat<14, 8>, t4: f64, t5: f64) -> (f64, f64) {
    let m = m45(t4, t5);
    let mut rhs = SMat::<14, 1>::zeros();
    for k in 0..14 {
        let mut s = 0.0;
        for (j, &mj) in m.iter().enumerate() {
            s += p.at(k, j) * mj;
        }
        rhs.set(k, 0, s);
    }
    // normal equations (QᵀQ) n = Qᵀ rhs
    let qt = q.transpose(); // 8×14
    let ata = qt.matmul(q); // 8×8
    let atb = qt.matmul(&rhs); // 8×1
    let n = match gauss_solve(ata, atb) {
        Some(x) => x,
        None => return (0.0, 0.0),
    };
    let (s1, c1) = normalize_sc(n.at(4, 0), n.at(5, 0));
    let (s2, c2) = normalize_sc(n.at(6, 0), n.at(7, 0));
    (normalize_angle(s1.atan2(c1)), normalize_angle(s2.atan2(c2)))
}

/// θ6 from the last link: `RotZ(θ6) = (A1…A5)⁻¹·T·C6⁻¹`, read off (0,0),(1,0).
/// Right-multiplying by `C6⁻¹` strips the constant part so the residual is a
/// pure `RotZ(θ6)`; for the DH case `C6 = ais(joint)` whose RotX leaves column 0
/// unchanged, so this reduces to the textbook read-off.
fn recover_theta6(c: &[M4; 6], target: &M4, q15: &[f64; 5]) -> f64 {
    let mut t = m4_identity();
    for i in 0..5 {
        t = m4_mul(&t, &a_link(&c[i], q15[i]));
    }
    let a6 = m4_mul(&m4_inv_se3(&t), target);
    let rotz = m4_mul(&a6, &m4_inv_se3(&c[5]));
    normalize_angle(rotz[1][0].atan2(rotz[0][0]))
}

fn pose_residual(c: &[M4; 6], target: &M4, q: &[f64; 6]) -> f64 {
    let fk = fk_screw(c, q);
    let mut s = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let d = fk[i][j] - target[i][j];
            s += d * d;
        }
    }
    s.sqrt()
}

/// Solve general-6R IK for a screw chain `A_i = RotZ(θ_i)·C_i` and target pose.
/// Returns every distinct, FK-verified joint solution (up to 16). This is the
/// validated core; [`solve_dh`] and [`solve_kinspec`] are thin wrappers.
pub fn solve_screw(c: &[M4; 6], target: &M4, cfg: &RrConfig) -> Vec<[f64; 6]> {
    let finv = FInv::new();
    let model = build_pq_model(c, target, &finv);
    let pivot = match select_pivot_rows(&model.q) {
        Some(p) => p,
        None => return Vec::new(),
    };
    let n = match left_elim_matrix(&model.q, &pivot) {
        Some(n) => n,
        None => return Vec::new(),
    };
    let map = halfangle_map();
    let (m0, m1, m2) = sigma_coeffs(&model, &n, &map);
    let roots = x3_roots(&m0, &m1, &m2, cfg);

    let mut sols: Vec<[f64; 6]> = Vec::new();
    for x3 in roots {
        let theta3 = normalize_angle(2.0 * x3.atan());
        let (p, q) = pq_at_x3(c, target, x3, &finv);
        let e9 = e9_at_x3(&model, &n, &map, x3);
        for (x4, x5) in solve_x4x5(&e9) {
            let theta4 = normalize_angle(2.0 * x4.atan());
            let theta5 = normalize_angle(2.0 * x5.atan());
            let (theta1, theta2) = recover_theta12(&p, &q, theta4, theta5);
            let theta6 = recover_theta6(c, target, &[theta1, theta2, theta3, theta4, theta5]);
            let cand = [
                normalize_angle(theta1),
                normalize_angle(theta2),
                theta3,
                theta4,
                theta5,
                theta6,
            ];
            if pose_residual(c, target, &cand) <= cfg.residual_tol {
                let dup = sols.iter().any(|s| {
                    (0..6)
                        .map(|k| normalize_angle(s[k] - cand[k]).abs())
                        .fold(0.0f64, f64::max)
                        <= cfg.dedup_tol
                });
                if !dup {
                    sols.push(cand);
                }
            }
        }
    }
    sols
}

/// Solve general-6R IK for a classic-DH chain (`A_i = RotZ(θ_i)·ais(joint_i)`)
/// and target pose. Thin wrapper over [`solve_screw`].
pub fn solve_dh(dh: &[DhJoint; 6], target: &M4, cfg: &RrConfig) -> Vec<[f64; 6]> {
    let c: [M4; 6] = std::array::from_fn(|i| ais(&dh[i]));
    solve_screw(&c, target, cfg)
}

// ----- KinSpec entry: derive the screw decomposition + solve --------------

use deke_types::SRobotQ;
use deke_types::{JointSpec, KinSpec};
use glam::{DAffine3, DMat3, DVec3};

/// Why a [`KinSpec`] cannot be solved by the general-6R RR/MC solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RrSpecError {
    /// A prismatic joint was present; this solver handles 6R (all-revolute).
    PrismaticJoint(usize),
}

impl std::fmt::Display for RrSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrismaticJoint(i) => {
                write!(
                    f,
                    "joint {i} is prismatic; general-6R RR solver requires all-revolute"
                )
            }
        }
    }
}

impl std::error::Error for RrSpecError {}

fn affine_to_m4(a: &DAffine3) -> M4 {
    let m = a.matrix3;
    let t = a.translation;
    [
        [m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x],
        [m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y],
        [m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat3_to_m4(m: &DMat3) -> M4 {
    [
        [m.x_axis.x, m.y_axis.x, m.z_axis.x, 0.0],
        [m.x_axis.y, m.y_axis.y, m.z_axis.y, 0.0],
        [m.x_axis.z, m.y_axis.z, m.z_axis.z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// A rotation `R` with `R·ẑ = axis` (axis assumed unit). Columns `[n1, n2, axis]`
/// form a right-handed orthonormal frame, so conjugating `RotZ(θ)` by `R`
/// reproduces a rotation about `axis`.
fn frame_mapping_z_to(axis: DVec3) -> DMat3 {
    let z = axis.normalize();
    let t = if z.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let n1 = (t - z * t.dot(z)).normalize();
    let n2 = z.cross(n1);
    DMat3::from_cols(n1, n2, z)
}

/// Decompose a 6R `KinSpec` into the screw constants `C_i` (so each joint is
/// `RotZ(θ_i)·C_i`), the leading constant `L`, base `B`, and tool `E`, such that
/// the chain pose is `B · L · (∏ RotZ(θ_i)·C_i) · E`. The solver target is then
/// `L⁻¹ B⁻¹ pose E⁻¹`.
fn screw_from_kinspec(spec: &KinSpec<f64, 6>) -> Result<([M4; 6], M4, M4, M4), RrSpecError> {
    let b = affine_to_m4(&spec.base_to_first);
    let e = affine_to_m4(&spec.end_to_ee);

    let mut g = [m4_identity(); 6];
    let mut r = [m4_identity(); 6];
    for i in 0..6 {
        let axis = match spec.joints[i].1 {
            JointSpec::Revolute { axis_local } => axis_local,
            JointSpec::Prismatic { .. } => return Err(RrSpecError::PrismaticJoint(i)),
        };
        g[i] = affine_to_m4(&spec.joints[i].0);
        r[i] = mat3_to_m4(&frame_mapping_z_to(axis));
    }

    // L = G1 R1 ; C_i = R_iᵀ G_{i+1} R_{i+1} (i<6) ; C_6 = R_6ᵀ.
    let l = m4_mul(&g[0], &r[0]);
    let mut c = [m4_identity(); 6];
    for i in 0..5 {
        let rt = m4_inv_se3(&r[i]);
        c[i] = m4_mul(&m4_mul(&rt, &g[i + 1]), &r[i + 1]);
    }
    c[5] = m4_inv_se3(&r[5]);
    Ok((c, l, b, e))
}

/// Solve general-6R IK for any all-revolute 6R [`KinSpec`] at end-effector pose
/// `pose` (a `glam::DMat4` in the same frame as `KinSpec` FK). Returns every
/// distinct, FK-verified joint solution as [`SRobotQ`], or [`RrSpecError`] if
/// the spec is not solvable by this method (e.g. contains a prismatic joint).
pub fn solve_kinspec(
    spec: &KinSpec<f64, 6>,
    pose: glam::DMat4,
    cfg: &RrConfig,
) -> Result<Vec<SRobotQ<6, f64>>, RrSpecError> {
    let (c, l, b, e) = screw_from_kinspec(spec)?;
    let pose_m4 = {
        let m = pose;
        [
            [m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x],
            [m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y],
            [m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    };
    // target = L⁻¹ B⁻¹ pose E⁻¹
    let target = m4_mul(
        &m4_mul(&m4_mul(&m4_inv_se3(&l), &m4_inv_se3(&b)), &pose_m4),
        &m4_inv_se3(&e),
    );
    let sols = solve_screw(&c, &target, cfg);
    Ok(sols
        .into_iter()
        .map(SRobotQ::<6, f64>::from_array)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deg(x: f64) -> f64 {
        x * PI / 180.0
    }

    /// FK pose-residual for a DH chain (the screw-form `pose_residual` takes the
    /// `C_i` constants; here we want to verify against DH joints directly).
    fn dh_residual(dh: &[DhJoint; 6], target: &M4, q: &[f64; 6]) -> f64 {
        let fk = fk_dh(dh, q);
        let mut s = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                let d = fk[i][j] - target[i][j];
                s += d * d;
            }
        }
        s.sqrt()
    }

    /// Published Raghavan–Roth (1990) numerical example. DH (a, alpha, d) and EE
    /// pose from the ambuj-Shahi `example1` test vector; the paper reports 16
    /// solutions, two of which are real.
    #[allow(clippy::approx_constant)] // published DH twist values, not an approximation of π/4
    fn rr1990() -> ([DhJoint; 6], M4) {
        let a = [0.8, 1.2, 0.33, 1.8, 0.6, 2.2];
        let alpha = [0.349066, 0.541052, 0.785398, 1.41372, 0.20944, 1.74533];
        let d = [0.9, 3.7, 1.0, 0.5, 2.1, 0.63];
        let dh = std::array::from_fn(|i| DhJoint {
            a: a[i],
            alpha: alpha[i],
            d: d[i],
        });
        // EE pose from eemat_Raghavan_Roth.csv. The CSV stores the rotation
        // column-major, so the 3×3 block is transposed into row-major here;
        // the translation column is unchanged. (Verified: this matches the FK
        // of the published real generating angles to 1e-4.)
        let target = [
            [
                0.354937475307970,
                0.461639573991743,
                -0.812962663562556,
                6.82151837150213,
            ],
            [
                0.876709605247149,
                0.137616185817977,
                0.460914366741046,
                1.46146704002829,
            ],
            [
                0.324653132880913,
                -0.876327957516839,
                -0.355878707125018,
                5.36950521368663,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ];
        (dh, target)
    }

    #[test]
    fn dh_fk_matches_known_real_solution() {
        // Solution #10 from the published table is exactly real.
        let (dh, _) = rr1990();
        let q = [
            deg(14.000158),
            deg(29.699750),
            deg(-45.000135),
            deg(71.000293),
            deg(-63.000511),
            deg(10.000427),
        ];
        let fk = fk_dh(&dh, &q);
        let (_, target) = rr1990();
        let mut worst = 0.0f64;
        for i in 0..3 {
            for j in 0..4 {
                worst = worst.max((fk[i][j] - target[i][j]).abs());
            }
        }
        assert!(
            worst < 1e-4,
            "published real solution must reproduce the target pose, worst={worst}"
        );
    }

    #[test]
    fn solves_rr1990_real_solutions() {
        let (dh, target) = rr1990();
        let cfg = RrConfig::default();
        let sols = solve_dh(&dh, &target, &cfg);

        // Every returned solution must be a genuine IK solution.
        for s in &sols {
            assert!(dh_residual(&dh, &target, s) < 1e-6, "stale solution {s:?}");
        }

        // The two published real solutions must both be recovered.
        let real_targets = [
            [
                deg(13.109881),
                deg(50.992641),
                deg(-72.044721),
                deg(72.065198),
                deg(-7.195753),
                deg(-37.852729),
            ],
            [
                deg(14.000158),
                deg(29.699750),
                deg(-45.000135),
                deg(71.000293),
                deg(-63.000511),
                deg(10.000427),
            ],
        ];
        for rt in &real_targets {
            let found = sols.iter().any(|s| {
                (0..6)
                    .map(|k| normalize_angle(s[k] - rt[k]).abs())
                    .fold(0.0f64, f64::max)
                    < 1e-3
            });
            assert!(
                found,
                "did not recover published real solution {rt:?}; got {} solutions",
                sols.len()
            );
        }
    }

    #[test]
    fn roundtrip_random_generic_chain() {
        // A generic chain (no parallel/intersecting axes) and a planted q.
        let dh = [
            DhJoint {
                a: 0.32,
                alpha: 0.70,
                d: 0.18,
            },
            DhJoint {
                a: 0.25,
                alpha: -0.90,
                d: 0.21,
            },
            DhJoint {
                a: 0.29,
                alpha: 0.80,
                d: 0.14,
            },
            DhJoint {
                a: 0.22,
                alpha: -1.10,
                d: 0.19,
            },
            DhJoint {
                a: 0.18,
                alpha: 0.60,
                d: 0.11,
            },
            DhJoint {
                a: 0.15,
                alpha: -0.70,
                d: 0.17,
            },
        ];
        let q_true = [0.60, -1.00, 0.90, -0.80, 1.20, -0.40];
        let target = fk_dh(&dh, &q_true);
        let sols = solve_dh(&dh, &target, &RrConfig::default());
        assert!(!sols.is_empty(), "no solutions for planted pose");
        for s in &sols {
            assert!(dh_residual(&dh, &target, s) < 1e-6, "stale solution {s:?}");
        }
        let found = sols.iter().any(|s| {
            (0..6)
                .map(|k| normalize_angle(s[k] - q_true[k]).abs())
                .fold(0.0f64, f64::max)
                < 1e-3
        });
        assert!(
            found,
            "planted solution not recovered; got {} solutions",
            sols.len()
        );
    }

    /// A 6R `KinSpec` with arbitrary (non-DH, non-Z) joint axes. The solver must
    /// (a) reproduce the planted configuration and (b) FK-verify every returned
    /// root against the *KinSpec* forward kinematics — proving the screw
    /// extraction is consistent with `KinSpec` FK.
    #[test]
    fn kinspec_roundtrip_arbitrary_axes() {
        use deke_types::JointSpec;
        use glam::{DAffine3, DMat4, DVec3};

        let axes = [
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(0.0, 1.0, 0.3).normalize(),
            DVec3::new(0.2, 1.0, 0.0).normalize(),
            DVec3::new(1.0, 0.2, 0.4).normalize(),
            DVec3::new(0.0, 1.0, 0.5).normalize(),
            DVec3::new(0.3, 0.2, 1.0).normalize(),
        ];
        let offs = [
            DVec3::new(0.0, 0.0, 0.30),
            DVec3::new(0.10, 0.02, 0.05),
            DVec3::new(0.30, 0.0, 0.04),
            DVec3::new(0.0, 0.05, 0.28),
            DVec3::new(0.06, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.07),
        ];
        let joints = std::array::from_fn(|i| {
            (
                DAffine3::from_translation(offs[i]),
                JointSpec::Revolute {
                    axis_local: axes[i],
                },
            )
        });
        let spec = KinSpec::<f64, 6>::new(
            DAffine3::from_translation(DVec3::new(0.0, 0.0, 0.1)),
            joints,
            DAffine3::from_translation(DVec3::new(0.0, 0.0, 0.12)),
        );

        // KinSpec FK (mirrors deke_types forward_pass) to build the target.
        let kinspec_fk = |q: &[f64; 6]| -> DMat4 {
            let mut t = spec.base_to_first;
            for (joint, &qi) in spec.joints.iter().zip(q.iter()) {
                t *= joint.0;
                let axis = match joint.1 {
                    JointSpec::Revolute { axis_local } => axis_local.normalize(),
                    JointSpec::Prismatic { .. } => unreachable!(),
                };
                t *= DAffine3::from_axis_angle(axis, qi);
            }
            t *= spec.end_to_ee;
            DMat4::from(t)
        };

        let q_true = [0.4, -0.8, 1.0, -0.5, 0.9, -0.3];
        let pose = kinspec_fk(&q_true);

        let sols = solve_kinspec(&spec, pose, &RrConfig::default()).expect("revolute spec");
        assert!(!sols.is_empty(), "no KinSpec solutions");

        for s in &sols {
            let fk = kinspec_fk(&s.0);
            let mut worst = 0.0f64;
            let a = fk.to_cols_array();
            let b = pose.to_cols_array();
            for k in 0..16 {
                worst = worst.max((a[k] - b[k]).abs());
            }
            assert!(
                worst < 1e-6,
                "KinSpec solution does not reproduce pose: {worst}"
            );
        }

        let found = sols.iter().any(|s| {
            (0..6)
                .map(|k| normalize_angle(s.0[k] - q_true[k]).abs())
                .fold(0.0f64, f64::max)
                < 1e-3
        });
        assert!(
            found,
            "planted KinSpec solution not recovered; got {} solutions",
            sols.len()
        );
    }

    #[test]
    fn kinspec_rejects_prismatic() {
        use deke_types::JointSpec;
        use glam::{DAffine3, DMat4, DVec3};
        let joints = std::array::from_fn(|i| {
            let js = if i == 2 {
                JointSpec::Prismatic {
                    axis_local: DVec3::Z,
                }
            } else {
                JointSpec::Revolute {
                    axis_local: DVec3::Z,
                }
            };
            (DAffine3::from_translation(DVec3::new(0.0, 0.0, 0.2)), js)
        });
        let spec = KinSpec::<f64, 6>::new(DAffine3::IDENTITY, joints, DAffine3::IDENTITY);
        let r = solve_kinspec(&spec, DMat4::IDENTITY, &RrConfig::default());
        assert_eq!(r, Err(RrSpecError::PrismaticJoint(2)));
    }

    /// On a spherical-wrist chain that the analytical solver fully supports, the
    /// generic RR/MC solver must capture *the same set* of solutions: for every
    /// pose, the analytical solutions are a subset of the generic ones (the
    /// generic solver may additionally surface branches the analytical path
    /// drops at singularities, but must never miss one).
    #[test]
    fn matches_analytical_solution_set_puma() {
        use crate::{DHJoint, IkStrategy, Kinematics};
        use deke_types::SRobotQ;
        use deke_types::{ContinuousFKChain, FKChain, IkSolver};
        use glam::DMat4;

        let pi = std::f64::consts::PI;
        let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
        let a = [0.0, 0.4318, -0.0203, 0.0, 0.0, 0.0];
        let d = [0.6718, 0.1397, 0.0, 0.4318, 0.0, 0.0565];
        let chain: Kinematics<6, f64> = Kinematics::from_dh(
            std::array::from_fn(|i| DHJoint {
                a: a[i],
                alpha: alpha[i],
                d: d[i],
                theta_offset: 0.0,
            }),
            crate::JointLimits::symmetric(100.0),
            &[],
        );
        let spec = chain.structure();
        // Puma must resolve to the analytic strategy; `chain.ik` then exercises
        // the closed-form path we compare the generic solver against.
        assert!(
            matches!(chain.ik_diagnostic().strategy, IkStrategy::Analytic { .. }),
            "Puma should resolve to an analytic decomposition, got {:?}",
            chain.ik_diagnostic().strategy
        );
        let cfg = RrConfig::default();

        let mut total_analytic = 0usize;
        let mut total_generic = 0usize;
        let mut poses = 0usize;
        let mut poses_with_gap = 0usize;

        for i in 0..40 {
            let s = (i as f64 / 40.0 - 0.5) * std::f64::consts::TAU;
            let q = SRobotQ::<6, f64>::from_array([
                0.7 * s,
                0.5 * (s + 0.4),
                -0.3 * (s - 0.2),
                0.6 * (s + 0.1),
                -0.4 * s,
                0.8 * (s - 0.3),
            ]);
            let ee = chain.fk_end(&q).unwrap();

            let analytic_sols = match chain.ik(ee).unwrap() {
                deke_types::IkOutcome::Solved(v) => v,
                deke_types::IkOutcome::Failed { .. } => continue,
            };
            if analytic_sols.is_empty() {
                continue;
            }
            poses += 1;

            let generic = solve_kinspec(&spec, DMat4::from(ee), &cfg).unwrap();
            total_analytic += analytic_sols.len();
            total_generic += generic.len();

            // The target is degenerate when its *generating* configuration is
            // wrist-singular (q5 ≈ 0 or ±π → joints 4 and 6 coaxial). At such a
            // pose the degree-16 characteristic polynomial gains repeated roots,
            // and the eigenvalue extraction collapses some discrete branches the
            // closed-form path still enumerates separately. This is the one
            // place the generic solver may legitimately under-count.
            let gen_q5 = normalize_angle(q.0[4]).abs();
            let wrist_singular = gen_q5 < 1e-3 || (gen_q5 - pi).abs() < 1e-3;

            let mut missing = 0usize;
            for asol in &analytic_sols {
                let matched = generic.iter().any(|g| {
                    (0..6)
                        .map(|k| normalize_angle(g.0[k] - asol.0[k]).abs())
                        .fold(0.0f64, f64::max)
                        < 1e-4
                });
                if !matched {
                    missing += 1;
                }
            }

            if wrist_singular {
                poses_with_gap += 1;
            } else {
                // Away from singularities the generic solver must capture every
                // analytical solution AND return the same count.
                assert_eq!(
                    missing,
                    0,
                    "pose {i}: generic solver missed {missing} of {} analytical solutions",
                    analytic_sols.len()
                );
                assert_eq!(
                    generic.len(),
                    analytic_sols.len(),
                    "pose {i}: generic count {} != analytic count {}",
                    generic.len(),
                    analytic_sols.len()
                );
            }
        }

        assert!(poses >= 30, "expected most poses reachable, got {poses}");
        println!(
            "puma: {poses} poses | analytic total={total_analytic} (avg {:.2}) | generic total={total_generic} (avg {:.2}) | wrist-singular poses (gap allowed)={poses_with_gap}",
            total_analytic as f64 / poses as f64,
            total_generic as f64 / poses as f64,
        );
    }
}
