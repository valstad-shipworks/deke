//! Small rectangular matrices the IK subproblems need but glam doesn't ship.
//!
//! Layout choice for `Mat3x2`: **columns** as two `DVec3`s.
//!
//! Every subproblem builds the matrix with the same idiom
//! ```ignore
//! let a = Mat3x2::from_columns(k.cross(p), -k.cross(&k.cross(p)));
//! ```
//! and then immediately does either `a.transpose_mul_vec3(v)` (returns
//! `DVec2`, the hot path — two `DVec3::dot`s, both inlined to scalar FMAs) or
//! `a.mul_vec2(v)` (returns `DVec3`, a single `DVec3 * f64 + DVec3 * f64`).
//! Storing the two columns as `DVec3` keeps construction free and lets both
//! ops collapse to direct field accesses with no shuffles.
//!
//! `Mat3x4` (used by `subproblem2extended`) and `Mat2x4` (used by
//! `subproblem6`) follow the same column-major convention.

use glam::{DMat2, DVec2, DVec3, DVec4};

/// 3-row × 2-column matrix stored as two column `DVec3`s.
#[derive(Clone, Copy, Debug)]
pub struct Mat3x2 {
    pub c0: DVec3,
    pub c1: DVec3,
}

impl Mat3x2 {
    #[inline]
    pub const fn from_columns(c0: DVec3, c1: DVec3) -> Self {
        Self { c0, c1 }
    }

    /// Build the standard subproblem projection `[k×p, p − k(k·p)]` for a
    /// **unit** axis `k`. Uses the identity `−k × (k × p) = (k·k) p − (k·p) k`
    /// (= `p − (k·p) k` when `|k| = 1`) to save one cross product per call —
    /// every subproblem hits this once.
    #[inline]
    pub fn perp_basis(k: DVec3, p: DVec3) -> Self {
        Self {
            c0: k.cross(p),
            c1: p - k * k.dot(p),
        }
    }

    /// `self.T * v` — projects a 3-D vector onto the 2-D plane defined by the
    /// two columns. The hot path of every subproblem.
    #[inline]
    pub fn transpose_mul_vec3(&self, v: DVec3) -> DVec2 {
        DVec2::new(self.c0.dot(v), self.c1.dot(v))
    }

    /// `self * v` — lift a 2-D coordinate back to 3-D.
    #[inline]
    pub fn mul_vec2(&self, v: DVec2) -> DVec3 {
        self.c0 * v.x + self.c1 * v.y
    }

    /// `self.T * self` — the 2×2 Gram matrix. Symmetric, but we return all 4
    /// entries for convenience.
    #[inline]
    pub fn transpose_mul_self(&self) -> DMat2 {
        let a = self.c0.dot(self.c0);
        let b = self.c0.dot(self.c1);
        let d = self.c1.dot(self.c1);
        DMat2::from_cols_array(&[a, b, b, d])
    }

    #[inline]
    pub fn column(&self, i: usize) -> DVec3 {
        match i {
            0 => self.c0,
            1 => self.c1,
            _ => panic!("Mat3x2::column index out of range"),
        }
    }

    #[inline]
    pub fn neg(self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
        }
    }
}

impl std::ops::Neg for Mat3x2 {
    type Output = Mat3x2;
    #[inline]
    fn neg(self) -> Self {
        Mat3x2::neg(self)
    }
}

impl std::ops::Mul<f64> for Mat3x2 {
    type Output = Mat3x2;
    #[inline]
    fn mul(self, s: f64) -> Mat3x2 {
        Mat3x2 {
            c0: self.c0 * s,
            c1: self.c1 * s,
        }
    }
}

/// 3-row × 4-column matrix, columns stored as `DVec3`. Used by
/// `subproblem2extended`.
#[derive(Clone, Copy, Debug)]
pub struct Mat3x4 {
    pub c0: DVec3,
    pub c1: DVec3,
    pub c2: DVec3,
    pub c3: DVec3,
}

impl Mat3x4 {
    #[inline]
    pub const fn from_columns(c0: DVec3, c1: DVec3, c2: DVec3, c3: DVec3) -> Self {
        Self { c0, c1, c2, c3 }
    }

    /// `self * v` (3×4 · 4 = 3).
    #[inline]
    pub fn mul_vec4(&self, v: DVec4) -> DVec3 {
        self.c0 * v.x + self.c1 * v.y + self.c2 * v.z + self.c3 * v.w
    }

    /// `self.T * v` (4×3 · 3 = 4).
    #[inline]
    pub fn transpose_mul_vec3(&self, v: DVec3) -> DVec4 {
        DVec4::new(
            self.c0.dot(v),
            self.c1.dot(v),
            self.c2.dot(v),
            self.c3.dot(v),
        )
    }
}

/// 2-row × 4-column matrix, columns stored as `DVec2`. Used by `subproblem6`.
#[derive(Clone, Copy, Debug)]
pub struct Mat2x4 {
    pub c0: DVec2,
    pub c1: DVec2,
    pub c2: DVec2,
    pub c3: DVec2,
}

impl Mat2x4 {
    #[inline]
    pub const fn from_columns(c0: DVec2, c1: DVec2, c2: DVec2, c3: DVec2) -> Self {
        Self { c0, c1, c2, c3 }
    }

    /// `self * v` (2×4 · 4 = 2).
    #[inline]
    pub fn mul_vec4(&self, v: DVec4) -> DVec2 {
        self.c0 * v.x + self.c1 * v.y + self.c2 * v.z + self.c3 * v.w
    }

    /// `self.T * v` (4×2 · 2 = 4).
    #[inline]
    pub fn transpose_mul_vec2(&self, v: DVec2) -> DVec4 {
        DVec4::new(
            self.c0.dot(v),
            self.c1.dot(v),
            self.c2.dot(v),
            self.c3.dot(v),
        )
    }

    #[inline]
    pub fn column(&self, i: usize) -> DVec2 {
        match i {
            0 => self.c0,
            1 => self.c1,
            2 => self.c2,
            3 => self.c3,
            _ => panic!("Mat2x4::column index out of range"),
        }
    }
}
