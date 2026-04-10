use ndarray::Array1;
use wide::{CmpGt, CmpLt, f32x8};

use crate::DekeError;

#[inline(always)]
fn simd_load(slice: &[f32], off: usize) -> f32x8 {
    let n = 8.min(slice.len().saturating_sub(off));
    let mut buf = [0.0; 8];
    buf[..n].copy_from_slice(&slice[off..off + n]);
    f32x8::new(buf)
}

#[inline(always)]
fn simd_store(v: f32x8, dst: &mut [f32], off: usize) {
    let n = 8.min(dst.len().saturating_sub(off));
    dst[off..off + n].copy_from_slice(&v.to_array()[..n]);
}

#[inline(always)]
fn simd_binop<const N: usize>(
    a: &[f32; N],
    b: &[f32; N],
    out: &mut [f32; N],
    op: fn(f32x8, f32x8) -> f32x8,
) {
    let mut off = 0;
    while off < N {
        simd_store(op(simd_load(a, off), simd_load(b, off)), out, off);
        off += 8;
    }
}

#[inline(always)]
fn simd_unaryop<const N: usize>(a: &[f32; N], out: &mut [f32; N], op: fn(f32x8) -> f32x8) {
    let mut off = 0;
    while off < N {
        simd_store(op(simd_load(a, off)), out, off);
        off += 8;
    }
}

#[inline(always)]
fn simd_scalarop<const N: usize>(
    a: &[f32; N],
    s: f32x8,
    out: &mut [f32; N],
    op: fn(f32x8, f32x8) -> f32x8,
) {
    let mut off = 0;
    while off < N {
        simd_store(op(simd_load(a, off), s), out, off);
        off += 8;
    }
}

#[inline(always)]
fn simd_hsum<const N: usize>(a: &[f32; N]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut off = 0;
    while off < N {
        acc += simd_load(a, off);
        off += 8;
    }
    acc.reduce_add()
}

#[inline(always)]
fn simd_load_neg_inf(slice: &[f32], off: usize) -> f32x8 {
    let n = 8.min(slice.len().saturating_sub(off));
    let mut buf = [f32::NEG_INFINITY; 8];
    buf[..n].copy_from_slice(&slice[off..off + n]);
    f32x8::new(buf)
}

#[inline(always)]
fn simd_load_inf(slice: &[f32], off: usize) -> f32x8 {
    let n = 8.min(slice.len().saturating_sub(off));
    let mut buf = [f32::INFINITY; 8];
    buf[..n].copy_from_slice(&slice[off..off + n]);
    f32x8::new(buf)
}

#[inline(always)]
fn simd_dot<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut off = 0;
    while off < N {
        acc = simd_load(a, off).mul_add(simd_load(b, off), acc);
        off += 8;
    }
    acc.reduce_add()
}

pub type RobotQ = Array1<f32>;

/// Statically-sized joint configuration backed by `[f32; N]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SRobotQ<const N: usize>(pub [f32; N]);

impl<const N: usize> SRobotQ<N> {
    pub const fn zeros() -> Self {
        Self([0.0; N])
    }

    pub const fn from_array(arr: [f32; N]) -> Self {
        Self(arr)
    }

    pub const fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub const fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.0
    }

    pub fn to_robotq(&self) -> RobotQ {
        RobotQ::from(self.0.to_vec())
    }

    pub fn force_from_robotq(q: &RobotQ) -> Self {
        if let Ok(sq) = Self::try_from(q) {
            sq
        } else {
            let slice = q.as_slice().unwrap_or(&[]);
            let mut arr = [0.0; N];
            for i in 0..N {
                arr[i] = *slice.get(i).unwrap_or(&0.0);
            }
            Self(arr)
        }
    }

    pub fn norm(&self) -> f32 {
        if N <= 16 {
            self.dot(self).sqrt()
        } else {
            self.0.iter().map(|x| x * x).sum::<f32>().sqrt()
        }
    }

    pub fn dot(&self, other: &Self) -> f32 {
        if N <= 16 {
            simd_dot(&self.0, &other.0)
        } else {
            self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum()
        }
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = f(self.0[i]);
        }
        Self(out)
    }

    pub fn sum(&self) -> f32 {
        if N <= 16 {
            simd_hsum(&self.0)
        } else {
            self.0.iter().sum()
        }
    }

    pub fn splat(val: f32) -> Self {
        Self([val; N])
    }

    pub fn from_fn(f: impl Fn(usize) -> f32) -> Self {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = f(i);
        }
        Self(out)
    }

    pub fn norm_squared(&self) -> f32 {
        self.dot(self)
    }

    pub fn normalize(&self) -> Self {
        let n = self.norm();
        debug_assert!(n > 0.0, "cannot normalize zero-length SRobotQ");
        *self / n
    }

    pub fn distance(&self, other: &Self) -> f32 {
        (*self - *other).norm()
    }

    pub fn distance_squared(&self, other: &Self) -> f32 {
        (*self - *other).norm_squared()
    }

    pub fn abs(&self) -> Self {
        if N <= 16 {
            let mut out = [0.0; N];
            simd_unaryop(&self.0, &mut out, |a| a.abs());
            Self(out)
        } else {
            self.map(f32::abs)
        }
    }

    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        if N <= 16 {
            let mut out = [0.0; N];
            let mut off = 0;
            while off < N {
                let v = simd_load(&self.0, off);
                let lo = simd_load(&min.0, off);
                let hi = simd_load(&max.0, off);
                simd_store(v.fast_max(lo).fast_min(hi), &mut out, off);
                off += 8;
            }
            Self(out)
        } else {
            let mut out = [0.0; N];
            for i in 0..N {
                out[i] = self.0[i].clamp(min.0[i], max.0[i]);
            }
            Self(out)
        }
    }

    pub fn clamp_scalar(&self, min: f32, max: f32) -> Self {
        if N <= 16 {
            let mut out = [0.0; N];
            let lo = f32x8::splat(min);
            let hi = f32x8::splat(max);
            let mut off = 0;
            while off < N {
                let v = simd_load(&self.0, off);
                simd_store(v.fast_max(lo).fast_min(hi), &mut out, off);
                off += 8;
            }
            Self(out)
        } else {
            self.map(|x| x.clamp(min, max))
        }
    }

    pub fn max_element(&self) -> f32 {
        if N <= 16 {
            let mut acc = f32x8::splat(f32::NEG_INFINITY);
            let mut off = 0;
            while off < N {
                acc = acc.fast_max(simd_load_neg_inf(&self.0, off));
                off += 8;
            }
            let a = acc.to_array();
            a[0].max(a[1])
                .max(a[2].max(a[3]))
                .max(a[4].max(a[5]).max(a[6].max(a[7])))
        } else {
            self.0.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        }
    }

    pub fn min_element(&self) -> f32 {
        if N <= 16 {
            let mut acc = f32x8::splat(f32::INFINITY);
            let mut off = 0;
            while off < N {
                acc = acc.fast_min(simd_load_inf(&self.0, off));
                off += 8;
            }
            let a = acc.to_array();
            a[0].min(a[1])
                .min(a[2].min(a[3]))
                .min(a[4].min(a[5]).min(a[6].min(a[7])))
        } else {
            self.0.iter().copied().fold(f32::INFINITY, f32::min)
        }
    }

    pub fn linf_norm(&self) -> f32 {
        self.abs().max_element()
    }

    pub fn elementwise_mul(&self, other: &Self) -> Self {
        let mut out = [0.0; N];
        if N <= 16 {
            simd_binop(&self.0, &other.0, &mut out, |a, b| a * b);
        } else {
            for i in 0..N {
                out[i] = self.0[i] * other.0[i];
            }
        }
        Self(out)
    }

    pub fn elementwise_div(&self, other: &Self) -> Self {
        let mut out = [0.0; N];
        if N <= 16 {
            simd_binop(&self.0, &other.0, &mut out, |a, b| a / b);
        } else {
            for i in 0..N {
                out[i] = self.0[i] / other.0[i];
            }
        }
        Self(out)
    }

    pub fn zip_map(&self, other: &Self, f: impl Fn(f32, f32) -> f32) -> Self {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = f(self.0[i], other.0[i]);
        }
        Self(out)
    }

    pub fn sqrt(&self) -> Self {
        if N <= 16 {
            let mut out = [0.0; N];
            simd_unaryop(&self.0, &mut out, |a| a.sqrt());
            Self(out)
        } else {
            self.map(f32::sqrt)
        }
    }

    pub fn mul_add(&self, mul: &Self, add: &Self) -> Self {
        if N <= 16 {
            let mut out = [0.0; N];
            let mut off = 0;
            while off < N {
                let a = simd_load(&self.0, off);
                let m = simd_load(&mul.0, off);
                let d = simd_load(&add.0, off);
                simd_store(a.mul_add(m, d), &mut out, off);
                off += 8;
            }
            Self(out)
        } else {
            let mut out = [0.0; N];
            for i in 0..N {
                out[i] = self.0[i].mul_add(mul.0[i], add.0[i]);
            }
            Self(out)
        }
    }

    /// Returns `true` if any element of `self` is greater than the corresponding element of `other`.
    pub fn any_non_finite(&self) -> bool {
        let mut off = 0;
        while off < N {
            let v = simd_load(&self.0, off);
            let bad = v.is_nan() | v.is_inf();
            if (bad.to_bitmask() & Self::lane_mask(off)) != 0 {
                return true;
            }
            off += 8;
        }
        false
    }

    pub fn any_gt(&self, other: &Self) -> bool {
        let mut off = 0;
        while off < N {
            let a = simd_load(&self.0, off);
            let b = simd_load(&other.0, off);
            if (a.simd_gt(b).to_bitmask() & Self::lane_mask(off)) != 0 {
                return true;
            }
            off += 8;
        }
        false
    }

    /// Returns `true` if any element of `self` is less than the corresponding element of `other`.
    pub fn any_lt(&self, other: &Self) -> bool {
        let mut off = 0;
        while off < N {
            let a = simd_load(&self.0, off);
            let b = simd_load(&other.0, off);
            if (a.simd_lt(b).to_bitmask() & Self::lane_mask(off)) != 0 {
                return true;
            }
            off += 8;
        }
        false
    }

    #[inline(always)]
    const fn lane_mask(off: usize) -> u32 {
        let active = N.saturating_sub(off);
        if active >= 8 {
            0b11111111
        } else {
            (1 << active) - 1
        }
    }

    pub fn is_close(&self, other: &Self, tol: f32) -> bool {
        let diff = *self - *other;
        diff.dot(&diff).sqrt() < tol
    }

    pub fn interpolate(&self, other: &Self, t: f32) -> Self {
        *self + ((*other - *self) * t)
    }
}

impl<const N: usize> std::ops::Index<usize> for SRobotQ<N> {
    type Output = f32;
    #[inline]
    fn index(&self, i: usize) -> &f32 {
        &self.0[i]
    }
}

impl<const N: usize> std::ops::IndexMut<usize> for SRobotQ<N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        &mut self.0[i]
    }
}

impl<const N: usize> std::ops::Add for SRobotQ<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = [0.0; N];
        if N <= 16 {
            simd_binop(&self.0, &rhs.0, &mut out, |a, b| a + b);
        } else {
            for i in 0..N {
                out[i] = self.0[i] + rhs.0[i];
            }
        }
        Self(out)
    }
}

impl<const N: usize> std::ops::Sub for SRobotQ<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = [0.0; N];
        if N <= 16 {
            simd_binop(&self.0, &rhs.0, &mut out, |a, b| a - b);
        } else {
            for i in 0..N {
                out[i] = self.0[i] - rhs.0[i];
            }
        }
        Self(out)
    }
}

impl<const N: usize> std::ops::Neg for SRobotQ<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut out = [0.0; N];
        if N <= 16 {
            simd_unaryop(&self.0, &mut out, |a| f32x8::ZERO - a);
        } else {
            for i in 0..N {
                out[i] = -self.0[i];
            }
        }
        Self(out)
    }
}

impl<const N: usize> std::ops::Mul<f32> for SRobotQ<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        let mut out = [0.0; N];
        if N <= 16 {
            simd_scalarop(&self.0, f32x8::splat(rhs), &mut out, |a, s| a * s);
        } else {
            for i in 0..N {
                out[i] = self.0[i] * rhs;
            }
        }
        Self(out)
    }
}

impl<const N: usize> std::ops::Mul<SRobotQ<N>> for f32 {
    type Output = SRobotQ<N>;
    #[inline]
    fn mul(self, rhs: SRobotQ<N>) -> SRobotQ<N> {
        rhs * self
    }
}

impl<const N: usize> std::ops::Div<f32> for SRobotQ<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        let mut out = [0.0; N];
        if N <= 16 {
            simd_scalarop(&self.0, f32x8::splat(rhs), &mut out, |a, s| a / s);
        } else {
            for i in 0..N {
                out[i] = self.0[i] / rhs;
            }
        }
        Self(out)
    }
}

impl<const N: usize> std::ops::AddAssign for SRobotQ<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        if N <= 16 {
            let mut out = [0.0; N];
            simd_binop(&self.0, &rhs.0, &mut out, |a, b| a + b);
            self.0 = out;
        } else {
            for i in 0..N {
                self.0[i] += rhs.0[i];
            }
        }
    }
}

impl<const N: usize> std::ops::SubAssign for SRobotQ<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        if N <= 16 {
            let mut out = [0.0; N];
            simd_binop(&self.0, &rhs.0, &mut out, |a, b| a - b);
            self.0 = out;
        } else {
            for i in 0..N {
                self.0[i] -= rhs.0[i];
            }
        }
    }
}

impl<const N: usize> std::ops::MulAssign<f32> for SRobotQ<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        if N <= 16 {
            let mut out = [0.0; N];
            simd_scalarop(&self.0, f32x8::splat(rhs), &mut out, |a, s| a * s);
            self.0 = out;
        } else {
            for i in 0..N {
                self.0[i] *= rhs;
            }
        }
    }
}

impl<const N: usize> std::ops::DivAssign<f32> for SRobotQ<N> {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        if N <= 16 {
            let mut out = [0.0; N];
            simd_scalarop(&self.0, f32x8::splat(rhs), &mut out, |a, s| a / s);
            self.0 = out;
        } else {
            for i in 0..N {
                self.0[i] /= rhs;
            }
        }
    }
}

impl<const N: usize> std::ops::Add<SRobotQ<N>> for &RobotQ {
    type Output = SRobotQ<N>;
    #[inline]
    fn add(self, rhs: SRobotQ<N>) -> SRobotQ<N> {
        SRobotQ::<N>::force_from_robotq(self) + rhs
    }
}

impl<const N: usize> std::ops::Sub<SRobotQ<N>> for &RobotQ {
    type Output = SRobotQ<N>;
    #[inline]
    fn sub(self, rhs: SRobotQ<N>) -> SRobotQ<N> {
        SRobotQ::<N>::force_from_robotq(self) - rhs
    }
}

impl<const N: usize> Default for SRobotQ<N> {
    #[inline]
    fn default() -> Self {
        Self::zeros()
    }
}

impl<const N: usize> AsRef<[f32; N]> for SRobotQ<N> {
    #[inline]
    fn as_ref(&self) -> &[f32; N] {
        &self.0
    }
}

impl<const N: usize> AsMut<[f32; N]> for SRobotQ<N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32; N] {
        &mut self.0
    }
}

impl<const N: usize> AsRef<[f32]> for SRobotQ<N> {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        &self.0
    }
}

impl<const N: usize> AsMut<[f32]> for SRobotQ<N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32] {
        &mut self.0
    }
}

impl<const N: usize> From<[f32; N]> for SRobotQ<N> {
    #[inline]
    fn from(arr: [f32; N]) -> Self {
        Self(arr)
    }
}

impl<const N: usize> From<&[f32; N]> for SRobotQ<N> {
    #[inline]
    fn from(arr: &[f32; N]) -> Self {
        Self(*arr)
    }
}

impl<const N: usize> From<[f64; N]> for SRobotQ<N> {
    #[inline]
    fn from(arr: [f64; N]) -> Self {
        let mut out = [0.0f32; N];
        let mut i = 0;
        while i < N {
            out[i] = arr[i] as f32;
            i += 1;
        }
        Self(out)
    }
}

impl<const N: usize> From<&[f64; N]> for SRobotQ<N> {
    #[inline]
    fn from(arr: &[f64; N]) -> Self {
        Self::from(*arr)
    }
}

impl<const N: usize> From<SRobotQ<N>> for [f32; N] {
    #[inline]
    fn from(q: SRobotQ<N>) -> [f32; N] {
        q.0
    }
}

impl<const N: usize> From<SRobotQ<N>> for Vec<f32> {
    #[inline]
    fn from(q: SRobotQ<N>) -> Vec<f32> {
        q.0.to_vec()
    }
}

impl<const N: usize> From<SRobotQ<N>> for RobotQ {
    #[inline]
    fn from(q: SRobotQ<N>) -> RobotQ {
        q.to_robotq()
    }
}

impl<const N: usize> TryFrom<&SRobotQ<N>> for SRobotQ<N> {
    type Error = DekeError;

    #[inline]
    fn try_from(q: &SRobotQ<N>) -> Result<Self, Self::Error> {
        Ok(*q)
    }
}

impl<const N: usize> TryFrom<&[f32]> for SRobotQ<N> {
    type Error = DekeError;

    #[inline]
    fn try_from(slice: &[f32]) -> Result<Self, Self::Error> {
        if slice.len() != N {
            return Err(DekeError::ShapeMismatch {
                expected: N,
                found: slice.len(),
            });
        }
        let mut arr = [0.0; N];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}

impl<const N: usize> TryFrom<Vec<f32>> for SRobotQ<N> {
    type Error = DekeError;

    #[inline]
    fn try_from(v: Vec<f32>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl<const N: usize> TryFrom<&Vec<f32>> for SRobotQ<N> {
    type Error = DekeError;

    #[inline]
    fn try_from(v: &Vec<f32>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl<const N: usize> TryFrom<&[f64]> for SRobotQ<N> {
    type Error = DekeError;

    #[inline]
    fn try_from(slice: &[f64]) -> Result<Self, Self::Error> {
        if slice.len() != N {
            return Err(DekeError::ShapeMismatch {
                expected: N,
                found: slice.len(),
            });
        }
        let mut arr = [0.0f32; N];
        let mut i = 0;
        while i < N {
            arr[i] = slice[i] as f32;
            i += 1;
        }
        Ok(Self(arr))
    }
}

impl<const N: usize> TryFrom<Vec<f64>> for SRobotQ<N> {
    type Error = DekeError;

    #[inline]
    fn try_from(v: Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl<const N: usize> TryFrom<&Vec<f64>> for SRobotQ<N> {
    type Error = DekeError;

    #[inline]
    fn try_from(v: &Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl<const N: usize> TryFrom<&RobotQ> for SRobotQ<N> {
    type Error = DekeError;

    #[inline]
    fn try_from(q: &RobotQ) -> Result<Self, Self::Error> {
        let slice = q.as_slice().unwrap_or(&[]);
        if slice.len() != N {
            return Err(DekeError::ShapeMismatch {
                expected: N,
                found: slice.len(),
            });
        }
        let mut arr = [0.0; N];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}

impl<const N: usize> TryFrom<RobotQ> for SRobotQ<N> {
    type Error = DekeError;

    #[inline]
    fn try_from(q: RobotQ) -> Result<Self, Self::Error> {
        let slice = q.as_slice().unwrap_or(&[]);
        if slice.len() != N {
            return Err(DekeError::ShapeMismatch {
                expected: N,
                found: slice.len(),
            });
        }
        let mut arr = [0.0; N];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}
