use std::convert::Infallible;

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use ndarray::Array1;
use num_traits::Float;

use crate::DekeError;

pub type RobotQ<T = f32> = Array1<T>;

/// Statically-sized joint configuration backed by `[T; N]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SRobotQ<const N: usize, T: Float = f32>(pub [T; N]);

impl<const N: usize, T: Float> SRobotQ<N, T> {
    pub fn zeros() -> Self {
        Self([T::zero(); N])
    }

    pub fn from_array(arr: [T; N]) -> Self {
        Self(arr)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    pub fn to_robotq(&self) -> RobotQ<T> {
        RobotQ::from(self.0.to_vec())
    }

    pub fn force_from_robotq(q: &RobotQ<T>) -> Self {
        if let Ok(sq) = Self::try_from(q) {
            sq
        } else {
            let slice = q.as_slice().unwrap_or(&[]);
            let mut arr = [T::zero(); N];
            for i in 0..N {
                arr[i] = slice.get(i).copied().unwrap_or_else(T::zero);
            }
            Self(arr)
        }
    }

    pub fn norm(&self) -> T {
        self.dot(self).sqrt()
    }

    pub fn dot(&self, other: &Self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = self.0[i].mul_add(other.0[i], sum);
        }
        sum
    }

    pub fn map(&self, f: impl Fn(T) -> T) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = f(self.0[i]);
        }
        Self(out)
    }

    pub fn sum(&self) -> T {
        let mut s = T::zero();
        for i in 0..N {
            s = s + self.0[i];
        }
        s
    }

    pub fn splat(val: T) -> Self {
        Self([val; N])
    }

    pub fn from_fn(f: impl Fn(usize) -> T) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = f(i);
        }
        Self(out)
    }

    pub fn norm_squared(&self) -> T {
        self.dot(self)
    }

    pub fn normalize(&self) -> Self {
        let n = self.norm();
        debug_assert!(n > T::zero(), "cannot normalize zero-length SRobotQ");
        *self / n
    }

    pub fn distance(&self, other: &Self) -> T {
        (*self - *other).norm()
    }

    pub fn distance_squared(&self, other: &Self) -> T {
        (*self - *other).norm_squared()
    }

    pub fn abs(&self) -> Self {
        self.map(Float::abs)
    }

    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            if self.0[i] < min.0[i] {
                out[i] = min.0[i];
            } else if self.0[i] > max.0[i] {
                out[i] = max.0[i];
            } else {
                out[i] = self.0[i];
            }
        }
        Self(out)
    }

    pub fn clamp_scalar(&self, min: T, max: T) -> Self {
        self.map(|x| {
            if x < min {
                min
            } else if x > max {
                max
            } else {
                x
            }
        })
    }

    pub fn max_element(&self) -> T {
        self.0
            .iter()
            .copied()
            .fold(T::neg_infinity(), |a, b| if b > a { b } else { a })
    }

    pub fn min_element(&self) -> T {
        self.0
            .iter()
            .copied()
            .fold(T::infinity(), |a, b| if b < a { b } else { a })
    }

    pub fn linf_norm(&self) -> T {
        self.abs().max_element()
    }

    pub fn elementwise_mul(&self, other: &Self) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = self.0[i] * other.0[i];
        }
        Self(out)
    }

    pub fn elementwise_div(&self, other: &Self) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = self.0[i] / other.0[i];
        }
        Self(out)
    }

    pub fn zip_map(&self, other: &Self, f: impl Fn(T, T) -> T) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = f(self.0[i], other.0[i]);
        }
        Self(out)
    }

    pub fn sqrt(&self) -> Self {
        self.map(Float::sqrt)
    }

    pub fn mul_add(&self, mul: &Self, add: &Self) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = self.0[i].mul_add(mul.0[i], add.0[i]);
        }
        Self(out)
    }

    pub fn any_non_finite(&self) -> bool {
        self.0.iter().any(|x| x.is_nan() || x.is_infinite())
    }

    pub fn any_gt(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).any(|(a, b)| *a > *b)
    }

    pub fn any_lt(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).any(|(a, b)| *a < *b)
    }

    pub fn is_close(&self, other: &Self, tol: T) -> bool {
        (*self - *other).norm() < tol
    }

    pub fn interpolate(&self, other: &Self, t: T) -> Self {
        *self + ((*other - *self) * t)
    }
}

impl<const N: usize> SRobotQ<N, f32> {
    pub const fn from_array_d(arr: [f64; N]) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = arr[i] as f32;
            i += 1;
        }
        Self(out)
    }
}

impl<const N: usize, T: Float> std::ops::Index<usize> for SRobotQ<N, T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}

impl<const N: usize, T: Float> std::ops::IndexMut<usize> for SRobotQ<N, T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

impl<const N: usize, T: Float> std::ops::Add for SRobotQ<N, T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = self.0[i] + rhs.0[i];
        }
        Self(out)
    }
}

impl<const N: usize, T: Float> std::ops::Sub for SRobotQ<N, T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = self.0[i] - rhs.0[i];
        }
        Self(out)
    }
}

impl<const N: usize, T: Float> std::ops::Neg for SRobotQ<N, T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = T::zero() - self.0[i];
        }
        Self(out)
    }
}

impl<const N: usize, T: Float> std::ops::Mul<T> for SRobotQ<N, T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = self.0[i] * rhs;
        }
        Self(out)
    }
}

impl<const N: usize> std::ops::Mul<SRobotQ<N, f32>> for f32 {
    type Output = SRobotQ<N, f32>;
    #[inline]
    fn mul(self, rhs: SRobotQ<N, f32>) -> SRobotQ<N, f32> {
        rhs * self
    }
}

impl<const N: usize> std::ops::Mul<SRobotQ<N, f64>> for f64 {
    type Output = SRobotQ<N, f64>;
    #[inline]
    fn mul(self, rhs: SRobotQ<N, f64>) -> SRobotQ<N, f64> {
        rhs * self
    }
}

impl<const N: usize, T: Float> std::ops::Div<T> for SRobotQ<N, T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = self.0[i] / rhs;
        }
        Self(out)
    }
}

impl<const N: usize, T: Float> std::ops::AddAssign for SRobotQ<N, T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] = self.0[i] + rhs.0[i];
        }
    }
}

impl<const N: usize, T: Float> std::ops::SubAssign for SRobotQ<N, T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] = self.0[i] - rhs.0[i];
        }
    }
}

impl<const N: usize, T: Float> std::ops::MulAssign<T> for SRobotQ<N, T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..N {
            self.0[i] = self.0[i] * rhs;
        }
    }
}

impl<const N: usize, T: Float> std::ops::DivAssign<T> for SRobotQ<N, T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        for i in 0..N {
            self.0[i] = self.0[i] / rhs;
        }
    }
}

macro_rules! impl_cross_float_ops {
    ($lhs:ty, $rhs:ty) => {
        impl<const N: usize> std::ops::Add<SRobotQ<N, $rhs>> for SRobotQ<N, $lhs> {
            type Output = SRobotQ<N, $lhs>;
            #[inline]
            fn add(self, rhs: SRobotQ<N, $rhs>) -> SRobotQ<N, $lhs> {
                let mut out = [0.0 as $lhs; N];
                for i in 0..N {
                    out[i] = self.0[i] + rhs.0[i] as $lhs;
                }
                SRobotQ(out)
            }
        }

        impl<const N: usize> std::ops::Sub<SRobotQ<N, $rhs>> for SRobotQ<N, $lhs> {
            type Output = SRobotQ<N, $lhs>;
            #[inline]
            fn sub(self, rhs: SRobotQ<N, $rhs>) -> SRobotQ<N, $lhs> {
                let mut out = [0.0 as $lhs; N];
                for i in 0..N {
                    out[i] = self.0[i] - rhs.0[i] as $lhs;
                }
                SRobotQ(out)
            }
        }

        impl<const N: usize> std::ops::AddAssign<SRobotQ<N, $rhs>> for SRobotQ<N, $lhs> {
            #[inline]
            fn add_assign(&mut self, rhs: SRobotQ<N, $rhs>) {
                for i in 0..N {
                    self.0[i] = self.0[i] + rhs.0[i] as $lhs;
                }
            }
        }

        impl<const N: usize> std::ops::SubAssign<SRobotQ<N, $rhs>> for SRobotQ<N, $lhs> {
            #[inline]
            fn sub_assign(&mut self, rhs: SRobotQ<N, $rhs>) {
                for i in 0..N {
                    self.0[i] = self.0[i] - rhs.0[i] as $lhs;
                }
            }
        }

        impl<const N: usize> std::ops::Mul<$rhs> for SRobotQ<N, $lhs> {
            type Output = SRobotQ<N, $lhs>;
            #[inline]
            fn mul(self, rhs: $rhs) -> SRobotQ<N, $lhs> {
                let rhs = rhs as $lhs;
                let mut out = [0.0 as $lhs; N];
                for i in 0..N {
                    out[i] = self.0[i] * rhs;
                }
                SRobotQ(out)
            }
        }

        impl<const N: usize> std::ops::Div<$rhs> for SRobotQ<N, $lhs> {
            type Output = SRobotQ<N, $lhs>;
            #[inline]
            fn div(self, rhs: $rhs) -> SRobotQ<N, $lhs> {
                let rhs = rhs as $lhs;
                let mut out = [0.0 as $lhs; N];
                for i in 0..N {
                    out[i] = self.0[i] / rhs;
                }
                SRobotQ(out)
            }
        }

        impl<const N: usize> std::ops::MulAssign<$rhs> for SRobotQ<N, $lhs> {
            #[inline]
            fn mul_assign(&mut self, rhs: $rhs) {
                let rhs = rhs as $lhs;
                for i in 0..N {
                    self.0[i] = self.0[i] * rhs;
                }
            }
        }

        impl<const N: usize> std::ops::DivAssign<$rhs> for SRobotQ<N, $lhs> {
            #[inline]
            fn div_assign(&mut self, rhs: $rhs) {
                let rhs = rhs as $lhs;
                for i in 0..N {
                    self.0[i] = self.0[i] / rhs;
                }
            }
        }

        impl<const N: usize> std::ops::Mul<SRobotQ<N, $lhs>> for $rhs {
            type Output = SRobotQ<N, $lhs>;
            #[inline]
            fn mul(self, rhs: SRobotQ<N, $lhs>) -> SRobotQ<N, $lhs> {
                rhs * (self as $lhs)
            }
        }
    };
}

impl_cross_float_ops!(f32, f64);
impl_cross_float_ops!(f64, f32);

impl<const N: usize> std::ops::Add<SRobotQ<N, f32>> for &RobotQ {
    type Output = SRobotQ<N, f32>;
    #[inline]
    fn add(self, rhs: SRobotQ<N, f32>) -> SRobotQ<N, f32> {
        SRobotQ::<N, f32>::force_from_robotq(self) + rhs
    }
}

impl<const N: usize> std::ops::Sub<SRobotQ<N, f32>> for &RobotQ {
    type Output = SRobotQ<N, f32>;
    #[inline]
    fn sub(self, rhs: SRobotQ<N, f32>) -> SRobotQ<N, f32> {
        SRobotQ::<N, f32>::force_from_robotq(self) - rhs
    }
}

impl<const N: usize, T: Float> Default for SRobotQ<N, T> {
    #[inline]
    fn default() -> Self {
        Self::zeros()
    }
}

impl<const N: usize, T: Float> AsRef<[T; N]> for SRobotQ<N, T> {
    #[inline]
    fn as_ref(&self) -> &[T; N] {
        &self.0
    }
}

impl<const N: usize, T: Float> AsMut<[T; N]> for SRobotQ<N, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

impl<const N: usize, T: Float> AsRef<[T]> for SRobotQ<N, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<const N: usize, T: Float> AsMut<[T]> for SRobotQ<N, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<const N: usize, T: Float> From<[T; N]> for SRobotQ<N, T> {
    #[inline]
    fn from(arr: [T; N]) -> Self {
        Self(arr)
    }
}

impl<const N: usize, T: Float> From<&[T; N]> for SRobotQ<N, T> {
    #[inline]
    fn from(arr: &[T; N]) -> Self {
        Self(*arr)
    }
}

impl<const N: usize> From<[f64; N]> for SRobotQ<N, f32> {
    #[inline]
    fn from(arr: [f64; N]) -> Self {
        let mut out = [0.0f32; N];
        for i in 0..N {
            out[i] = arr[i] as f32;
        }
        Self(out)
    }
}

impl<const N: usize> From<&[f64; N]> for SRobotQ<N, f32> {
    #[inline]
    fn from(arr: &[f64; N]) -> Self {
        Self::from(*arr)
    }
}

impl<const N: usize> From<[f32; N]> for SRobotQ<N, f64> {
    #[inline]
    fn from(arr: [f32; N]) -> Self {
        let mut out = [0.0f64; N];
        for i in 0..N {
            out[i] = arr[i] as f64;
        }
        Self(out)
    }
}

impl<const N: usize> From<&[f32; N]> for SRobotQ<N, f64> {
    #[inline]
    fn from(arr: &[f32; N]) -> Self {
        Self::from(*arr)
    }
}

impl<const N: usize, T: Float> From<SRobotQ<N, T>> for [T; N] {
    #[inline]
    fn from(q: SRobotQ<N, T>) -> [T; N] {
        q.0
    }
}

impl<const N: usize> From<SRobotQ<N, f32>> for [f64; N] {
    #[inline]
    fn from(q: SRobotQ<N, f32>) -> [f64; N] {
        let mut out = [0.0f64; N];
        for i in 0..N {
            out[i] = q.0[i] as f64;
        }
        out
    }
}

impl<const N: usize> From<SRobotQ<N, f64>> for [f32; N] {
    #[inline]
    fn from(q: SRobotQ<N, f64>) -> [f32; N] {
        let mut out = [0.0f32; N];
        for i in 0..N {
            out[i] = q.0[i] as f32;
        }
        out
    }
}

impl<const N: usize, T: Float> From<SRobotQ<N, T>> for Vec<T> {
    #[inline]
    fn from(q: SRobotQ<N, T>) -> Vec<T> {
        q.0.to_vec()
    }
}

impl<const N: usize> From<SRobotQ<N, f32>> for Vec<f64> {
    #[inline]
    fn from(q: SRobotQ<N, f32>) -> Vec<f64> {
        q.0.iter().map(|&v| v as f64).collect()
    }
}

impl<const N: usize> From<SRobotQ<N, f64>> for Vec<f32> {
    #[inline]
    fn from(q: SRobotQ<N, f64>) -> Vec<f32> {
        q.0.iter().map(|&v| v as f32).collect()
    }
}

impl<const N: usize, T: Float> From<SRobotQ<N, T>> for RobotQ<T> {
    #[inline]
    fn from(q: SRobotQ<N, T>) -> RobotQ<T> {
        q.to_robotq()
    }
}

impl<const N: usize, T: Float> From<&SRobotQ<N, T>> for SRobotQ<N, T> {
    #[inline]
    fn from(q: &SRobotQ<N, T>) -> Self {
        *q
    }
}

impl<const N: usize, T: Float> TryFrom<&[T]> for SRobotQ<N, T> {
    type Error = DekeError;

    #[inline]
    fn try_from(slice: &[T]) -> Result<Self, Self::Error> {
        if slice.len() != N {
            return Err(DekeError::ShapeMismatch {
                expected: N,
                found: slice.len(),
            });
        }
        let mut arr = [T::zero(); N];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}

impl<const N: usize, T: Float> TryFrom<Vec<T>> for SRobotQ<N, T> {
    type Error = DekeError;

    #[inline]
    fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl<const N: usize, T: Float> TryFrom<&Vec<T>> for SRobotQ<N, T> {
    type Error = DekeError;

    #[inline]
    fn try_from(v: &Vec<T>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl<const N: usize> TryFrom<&[f64]> for SRobotQ<N, f32> {
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
        for i in 0..N {
            arr[i] = slice[i] as f32;
        }
        Ok(Self(arr))
    }
}

impl<const N: usize> TryFrom<Vec<f64>> for SRobotQ<N, f32> {
    type Error = DekeError;

    #[inline]
    fn try_from(v: Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl<const N: usize> TryFrom<&Vec<f64>> for SRobotQ<N, f32> {
    type Error = DekeError;

    #[inline]
    fn try_from(v: &Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl<const N: usize> TryFrom<&[f32]> for SRobotQ<N, f64> {
    type Error = DekeError;

    #[inline]
    fn try_from(slice: &[f32]) -> Result<Self, Self::Error> {
        if slice.len() != N {
            return Err(DekeError::ShapeMismatch {
                expected: N,
                found: slice.len(),
            });
        }
        let mut arr = [0.0f64; N];
        for i in 0..N {
            arr[i] = slice[i] as f64;
        }
        Ok(Self(arr))
    }
}

impl<const N: usize> TryFrom<Vec<f32>> for SRobotQ<N, f64> {
    type Error = DekeError;

    #[inline]
    fn try_from(v: Vec<f32>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl<const N: usize> TryFrom<&Vec<f32>> for SRobotQ<N, f64> {
    type Error = DekeError;

    #[inline]
    fn try_from(v: &Vec<f32>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl<const N: usize, T: Float> TryFrom<&RobotQ<T>> for SRobotQ<N, T> {
    type Error = DekeError;

    #[inline]
    fn try_from(q: &RobotQ<T>) -> Result<Self, Self::Error> {
        let slice = q.as_slice().unwrap_or(&[]);
        if slice.len() != N {
            return Err(DekeError::ShapeMismatch {
                expected: N,
                found: slice.len(),
            });
        }
        let mut arr = [T::zero(); N];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}

impl<const N: usize, T: Float + AbsDiffEq<Epsilon = T>> AbsDiffEq for SRobotQ<N, T> {
    type Epsilon = T;

    fn default_epsilon() -> T {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl<const N: usize, T: Float + RelativeEq + AbsDiffEq<Epsilon = T>> RelativeEq
    for SRobotQ<N, T>
{
    fn default_max_relative() -> T {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T, max_relative: T) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| a.relative_eq(b, epsilon, max_relative))
    }
}

impl<const N: usize, T: Float + UlpsEq + AbsDiffEq<Epsilon = T>> UlpsEq for SRobotQ<N, T> {
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: T, max_ulps: u32) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| a.ulps_eq(b, epsilon, max_ulps))
    }
}

impl<const N: usize, T: Float> TryFrom<RobotQ<T>> for SRobotQ<N, T> {
    type Error = DekeError;

    #[inline]
    fn try_from(q: RobotQ<T>) -> Result<Self, Self::Error> {
        let slice = q.as_slice().unwrap_or(&[]);
        if slice.len() != N {
            return Err(DekeError::ShapeMismatch {
                expected: N,
                found: slice.len(),
            });
        }
        let mut arr = [T::zero(); N];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}

impl<const N: usize, T: Float> From<&SRobotQ<N, T>> for RobotQ<T> {
    #[inline]
    fn from(sq: &SRobotQ<N, T>) -> RobotQ<T> {
        sq.to_robotq()
    }
}

impl<const N: usize> From<SRobotQ<N, f32>> for RobotQ<f64> {
    #[inline]
    fn from(q: SRobotQ<N, f32>) -> RobotQ<f64> {
        q.0.iter().map(|&v| v as f64).collect()
    }
}

impl<const N: usize> From<SRobotQ<N, f64>> for RobotQ<f32> {
    #[inline]
    fn from(q: SRobotQ<N, f64>) -> RobotQ<f32> {
        q.0.iter().map(|&v| v as f32).collect()
    }
}

impl<const N: usize> From<&SRobotQ<N, f32>> for RobotQ<f64> {
    #[inline]
    fn from(q: &SRobotQ<N, f32>) -> RobotQ<f64> {
        q.0.iter().map(|&v| v as f64).collect()
    }
}

impl<const N: usize> From<&SRobotQ<N, f64>> for RobotQ<f32> {
    #[inline]
    fn from(q: &SRobotQ<N, f64>) -> RobotQ<f32> {
        q.0.iter().map(|&v| v as f32).collect()
    }
}

impl<const N: usize> TryFrom<&RobotQ<f64>> for SRobotQ<N, f32> {
    type Error = DekeError;

    #[inline]
    fn try_from(q: &RobotQ<f64>) -> Result<Self, Self::Error> {
        let slice = q.as_slice().unwrap_or(&[]);
        if slice.len() != N {
            return Err(DekeError::ShapeMismatch {
                expected: N,
                found: slice.len(),
            });
        }
        let mut arr = [0.0f32; N];
        for i in 0..N {
            arr[i] = slice[i] as f32;
        }
        Ok(Self(arr))
    }
}

impl<const N: usize> TryFrom<RobotQ<f64>> for SRobotQ<N, f32> {
    type Error = DekeError;

    #[inline]
    fn try_from(q: RobotQ<f64>) -> Result<Self, Self::Error> {
        Self::try_from(&q)
    }
}

impl<const N: usize> TryFrom<&RobotQ<f32>> for SRobotQ<N, f64> {
    type Error = DekeError;

    #[inline]
    fn try_from(q: &RobotQ<f32>) -> Result<Self, Self::Error> {
        let slice = q.as_slice().unwrap_or(&[]);
        if slice.len() != N {
            return Err(DekeError::ShapeMismatch {
                expected: N,
                found: slice.len(),
            });
        }
        let mut arr = [0.0f64; N];
        for i in 0..N {
            arr[i] = slice[i] as f64;
        }
        Ok(Self(arr))
    }
}

impl<const N: usize> TryFrom<RobotQ<f32>> for SRobotQ<N, f64> {
    type Error = DekeError;

    #[inline]
    fn try_from(q: RobotQ<f32>) -> Result<Self, Self::Error> {
        Self::try_from(&q)
    }
}

impl<const N: usize> From<SRobotQ<N, f32>> for SRobotQ<N, f64> {
    #[inline]
    fn from(q: SRobotQ<N, f32>) -> Self {
        let mut out = [0.0f64; N];
        for i in 0..N {
            out[i] = q.0[i] as f64;
        }
        Self(out)
    }
}

impl<const N: usize> From<SRobotQ<N, f64>> for SRobotQ<N, f32> {
    #[inline]
    fn from(q: SRobotQ<N, f64>) -> Self {
        let mut out = [0.0f32; N];
        for i in 0..N {
            out[i] = q.0[i] as f32;
        }
        Self(out)
    }
}

pub fn robotq<T: Float, U: Float>(vals: impl IntoIterator<Item = T>) -> RobotQ<U> {
    use num_traits::NumCast;
    vals.into_iter().map(|v| NumCast::from(v).unwrap_or(U::zero())).collect()
}


pub trait SRobotQLike<const N: usize, E: Into<DekeError>>: TryInto<SRobotQ<N, f32>, Error = E> + TryInto<SRobotQ<N, f64>, Error = E> {
    fn to_srobotq(self) -> Result<SRobotQ<N, f32>, E>;
    fn to_srobotq_f64(self) -> Result<SRobotQ<N, f64>, E>;
}

impl<const N: usize, T, E: Into<DekeError>> SRobotQLike<N, E> for T
where
    T: TryInto<SRobotQ<N, f32>, Error = E> + TryInto<SRobotQ<N, f64>, Error = E>,
{
    #[inline]
    fn to_srobotq(self) -> Result<SRobotQ<N, f32>, E> {
        self.try_into()
    }

    #[inline]
    fn to_srobotq_f64(self) -> Result<SRobotQ<N, f64>, E> {
        self.try_into()
    }
}

#[allow(dead_code)]
const _: () = {
    fn assert_srobotq_like<const N: usize, E: Into<DekeError>, T: SRobotQLike<N, E>>() {}

    fn assert_all<const N: usize>() {
        assert_srobotq_like::<N, Infallible, SRobotQ<N, f32>>();
        assert_srobotq_like::<N, Infallible, SRobotQ<N, f64>>();
        assert_srobotq_like::<N, DekeError, RobotQ<f32>>();
        assert_srobotq_like::<N, DekeError, RobotQ<f64>>();
        assert_srobotq_like::<N, DekeError, Vec<f32>>();
        assert_srobotq_like::<N, DekeError, Vec<f64>>();
        assert_srobotq_like::<N, DekeError, &[f32]>();
        assert_srobotq_like::<N, DekeError, &[f64]>();
    }
};