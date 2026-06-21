use tinyrand::{Rand, Seeded, SplitMix, Wyrand, Xorshift};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum RandomizerType {
    #[default]
    Wyrand,
    SplitMix,
    Xorshift,
    Halton,
}


const F64_FROM_U64: f64 = 1.0 / (1u64 << 53) as f64;

#[inline]
fn u64_to_unit_f64(bits: u64) -> f64 {
    (bits >> 11) as f64 * F64_FROM_U64
}

#[inline]
fn unit_f64_to_u64(v: f64) -> u64 {
    let bits = (v * (1u64 << 53) as f64) as u64;
    bits << 11
}

/// Random source parameterised by joint dimension `N`.
///
/// `next_u64` / `next_f64` are the scalar interface — useful for one-off draws
/// (e.g. picking a cost bound or feeding box-muller). `sample_unit` is the N-D
/// interface that returns one full unit-cube vector. The split exists so a
/// quasi-random source like [`HaltonRand`] can advance every dimension at once
/// with SIMD instead of pretending to be a stream of independent u64s; the `N`
/// const generic lets the impl bake the dimension count into its layout.
pub trait DekeRng<const N: usize> {
    fn next_u64(&mut self) -> u64;

    #[inline]
    fn next_f64(&mut self) -> f64 {
        u64_to_unit_f64(self.next_u64())
    }

    /// Draw one `N`-dim sample on the unit cube. Default is `N` scalar calls;
    /// stripe-aware sources override this to advance all dimensions in lockstep.
    #[inline]
    fn sample_unit(&mut self) -> [f64; N] {
        std::array::from_fn(|_| self.next_f64())
    }
}

impl<const N: usize> DekeRng<N> for Wyrand {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        <Self as Rand>::next_u64(self)
    }
}

impl<const N: usize> DekeRng<N> for SplitMix {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        <Self as Rand>::next_u64(self)
    }
}

impl<const N: usize> DekeRng<N> for Xorshift {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        <Self as Rand>::next_u64(self)
    }
}

pub enum DekeRand<const N: usize> {
    Wyrand(Wyrand),
    SplitMix(SplitMix),
    Xorshift(Xorshift),
    Halton(HaltonRand<N>),
}

impl<const N: usize> DekeRand<N> {
    /// `N` is the joint-dimension. Only the Halton variant uses it; the PRNG
    /// variants accept any `N`. Halton requires `0 < N <= HALTON_BASES.len()`.
    pub fn new(kind: RandomizerType, seed: u64) -> Self {
        match kind {
            RandomizerType::Wyrand => Self::Wyrand(Wyrand::seed(seed)),
            RandomizerType::SplitMix => Self::SplitMix(SplitMix::seed(seed)),
            RandomizerType::Xorshift => Self::Xorshift(Xorshift::seed(seed)),
            RandomizerType::Halton => Self::Halton(HaltonRand::new(seed)),
        }
    }
}

impl<const N: usize> DekeRng<N> for DekeRand<N> {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        match self {
            Self::Wyrand(r) => <Wyrand as DekeRng<N>>::next_u64(r),
            Self::SplitMix(r) => <SplitMix as DekeRng<N>>::next_u64(r),
            Self::Xorshift(r) => <Xorshift as DekeRng<N>>::next_u64(r),
            Self::Halton(r) => DekeRng::<N>::next_u64(r),
        }
    }

    #[inline]
    fn sample_unit(&mut self) -> [f64; N] {
        match self {
            Self::Halton(r) => DekeRng::<N>::sample_unit(r),
            Self::Wyrand(r) => {
                std::array::from_fn(|_| u64_to_unit_f64(<Wyrand as Rand>::next_u64(r)))
            }
            Self::SplitMix(r) => {
                std::array::from_fn(|_| u64_to_unit_f64(<SplitMix as Rand>::next_u64(r)))
            }
            Self::Xorshift(r) => {
                std::array::from_fn(|_| u64_to_unit_f64(<Xorshift as Rand>::next_u64(r)))
            }
        }
    }
}

/// Bases used by the Halton sampler.
const HALTON_BASES: [u32; 16] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];

const HALTON_MAX_ITERATIONS: usize = 1_000_000;

/// Anti-stuck epsilon from Kollig–Keller — keeps the recurrence from getting
/// trapped at a digit boundary due to f64 rounding.
const HALTON_EPS: f64 = 1e-10;

/// Quasi-random Halton source over `N` dimensions.
///
/// Uses the Kollig–Keller incremental recurrence — each call performs an O(1)
/// digit increment on the radical inverse instead of recomputing it from
/// scratch like a classical implementation, which is what makes this much
/// faster than the previous `halton::Sequence`-backed version.
///
/// `next_u64` advances one dimension via a round-robin cursor (kept for the
/// scalar interface). `sample_unit` advances every dimension at once using
/// SIMD on x86_64 (SSE2 baseline) — that's the path joint-space samplers
/// should take.
pub struct HaltonRand<const N: usize> {
    inv_bases: [f64; N],
    values: [f64; N],
    cursor: usize,
    outer_iters: usize,
    seed: u64,
    base_offset: usize,
}

impl<const N: usize> HaltonRand<N> {
    pub fn new(seed: u64) -> Self {
        const {
            assert!(N > 0, "HaltonRand requires N > 0");
            assert!(
                N <= HALTON_BASES.len(),
                "HaltonRand supports up to 16 dimensions"
            );
        }
        let mut s = Self {
            inv_bases: [0.0; N],
            values: [0.0; N],
            cursor: 0,
            outer_iters: 0,
            seed,
            base_offset: 0,
        };
        s.reset_bases();
        s.apply_seed_skip();
        s
    }

    fn reset_bases(&mut self) {
        for i in 0..N {
            let base_idx = (i + self.base_offset) % HALTON_BASES.len();
            self.inv_bases[i] = 1.0 / HALTON_BASES[base_idx] as f64;
        }
        for v in &mut self.values {
            *v = 0.0;
        }
    }

    fn apply_seed_skip(&mut self) {
        // Seed picks a starting offset within the deterministic sequence; cap
        // the skip so we don't burn the precision budget before the first
        // sample.
        let skip = (self.seed % 4096) as usize;
        for _ in 0..skip {
            for d in 0..N {
                self.advance_scalar(d);
            }
        }
    }

    fn rotate_bases(&mut self) {
        // VAMP rotates the dimension→base mapping after the precision budget
        // is exhausted; we mirror that by shifting `base_offset` and
        // resetting the recurrence state.
        self.base_offset = (self.base_offset + 1) % HALTON_BASES.len();
        self.outer_iters = 0;
        self.reset_bases();
        self.apply_seed_skip();
    }

    #[inline(always)]
    fn advance_scalar(&mut self, dim: usize) -> f64 {
        let inv_b = self.inv_bases[dim];
        let v = self.values[dim];
        let r = 1.0 - v - HALTON_EPS;
        let new_v = if r > inv_b {
            v + inv_b
        } else {
            // Slow path: walk the digit boundary. 30 iters covers base 3 down
            // to ~5e-15, well past f64 precision.
            let mut h = inv_b;
            let mut hh = h;
            for _ in 0..30 {
                hh = h;
                h *= inv_b;
                if r > h {
                    break;
                }
            }
            v + hh + h - 1.0
        };
        self.values[dim] = new_v;
        new_v
    }

    /// Advance every dimension by one step. Pairs of lanes go through SSE2 on
    /// x86_64 (always available — it's part of the x86_64 baseline ABI); any
    /// pair where either lane needs the slow path falls back to scalar.
    fn advance_all(&mut self) -> [f64; N] {
        let mut out = [0.0f64; N];
        let mut idx = 0;

        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let one_minus_eps = _mm_set1_pd(1.0 - HALTON_EPS);
            while idx + 2 <= N {
                let v = _mm_loadu_pd(self.values.as_ptr().add(idx));
                let inv = _mm_loadu_pd(self.inv_bases.as_ptr().add(idx));
                let r = _mm_sub_pd(one_minus_eps, v);
                let mask = _mm_cmpgt_pd(r, inv);
                if _mm_movemask_pd(mask) == 0b11 {
                    let new_v = _mm_add_pd(v, inv);
                    _mm_storeu_pd(self.values.as_mut_ptr().add(idx), new_v);
                    _mm_storeu_pd(out.as_mut_ptr().add(idx), new_v);
                } else {
                    out[idx] = self.advance_scalar(idx);
                    out[idx + 1] = self.advance_scalar(idx + 1);
                }
                idx += 2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let one_minus_eps = vdupq_n_f64(1.0 - HALTON_EPS);
            while idx + 2 <= N {
                let v = vld1q_f64(self.values.as_ptr().add(idx));
                let inv = vld1q_f64(self.inv_bases.as_ptr().add(idx));
                let r = vsubq_f64(one_minus_eps, v);
                let mask = vcgtq_f64(r, inv);
                let lo = vgetq_lane_u64(mask, 0);
                let hi = vgetq_lane_u64(mask, 1);
                if lo == u64::MAX && hi == u64::MAX {
                    let new_v = vaddq_f64(v, inv);
                    vst1q_f64(self.values.as_mut_ptr().add(idx), new_v);
                    vst1q_f64(out.as_mut_ptr().add(idx), new_v);
                } else {
                    out[idx] = self.advance_scalar(idx);
                    out[idx + 1] = self.advance_scalar(idx + 1);
                }
                idx += 2;
            }
        }

        while idx < N {
            out[idx] = self.advance_scalar(idx);
            idx += 1;
        }

        out
    }
}

impl<const N: usize> DekeRng<N> for HaltonRand<N> {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let v = self.advance_scalar(self.cursor);
        self.cursor += 1;
        if self.cursor == N {
            self.cursor = 0;
            self.outer_iters += 1;
            if self.outer_iters >= HALTON_MAX_ITERATIONS {
                self.rotate_bases();
            }
        }
        unit_f64_to_u64(v)
    }

    #[inline]
    fn sample_unit(&mut self) -> [f64; N] {
        let result = self.advance_all();
        // sample_unit reads one full row; align the per-dim cursor accordingly
        // so subsequent next_u64 calls stay coherent.
        self.cursor = 0;
        self.outer_iters += 1;
        if self.outer_iters >= HALTON_MAX_ITERATIONS {
            self.rotate_bases();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classical_halton(base: u32, mut i: u32) -> f64 {
        // Standard radical-inverse, used as ground truth for the recurrence.
        let mut f = 1.0;
        let mut r = 0.0;
        let bf = base as f64;
        while i > 0 {
            f /= bf;
            r += f * (i % base) as f64;
            i /= base;
        }
        r
    }

    #[test]
    fn halton_matches_radical_inverse_seed_zero() {
        const M: usize = 4;
        let mut h = HaltonRand::<M>::new(0);
        for k in 1..=64u32 {
            let s = h.sample_unit();
            for d in 0..M {
                let expected = classical_halton(HALTON_BASES[d], k);
                assert!(
                    (s[d] - expected).abs() < 1e-9,
                    "dim {d} step {k}: got {} want {}",
                    s[d],
                    expected,
                );
            }
        }
    }

    #[test]
    fn next_u64_and_sample_unit_agree_on_first_row() {
        const M: usize = 6;
        // After N next_u64 calls we should have advanced one full row, same as
        // one sample_unit call.
        let mut a = HaltonRand::<M>::new(0);
        let mut row_a = [0.0f64; M];
        for slot in row_a.iter_mut() {
            *slot = u64_to_unit_f64(<HaltonRand<M> as DekeRng<M>>::next_u64(&mut a));
        }

        let mut b = HaltonRand::<M>::new(0);
        let row_b = b.sample_unit();

        for d in 0..M {
            assert!(
                (row_a[d] - row_b[d]).abs() < 1e-9,
                "dim {d}: striped {} vs sample_unit {}",
                row_a[d],
                row_b[d]
            );
        }
    }

    #[test]
    fn halton_stays_in_unit_interval() {
        const M: usize = 6;
        let mut h = HaltonRand::<M>::new(12345);
        for _ in 0..10_000 {
            let s = h.sample_unit();
            for v in s {
                assert!((0.0..1.0).contains(&v), "out of range: {v}");
            }
        }
    }
}
