use halton::Sequence;
use tinyrand::{Rand, Seeded, SplitMix, Wyrand, Xorshift};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomizerType {
    Wyrand,
    SplitMix,
    Xorshift,
    Halton,
}

impl Default for RandomizerType {
    fn default() -> Self {
        Self::Wyrand
    }
}

pub enum DekeRand<const N: usize> {
    Wyrand(Wyrand),
    SplitMix(SplitMix),
    Xorshift(Xorshift),
    Halton(HaltonRand<N>),
}

impl<const N: usize> DekeRand<N> {
    /// `N` is the per-sample stripe width — the number of consecutive
    /// `next_u64` calls that map onto one logical N-D sample. Only the Halton
    /// variant uses it; the PRNG variants ignore it. Must satisfy
    /// `0 < N <= HALTON_BASES.len()` when `kind == Halton`.
    pub fn new(kind: RandomizerType, seed: u64) -> Self {
        match kind {
            RandomizerType::Wyrand => Self::Wyrand(Wyrand::seed(seed)),
            RandomizerType::SplitMix => Self::SplitMix(SplitMix::seed(seed)),
            RandomizerType::Xorshift => Self::Xorshift(Xorshift::seed(seed)),
            RandomizerType::Halton => Self::Halton(HaltonRand::new(seed)),
        }
    }
}

impl<const N: usize> Rand for DekeRand<N> {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        match self {
            DekeRand::Wyrand(r) => r.next_u64(),
            DekeRand::SplitMix(r) => r.next_u64(),
            DekeRand::Xorshift(r) => r.next_u64(),
            DekeRand::Halton(r) => r.next_u64(),
        }
    }
}

/// Bases used by the Halton sampler.
const HALTON_BASES: [u8; 16] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];

const HALTON_MAX_ITERATIONS: usize = 1_000_000;

/// Quasi-random source that emits a striped Halton sequence sized to the
/// caller's joint dimension `N`. Holds exactly `N` sequences (one prime per
/// dimension) and wraps every `N` calls so a caller that pulls `N` values
/// per outer iteration always reads dimension-0 from base `HALTON_BASES[0]`,
/// dimension-1 from `HALTON_BASES[1]`, etc. — the standard Halton
/// construction. If the cycle were tied to a length other than `N`, the
/// base-to-dimension assignment would drift across iterations and the
/// per-axis low-discrepancy guarantee would be lost.
pub struct HaltonRand<const N: usize> {
    seqs: [Sequence; N],
    base_offset: usize,
    cursor: usize,
    outer_iters: usize,
    seed: u64,
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
        let seqs: [Sequence; N] = std::array::from_fn(|i| Sequence::new(HALTON_BASES[i]));
        let mut s = Self {
            seqs,
            base_offset: 0,
            cursor: 0,
            outer_iters: 0,
            seed,
        };
        s.apply_seed_skip();
        s
    }

    fn apply_seed_skip(&mut self) {
        // Halton is deterministic; the seed picks a starting offset so different
        // seeds yield different (still low-discrepancy) subsequences. Cap the
        // skip so we don't burn the precision budget before the first sample.
        let skip = (self.seed % 4096) as usize;
        if skip > 0 {
            for s in &mut self.seqs {
                for _ in 0..skip {
                    s.next();
                }
            }
        }
    }

    fn rotate_bases(&mut self) {
        // VAMP rotates the dimension→base mapping after the precision budget
        // is exhausted; we mirror that by shifting `base_offset` and resetting
        // the underlying sequences.
        self.base_offset = (self.base_offset + 1) % N;
        self.seqs = std::array::from_fn(|i| Sequence::new(HALTON_BASES[i]));
        self.outer_iters = 0;
        self.apply_seed_skip();
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        let idx = (self.base_offset + self.cursor) % N;
        let v = self.seqs[idx].next().unwrap_or(0.5);

        self.cursor += 1;
        if self.cursor == N {
            self.cursor = 0;
            self.outer_iters += 1;
            if self.outer_iters >= HALTON_MAX_ITERATIONS {
                self.rotate_bases();
            }
        }
        v
    }
}

impl<const N: usize> Rand for HaltonRand<N> {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        // Map the (0,1) Halton value to a u64 such that the consumer's
        // `(x >> 11) as f64 * 2^-53` round-trip recovers the same f64.
        let v = self.next_f64();
        let bits = (v * (1u64 << 53) as f64) as u64;
        bits << 11
    }
}
