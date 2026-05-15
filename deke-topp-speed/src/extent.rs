//! Per-axis value extent with associated times.

use num_traits::Float;

/// The minimum and maximum value an axis reaches over its trajectory, plus the
/// times at which each extremum occurs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extent<F: Float = f32> {
    /// Lowest value observed.
    pub min: F,
    /// Highest value observed.
    pub max: F,
    /// Time at which `min` is reached.
    pub t_min: F,
    /// Time at which `max` is reached.
    pub t_max: F,
}

impl<F: Float> Extent<F> {
    /// Construct an `Extent` whose min and max both equal `value`, with both
    /// times set to zero.
    pub fn point(value: F) -> Self {
        Self {
            min: value,
            max: value,
            t_min: F::zero(),
            t_max: F::zero(),
        }
    }

    /// Width of the extent (`max - min`).
    pub fn span(&self) -> F {
        self.max - self.min
    }
}

impl<F: Float + Default> Default for Extent<F> {
    fn default() -> Self {
        Self {
            min: F::zero(),
            max: F::zero(),
            t_min: F::zero(),
            t_max: F::zero(),
        }
    }
}
