//! Spatial weave (torch oscillation) overlaid on a seam.
//!
//! Phase-1 weave: a transverse oscillation locked to **seam arc length** (pure
//! geometry), superimposed on the nominal seam pose in Stage A. It flows through
//! the analytic-IK planner unchanged (the planner just IK's a richer path) and
//! through the [`crate::ConstantSpeedRetimer`] at constant **travel speed** (seam
//! progress), which is what governs heat input per unit weld — see
//! [`crate::ConstantSpeedRetimer::retime_weave`]. The transverse direction is taken
//! from the **tool frame** (`R(s)·axis`), which is degeneracy-free, unlike the
//! Frenet normal that is undefined on the straight seams one weaves most.

use std::f64::consts::{PI, TAU};

use crate::redundant::RedundantAxis;

/// Transverse oscillation shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WeavePattern {
    #[default]
    Sine,
    /// Linear ramps between the extremes (sharper edge dwell than a sine).
    Triangle,
}

impl WeavePattern {
    /// Unit transverse offset in `[-1, 1]` at phase `phi` (radians).
    pub fn shape(&self, phi: f64) -> f64 {
        match self {
            WeavePattern::Sine => phi.sin(),
            WeavePattern::Triangle => (2.0 / PI) * phi.sin().asin(),
        }
    }
}

/// Spatial weave parameters. The oscillation is transverse to travel in the tool
/// frame and locked to seam arc length, so the bead pattern is fixed on the
/// workpiece regardless of travel speed.
#[derive(Clone, Copy, Debug)]
pub struct WeaveOptions {
    /// Tool-frame transverse axis the torch oscillates along (e.g. tool +Y). Taken
    /// through the run orientation `R(s)`, so it is robust on straight seams.
    pub axis: RedundantAxis,
    /// Peak-to-peak (tip-to-tip) transverse width, metres.
    pub amplitude: f64,
    /// Spatial period along the seam, metres (`= travel_speed / frequency`).
    pub wavelength: f64,
    /// Oscillation shape.
    pub pattern: WeavePattern,
    /// Distance (metres of seam) over which the amplitude ramps 0→1 at each run
    /// end, so the weave vanishes into the start/stop rest ramps.
    pub taper: f64,
}

impl WeaveOptions {
    /// A sine weave; taper defaults to two wavelengths.
    pub fn sine(amplitude: f64, wavelength: f64) -> Self {
        Self {
            axis: RedundantAxis::PosY,
            amplitude,
            wavelength,
            pattern: WeavePattern::Sine,
            taper: 2.0 * wavelength,
        }
    }

    /// The spatial wavelength for a temporal `frequency` (Hz) at a given
    /// `travel_speed` (m/s) — the bridge between the welder's "2 Hz at 18 IPM" and
    /// the spatial weave the path carries.
    pub fn wavelength_for(frequency: f64, travel_speed: f64) -> f64 {
        travel_speed / frequency.max(1e-9)
    }

    /// Largest planner `sample_ds` that still resolves the weave without aliasing
    /// (≥ ~15 samples per cycle). The weave *is* the high-frequency path content, so
    /// there is no coarse-grid escape — sample at least this finely.
    pub fn max_sample_ds(&self) -> f64 {
        self.wavelength / 15.0
    }

    /// Signed transverse offset (metres) at seam arc length `s` on a run of length
    /// `length`: `0.5·amplitude·envelope(s)·shape(2π s/λ)`.
    pub(crate) fn offset(&self, s: f64, length: f64) -> f64 {
        let t = self.taper.max(1e-9);
        let a = (s / t).clamp(0.0, 1.0);
        let b = ((length - s) / t).clamp(0.0, 1.0);
        let smooth = |x: f64| x * x * (3.0 - 2.0 * x);
        let env = smooth(a) * smooth(b);
        0.5 * self.amplitude * env * self.pattern.shape(TAU * s / self.wavelength.max(1e-9))
    }
}
