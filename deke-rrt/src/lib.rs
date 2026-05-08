use std::fmt;

use deke_types::{DekeError, DekeResult, Planner, SRobotPath, SRobotQLike, Validator};

mod aorrtc;
mod krrtc;
mod randomizer;
mod rrtc;
pub mod scurve;
mod tree;

#[cfg(feature = "valuable")]
mod valuable_impls;

pub use aorrtc::AorrtcSettings;
pub use krrtc::KrrtcSettings;
pub use randomizer::{DekeRand, DekeRng, HaltonRand, RandomizerType};
pub use rrtc::RrtcSettings;
pub use scurve::{JointKinLimits, KinematicLimits, direction_cosine};

#[derive(Debug, Clone)]
pub struct RrtDiagnostic {
    pub iterations: usize,
    pub start_tree_size: usize,
    pub goal_tree_size: usize,
    pub path_cost: f64,
    pub elapsed_ns: u128,
}

impl RrtDiagnostic {
    fn empty() -> Self {
        Self {
            iterations: 0,
            start_tree_size: 0,
            goal_tree_size: 0,
            path_cost: 0.0,
            elapsed_ns: 0,
        }
    }
}

impl fmt::Display for RrtDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "iters={} trees=({},{}) cost={:.4} time={:.2}ms",
            self.iterations,
            self.start_tree_size,
            self.goal_tree_size,
            self.path_cost,
            self.elapsed_ns as f64 / 1_000_000.0,
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct RrtcPlanner<const N: usize>;

impl<const N: usize> RrtcPlanner<N> {
    pub fn new() -> Self {
        Self
    }
}

impl<const N: usize> Planner<N, f64> for RrtcPlanner<N> {
    type Diagnostic = RrtDiagnostic;
    type Config = RrtcSettings<N>;

    fn plan<
        E: Into<DekeError>,
        A: SRobotQLike<N, E, f64>,
        B: SRobotQLike<N, E, f64>,
        V: Validator<N, (), f64>,
    >(
        &self,
        config: &Self::Config,
        start: A,
        goal: B,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotPath<N, f64>>, Self::Diagnostic) {
        let start = match start.to_srobotq().map_err(Into::into) {
            Ok(s) => s,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let goal = match goal.to_srobotq().map_err(Into::into) {
            Ok(g) => g,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let mut rng = DekeRand::<N>::new(config.randomizer, config.seed);
        rrtc::solve(&start, &goal, validator, ctx, config, &mut rng)
    }
}

#[derive(Debug, Clone, Default)]
pub struct AorrtcPlanner<const N: usize>;

impl<const N: usize> AorrtcPlanner<N> {
    pub fn new() -> Self {
        Self
    }
}

impl<const N: usize> Planner<N, f64> for AorrtcPlanner<N> {
    type Diagnostic = RrtDiagnostic;
    type Config = AorrtcSettings<N>;

    fn plan<
        E: Into<DekeError>,
        A: SRobotQLike<N, E, f64>,
        B: SRobotQLike<N, E, f64>,
        V: Validator<N, (), f64>,
    >(
        &self,
        config: &Self::Config,
        start: A,
        goal: B,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotPath<N, f64>>, Self::Diagnostic) {
        let start = match start.to_srobotq().map_err(Into::into) {
            Ok(s) => s,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let goal = match goal.to_srobotq().map_err(Into::into) {
            Ok(g) => g,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let mut sample_rng = DekeRand::<N>::new(config.rrtc.randomizer, config.rrtc.seed);
        let mut aux_rng = DekeRand::<N>::new(config.aux_randomizer, config.aux_seed);
        aorrtc::solve(
            &start,
            &goal,
            validator,
            ctx,
            config,
            &mut sample_rng,
            &mut aux_rng,
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct KrrtcPlanner<const N: usize>;

impl<const N: usize> KrrtcPlanner<N> {
    pub fn new() -> Self {
        Self
    }
}

impl<const N: usize> Planner<N, f64> for KrrtcPlanner<N> {
    type Diagnostic = RrtDiagnostic;
    type Config = KrrtcSettings<N>;

    fn plan<
        E: Into<DekeError>,
        A: SRobotQLike<N, E, f64>,
        B: SRobotQLike<N, E, f64>,
        V: Validator<N, (), f64>,
    >(
        &self,
        config: &Self::Config,
        start: A,
        goal: B,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotPath<N, f64>>, Self::Diagnostic) {
        let start = match start.to_srobotq().map_err(Into::into) {
            Ok(s) => s,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let goal = match goal.to_srobotq().map_err(Into::into) {
            Ok(g) => g,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let mut rng = DekeRand::<N>::new(config.randomizer, config.seed);
        krrtc::solve(&start, &goal, validator, ctx, config, &mut rng)
    }
}
