use std::fmt;

use deke_types::{Planner, DekeError, DekeResult, RobotPath, SRobotQ, Validator};
use tinyrand::{Seeded, StdRand};

mod aorrtc;
mod krrtc;
mod rrtc;
pub mod scurve;
mod tree;

pub use aorrtc::AorrtcSettings;
pub use krrtc::KrrtcSettings;
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

#[derive(Debug, Clone)]
pub struct RrtcPlanner<const N: usize> {
    pub settings: RrtcSettings<N>,
}

impl<const N: usize> RrtcPlanner<N> {
    pub fn new(settings: RrtcSettings<N>) -> Self {
        Self { settings }
    }
}

impl<const N: usize> Planner<N> for RrtcPlanner<N> {
    type Diagnostic = RrtDiagnostic;

    fn plan<
        E: Into<DekeError>,
        A: TryInto<SRobotQ<N>, Error = E>,
        B: TryInto<SRobotQ<N>, Error = E>,
    >(
        &self,
        start: A,
        goal: B,
        validators: &mut impl Validator<N>,
    ) -> (DekeResult<RobotPath>, Self::Diagnostic) {
        let start = match start.try_into().map_err(|e| e.into()) {
            Ok(s) => s,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let goal = match goal.try_into().map_err(|e| e.into()) {
            Ok(g) => g,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let mut rng = StdRand::seed(self.settings.seed);
        rrtc::solve(&start, &goal, validators, &self.settings, &mut rng)
    }
}

#[derive(Debug, Clone)]
pub struct AorrtcPlanner<const N: usize> {
    pub settings: AorrtcSettings<N>,
}

impl<const N: usize> AorrtcPlanner<N> {
    pub fn new(settings: AorrtcSettings<N>) -> Self {
        Self { settings }
    }
}

impl<const N: usize> Planner<N> for AorrtcPlanner<N> {
    type Diagnostic = RrtDiagnostic;

    fn plan<
        E: Into<DekeError>,
        A: TryInto<SRobotQ<N>, Error = E>,
        B: TryInto<SRobotQ<N>, Error = E>,
    >(
        &self,
        start: A,
        goal: B,
        validators: &mut impl Validator<N>,
    ) -> (DekeResult<RobotPath>, Self::Diagnostic) {
        let start = match start.try_into().map_err(|e| e.into()) {
            Ok(s) => s,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let goal = match goal.try_into().map_err(|e| e.into()) {
            Ok(g) => g,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let mut rng = StdRand::seed(self.settings.rrtc.seed);
        aorrtc::solve(&start, &goal, validators, &self.settings, &mut rng)
    }
}

#[derive(Debug, Clone)]
pub struct KrrtcPlanner<const N: usize> {
    pub settings: KrrtcSettings<N>,
}

impl<const N: usize> KrrtcPlanner<N> {
    pub fn new(settings: KrrtcSettings<N>) -> Self {
        Self { settings }
    }
}

impl<const N: usize> Planner<N> for KrrtcPlanner<N> {
    type Diagnostic = RrtDiagnostic;

    fn plan<
        E: Into<DekeError>,
        A: TryInto<SRobotQ<N>, Error = E>,
        B: TryInto<SRobotQ<N>, Error = E>,
    >(
        &self,
        start: A,
        goal: B,
        validators: &mut impl Validator<N>,
    ) -> (DekeResult<RobotPath>, Self::Diagnostic) {
        let start = match start.try_into().map_err(|e| e.into()) {
            Ok(s) => s,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let goal = match goal.try_into().map_err(|e| e.into()) {
            Ok(g) => g,
            Err(e) => return (Err(e), RrtDiagnostic::empty()),
        };
        let mut rng = StdRand::seed(self.settings.seed);
        krrtc::solve(&start, &goal, validators, &self.settings, &mut rng)
    }
}
