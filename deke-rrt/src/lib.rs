use std::fmt;

use deke_types::{DekeError, DekeResult, Planner, SRobotPath, SRobotQ, SRobotQLike, Validator};

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

/// How a planning run terminated. Use this to distinguish "we hit a wall"
/// from "we ran out of budget" without re-parsing the [`DekeError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum RrtTermination {
    /// Default for an unfinished diagnostic.
    #[default]
    NotStarted,
    /// `start ≈ goal`; a trivial single-waypoint path was returned.
    DegenerateStartGoal,
    /// Direct edge from start to goal validated; no tree search ran.
    DirectConnection,
    /// Trees connected (the normal success path).
    Solved,
    /// `max_iterations` reached without connecting.
    MaxIterationsExceeded,
    /// Combined tree size hit `max_samples` before the trees could connect.
    MaxSamplesExceeded,
    /// AORRTC saw no improvement for `stall_iterations`.
    Stalled,
    /// AORRTC reached the geometric lower bound on cost.
    OptimalReached,
    /// Planner could not parse the start/goal input.
    InputInvalid,
    /// AORRTC's initial RRTC phase failed to find any path.
    NoInitialPath,
}


impl fmt::Display for RrtTermination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::NotStarted => "not started",
            Self::DegenerateStartGoal => "degenerate start/goal",
            Self::DirectConnection => "direct connection",
            Self::Solved => "solved",
            Self::MaxIterationsExceeded => "max iterations exceeded",
            Self::MaxSamplesExceeded => "max samples exceeded",
            Self::Stalled => "stalled",
            Self::OptimalReached => "optimal cost reached",
            Self::InputInvalid => "input invalid",
            Self::NoInitialPath => "no initial path",
        };
        f.write_str(s)
    }
}

/// Counters accumulated across the main extend/connect loop. These are the
/// bread-and-butter signals when investigating why a plan failed:
///
/// - high `dynamic_domain_rejections` ⇒ the random samples keep landing
///   outside any node's adaptive ball; consider lowering `radius` /
///   `min_radius` or disabling `dynamic_domain`.
/// - high `edge_validation_failures` ⇒ steers keep colliding; consider a
///   shorter `range`, a tighter `resolution`, or revisiting collision
///   geometry.
/// - low `successful_extensions / extension_attempts` ⇒ the tree can't
///   grow; combine the two signals above.
/// - many `connect_attempts` but few `connect_successes` ⇒ trees are
///   close but separated by a thin obstacle; a finer `resolution` may help.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtensionStats {
    /// Number of main-loop iterations that performed an extend/connect attempt.
    /// Equal to `iterations` minus any iterations that exited the loop early
    /// (e.g. due to `max_samples`).
    pub extension_attempts: usize,
    /// Iterations rejected because the nearest tree node had a dynamic-domain
    /// ball smaller than the random sample's distance.
    pub dynamic_domain_rejections: usize,
    /// Total `validate_edge` calls made by the planner during extend/connect.
    /// (Path-smoothing passes after a solution is found are not counted.)
    pub edge_validations: usize,
    /// Edge validations that the validator rejected (collision or limit).
    pub edge_validation_failures: usize,
    /// Iterations where a new node was successfully added to a tree.
    pub successful_extensions: usize,
    /// Connect attempts (one per successful extension that tried to splice
    /// into the opposite tree).
    pub connect_attempts: usize,
    /// Connect attempts that joined the trees.
    pub connect_successes: usize,
}

impl ExtensionStats {
    /// Fraction of edge validations that the validator rejected. `NaN` if no
    /// validations happened.
    pub fn edge_validation_failure_rate(&self) -> f64 {
        if self.edge_validations == 0 {
            f64::NAN
        } else {
            self.edge_validation_failures as f64 / self.edge_validations as f64
        }
    }

    /// Fraction of extension attempts that the planner skipped because the
    /// dynamic-domain ball excluded the sample.
    pub fn dynamic_domain_rejection_rate(&self) -> f64 {
        if self.extension_attempts == 0 {
            f64::NAN
        } else {
            self.dynamic_domain_rejections as f64 / self.extension_attempts as f64
        }
    }

    /// Fraction of connect attempts that succeeded.
    pub fn connect_success_rate(&self) -> f64 {
        if self.connect_attempts == 0 {
            f64::NAN
        } else {
            self.connect_successes as f64 / self.connect_attempts as f64
        }
    }
}

impl fmt::Display for ExtensionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "attempts={} dyn_dom_rej={} validations={} (failed {}) extensions={} connects={}/{}",
            self.extension_attempts,
            self.dynamic_domain_rejections,
            self.edge_validations,
            self.edge_validation_failures,
            self.successful_extensions,
            self.connect_successes,
            self.connect_attempts,
        )
    }
}

/// AORRTC-specific anytime-refinement information. `None` for plain RRTC and
/// KRRTC.
#[derive(Debug, Clone, Copy)]
pub struct AnytimeInfo {
    /// Cost of the very first feasible path, before any anytime refinement.
    pub initial_cost: f64,
    /// Iterations consumed by the initial RRTC phase that produced
    /// `initial_cost`.
    pub initial_iterations: usize,
    /// Number of times a strictly cheaper path was found during refinement.
    pub improvements: usize,
    /// Iterations since the last cost improvement when planning ended.
    pub iters_since_last_improvement: usize,
    /// `path_cost / c_min` — `1.0` is provably optimal, `+inf` means no path.
    pub optimality_ratio: f64,
}

impl fmt::Display for AnytimeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "initial_cost={:.4} (in {} iters) improvements={} stall={} opt_ratio={:.3}",
            self.initial_cost,
            self.initial_iterations,
            self.improvements,
            self.iters_since_last_improvement,
            self.optimality_ratio,
        )
    }
}

#[derive(Debug, Clone)]
pub struct RrtDiagnostic {
    pub iterations: usize,
    pub start_tree_size: usize,
    pub goal_tree_size: usize,
    pub path_cost: f64,
    pub elapsed_ns: u128,
    /// Why the planner stopped. See [`RrtTermination`] for all variants.
    pub termination: RrtTermination,
    /// Counters over the main extend/connect loop. See [`ExtensionStats`] for
    /// what each field means and how to read them as a failure signature.
    pub extension_stats: ExtensionStats,
    /// Lower bound on path cost: weighted distance from start to goal under
    /// the planner's distance metric. Compare against `path_cost` for an
    /// optimality estimate (`path_cost / c_min`); when the planner failed,
    /// this is still meaningful as a "the goal was at least this far away."
    pub c_min: f64,
    /// Closest cross-tree distance observed during the run, in cost units.
    /// On success this is `0`; on failure it tells you how close the trees
    /// actually got — a small value points at a thin obstacle / tight passage,
    /// a large value points at trees stuck in disconnected regions.
    pub closest_approach: f64,
    /// AORRTC-only anytime info. `None` for plain RRTC and KRRTC.
    pub anytime: Option<AnytimeInfo>,
}

impl fmt::Display for RrtDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "iters={} trees=({},{}) cost={:.4} c_min={:.4} closest={:.4} time={:.2}ms term={} [{}]",
            self.iterations,
            self.start_tree_size,
            self.goal_tree_size,
            self.path_cost,
            self.c_min,
            self.closest_approach,
            self.elapsed_ns as f64 / 1_000_000.0,
            self.termination,
            self.extension_stats,
        )?;
        if let Some(at) = &self.anytime {
            write!(f, " anytime=[{}]", at)?;
        }
        Ok(())
    }
}

/// Start and goal configuration for a single-query planner. Build it with
/// [`StartEnd::new`], which accepts any [`SRobotQLike`] inputs and resolves them
/// to fixed-size configurations up front, so planning itself never has to parse
/// the endpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StartEnd<const N: usize> {
    pub start: SRobotQ<N, f64>,
    pub end: SRobotQ<N, f64>,
}

impl<const N: usize> StartEnd<N> {
    /// Resolve a start and goal from any [`SRobotQLike`] inputs. The two
    /// endpoints may be different input types (e.g. an `SRobotQ` start and a
    /// `Vec` goal). Returns the input's conversion error (e.g.
    /// [`DekeError::ShapeMismatch`]) if either cannot be made into an `N`-joint
    /// configuration.
    pub fn new<Es, Eg, S, G>(start: S, end: G) -> DekeResult<Self>
    where
        Es: Into<DekeError>,
        Eg: Into<DekeError>,
        S: SRobotQLike<N, Es, f64>,
        G: SRobotQLike<N, Eg, f64>,
    {
        Ok(Self {
            start: start.to_srobotq().map_err(Into::into)?,
            end: end.to_srobotq().map_err(Into::into)?,
        })
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
    type Waypoints = StartEnd<N>;

    fn plan<E: Into<DekeError>, V: Validator<N, (), f64>>(
        &self,
        config: &Self::Config,
        waypoints: &Self::Waypoints,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotPath<N, f64>>, Self::Diagnostic) {
        let mut rng = DekeRand::<N>::new(config.randomizer, config.seed);
        rrtc::solve(&waypoints.start, &waypoints.end, validator, ctx, config, &mut rng)
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
    type Waypoints = StartEnd<N>;

    fn plan<E: Into<DekeError>, V: Validator<N, (), f64>>(
        &self,
        config: &Self::Config,
        waypoints: &Self::Waypoints,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotPath<N, f64>>, Self::Diagnostic) {
        let mut sample_rng = DekeRand::<N>::new(config.rrtc.randomizer, config.rrtc.seed);
        let mut aux_rng = DekeRand::<N>::new(config.aux_randomizer, config.aux_seed);
        aorrtc::solve(
            &waypoints.start,
            &waypoints.end,
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
    type Waypoints = StartEnd<N>;

    fn plan<E: Into<DekeError>, V: Validator<N, (), f64>>(
        &self,
        config: &Self::Config,
        waypoints: &Self::Waypoints,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotPath<N, f64>>, Self::Diagnostic) {
        let mut rng = DekeRand::<N>::new(config.randomizer, config.seed);
        krrtc::solve(&waypoints.start, &waypoints.end, validator, ctx, config, &mut rng)
    }
}
