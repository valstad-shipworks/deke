//! Optimal ordering and orientation of *required* robot paths.
//!
//! Given a set of paths that must each be traversed exactly once — some of
//! which may be reversible or have several interchangeable realizations — this
//! crate chooses one option per required path and orders them to minimise total
//! motion cost, starting at a configuration and optionally ending at one. It
//! returns the full stitched motion: connector segments interleaved with the
//! chosen required paths.
//!
//! The ordering is an asymmetric generalized TSP (see [`agtsp`]); cost is a
//! pluggable joint-space metric ([`TransitionCost`]). The planner is only used
//! to *generate* connector paths and is optional — [`plan_multipath`] uses a
//! planner for obstacle-aware connectors, [`plan_multipath_straight`] emits
//! validated straight-line connectors instead.
//!
//! ```no_run
//! # use deke_multipath::*;
//! # use deke_types::{SRobotQ, SRobotPath, Planner, Validator};
//! # fn go<'ctx, const N: usize, P, V, MW>(
//! #     paths: Vec<ReqPath<N>>, planner: &P, cfg: &P::Config, make_wp: MW,
//! #     validator: &V, ctx: &V::Context<'ctx>, start: SRobotQ<N, f64>, weights: SRobotQ<N, f64>,
//! # ) -> MultipathResult<Vec<SRobotPath<N, f64>>>
//! # where P: Planner<N, f64> + Sync, P::Config: Sync, V: Validator<N, (), f64>,
//! #       V::Context<'ctx>: Sync,
//! #       MW: Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> P::Waypoints + Sync {
//! let settings = MultiPathSettings::new(start);
//! let cost = TransitionCost::JointWeighted(weights);
//! let transition = TransitionPlanner { planner, config: cfg, make_waypoints: make_wp };
//! plan_multipath(&paths, &cost, &settings, &transition, validator, ctx)
//! # }
//! ```
//!
//! # Feature flags
//!
//! - `rayon` — use rayon to plan connectors concurrently and to fan out the
//!   heuristic's seed tours. Worth enabling when connector planning is the
//!   bottleneck (real collision-checked RRT, where each plan costs milliseconds)
//!   or for large heuristic instances; for trivially cheap connectors the
//!   thread-pool dispatch can cost more than it saves. The exact Held–Karp
//!   solver stays sequential — its layers depend on one another, so it does not
//!   parallelize without a structural rewrite. Enabling it adds `Sync` bounds on
//!   the planner, its config, the waypoint builder, and the validator context
//!   (all satisfied by the deke planners/validators).

mod agtsp;
mod cost;
mod error;
mod reqpath;

use deke_types::{DekeError, Planner, SRobotPath, SRobotQ, Validator};

use agtsp::Problem;
use cost::build_matrices;
use reqpath::{DirectedOption, expand};

pub use cost::{TransitionCost, weighted_distance};
pub use error::{MultipathError, MultipathResult};
pub use reqpath::ReqPath;

/// `Sync` exactly when the `rayon` feature is enabled. The public API requires
/// thread-safety only when work is actually dispatched across the rayon pool;
/// the sequential default build imposes no such bound. Blanket-implemented for
/// every type, so callers never name or implement it — it only appears in the
/// public bounds, hence `#[doc(hidden)] pub`.
#[doc(hidden)]
#[cfg(feature = "rayon")]
pub trait MaybeSync: Sync {}
#[cfg(feature = "rayon")]
impl<T: Sync> MaybeSync for T {}

#[doc(hidden)]
#[cfg(not(feature = "rayon"))]
pub trait MaybeSync {}
#[cfg(not(feature = "rayon"))]
impl<T> MaybeSync for T {}

/// Knobs for a multipath solve. Build with [`MultiPathSettings::new`].
pub struct MultiPathSettings<const N: usize> {
    /// Configuration the motion starts from.
    pub start: SRobotQ<N, f64>,
    /// Optional configuration the motion must finish at.
    pub end: Option<SRobotQ<N, f64>>,
    /// Connectors whose endpoints coincide within this (unweighted) joint
    /// distance are skipped — no point planning a move that goes nowhere.
    pub dedup_tol: f64,
    /// Above this many DP cells the solver switches from exact Held–Karp to the
    /// nearest-neighbour + 2-opt heuristic.
    pub exact_cell_budget: usize,
}

impl<const N: usize> MultiPathSettings<N> {
    pub fn new(start: SRobotQ<N, f64>) -> Self {
        Self {
            start,
            end: None,
            dedup_tol: 1e-6,
            exact_cell_budget: agtsp::DEFAULT_CELL_BUDGET,
        }
    }

    pub fn with_end(mut self, end: SRobotQ<N, f64>) -> Self {
        self.end = Some(end);
        self
    }
}

/// Everything needed to generate real connector paths with a deke planner. The
/// `make_waypoints` closure bridges the planner's associated `Waypoints` type —
/// for the RRT planners that is `|s, e| StartEnd { start: s, end: e }`.
pub struct TransitionPlanner<'a, const N: usize, P, MW>
where
    P: Planner<N, f64>,
{
    pub planner: &'a P,
    pub config: &'a P::Config,
    pub make_waypoints: MW,
}

/// Solve the ordering and stitch the full motion using `transition` to plan
/// obstacle-aware connectors between the chosen paths.
pub fn plan_multipath<'ctx, const N: usize, P, V, MW>(
    req_paths: &[ReqPath<N>],
    cost: &TransitionCost<N>,
    settings: &MultiPathSettings<N>,
    transition: &TransitionPlanner<'_, N, P, MW>,
    validator: &V,
    ctx: &V::Context<'ctx>,
) -> MultipathResult<Vec<SRobotPath<N, f64>>>
where
    P: Planner<N, f64> + MaybeSync,
    P::Config: MaybeSync,
    V: Validator<N, (), f64>,
    V::Context<'ctx>: MaybeSync,
    MW: Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> P::Waypoints + MaybeSync,
{
    run(req_paths, cost, settings, |from, to| {
        let waypoints = (transition.make_waypoints)(*from, *to);
        let (path, _diag) =
            transition
                .planner
                .plan::<DekeError, _>(transition.config, &waypoints, validator, ctx);
        Ok(path?)
    })
}

/// Solve the ordering and stitch the full motion using straight joint-space
/// connectors. Each connector is checked with `validator.validate_motion`; if a
/// straight line between two required paths is infeasible the solve fails (there
/// is no planner to route around the obstacle).
pub fn plan_multipath_straight<'ctx, const N: usize, V>(
    req_paths: &[ReqPath<N>],
    cost: &TransitionCost<N>,
    settings: &MultiPathSettings<N>,
    validator: &V,
    ctx: &V::Context<'ctx>,
) -> MultipathResult<Vec<SRobotPath<N, f64>>>
where
    V: Validator<N, (), f64>,
    V::Context<'ctx>: MaybeSync,
{
    run(req_paths, cost, settings, |from, to| {
        validator.validate_motion(&[*from, *to], ctx)?;
        Ok(SRobotPath::from_two(*from, *to))
    })
}

fn run<const N: usize, C>(
    req_paths: &[ReqPath<N>],
    cost: &TransitionCost<N>,
    settings: &MultiPathSettings<N>,
    connect: C,
) -> MultipathResult<Vec<SRobotPath<N, f64>>>
where
    C: Fn(&SRobotQ<N, f64>, &SRobotQ<N, f64>) -> MultipathResult<SRobotPath<N, f64>> + MaybeSync,
{
    let (options, n_clusters) = expand(req_paths)?;
    let matrices = build_matrices(&options, cost, &settings.start, settings.end.as_ref());
    let cluster_ids: Vec<usize> = options.iter().map(|o| o.cluster).collect();
    let problem = Problem {
        cluster_ids: &cluster_ids,
        n_clusters,
        transition: &matrices.transition,
        options: options.len(),
        start: &matrices.start,
        end: &matrices.end,
    };
    let solution =
        agtsp::solve(&problem, settings.exact_cell_budget).ok_or(MultipathError::NoFeasibleTour)?;
    tracing::debug!(
        n_required = req_paths.len(),
        n_options = options.len(),
        cost = solution.cost,
        "deke-multipath: ordering solved"
    );
    assemble(options, &solution.order, settings, connect)
}

/// One element of the stitched plan before connectors are realized: either a
/// chosen required path (already oriented) or a connector that still needs to be
/// planned between two configurations.
enum Segment<const N: usize> {
    Connector(SRobotQ<N, f64>, SRobotQ<N, f64>),
    Required(SRobotPath<N, f64>),
}

fn assemble<const N: usize, C>(
    options: Vec<DirectedOption<N>>,
    order: &[usize],
    settings: &MultiPathSettings<N>,
    connect: C,
) -> MultipathResult<Vec<SRobotPath<N, f64>>>
where
    C: Fn(&SRobotQ<N, f64>, &SRobotQ<N, f64>) -> MultipathResult<SRobotPath<N, f64>> + MaybeSync,
{
    // Lay out the plan as connector/required segments. Connectors are
    // independent point-to-point plans, so they can be realized concurrently
    // (the costly RRT case); the required paths just pass through. The tour
    // visits each option at most once, so the chosen paths are moved out rather
    // than cloned.
    let mut options: Vec<Option<DirectedOption<N>>> = options.into_iter().map(Some).collect();
    let mut segments: Vec<Segment<N>> = Vec::new();
    let mut cursor = settings.start;
    for &oi in order {
        let path = options[oi]
            .take()
            .expect("each option appears once in the tour")
            .path;
        let first = *path.first();
        if cursor.distance(&first) > settings.dedup_tol {
            segments.push(Segment::Connector(cursor, first));
        }
        cursor = *path.last();
        segments.push(Segment::Required(path));
    }
    if let Some(end) = settings.end
        && cursor.distance(&end) > settings.dedup_tol
    {
        segments.push(Segment::Connector(cursor, end));
    }

    let realize = |segment: Segment<N>| -> MultipathResult<SRobotPath<N, f64>> {
        match segment {
            Segment::Connector(from, to) => connect(&from, &to),
            Segment::Required(path) => Ok(path),
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        segments.into_par_iter().map(realize).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        segments.into_iter().map(realize).collect()
    }
}
