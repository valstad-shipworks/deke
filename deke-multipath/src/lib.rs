//! Optimal ordering and orientation of *required* robot paths.
//!
//! Given a set of paths that must each be traversed exactly once — some of
//! which may be reversible or have several interchangeable realizations — this
//! crate chooses one option per required path and orders them to minimise total
//! motion cost, starting at a configuration and optionally ending at one. It
//! returns the full stitched motion: connector segments interleaved with the
//! chosen required paths.
//!
//! The ordering is an asymmetric generalized TSP (see [`agtsp`]); cost is any
//! `Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> f64` scoring a transition between
//! two configurations. [`weighted_euclidean`], [`planned_path_length`] and
//! [`planned_trajectory_time`] build the common ones (cheap joint distance,
//! planner arc length, retimed trajectory time). The planner passed to
//! [`plan_multipath`] is only used to *generate* connector paths in the output
//! — [`plan_multipath_straight`] emits validated straight-line connectors
//! instead.
//!
//! If you already hold a precomputed `option × option` cost matrix and want the
//! chosen option index per cluster (rather than stitched paths), call
//! [`solve_matrix`] / [`solve_matrix_multi_start`] directly.
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
//! let cost = weighted_euclidean(weights);
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

use std::cmp::Ordering;
use std::collections::HashMap;

use deke_types::{DekeError, Planner, SRobotPath, SRobotQ, Validator};

use agtsp::Problem;
use cost::build_matrices;
use reqpath::{DirectedOption, expand};

pub use cost::{
    planned_path_length, planned_trajectory_time, weighted_distance, weighted_euclidean,
};
pub use error::{MultipathError, MultipathResult};
pub use reqpath::ReqPath;

/// Default Held–Karp cell budget: above this many DP cells the solver switches
/// from exact to the nearest-neighbour + 2-opt heuristic. `16 * 1024 * 1024`
/// cells ≈ 256 MB.
pub const DEFAULT_CELL_BUDGET: usize = agtsp::DEFAULT_CELL_BUDGET;

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
pub fn plan_multipath<'ctx, const N: usize, P, V, MW, C>(
    req_paths: &[ReqPath<N>],
    cost: &C,
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
    C: Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> f64,
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
pub fn plan_multipath_straight<'ctx, const N: usize, V, C>(
    req_paths: &[ReqPath<N>],
    cost: &C,
    settings: &MultiPathSettings<N>,
    validator: &V,
    ctx: &V::Context<'ctx>,
) -> MultipathResult<Vec<SRobotPath<N, f64>>>
where
    V: Validator<N, (), f64>,
    V::Context<'ctx>: MaybeSync,
    C: Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> f64,
{
    run(req_paths, cost, settings, |from, to| {
        validator.validate_motion(&[*from, *to], ctx)?;
        Ok(SRobotPath::from_two(*from, *to))
    })
}

/// Solve the asymmetric generalized TSP directly over a precomputed cost matrix,
/// returning the chosen option index per cluster in visiting order and the total
/// cost — without expanding paths or stitching motion. This is the bare ordering
/// core for callers that already have their own cost structure and only want the
/// selection and order back.
///
/// - `cluster_ids[i]` is the cluster option `i` belongs to. Cluster labels may be
///   any `usize`; they are densified internally, so the returned indices are
///   positions into `cluster_ids` regardless of how the clusters are numbered.
/// - `transition` is the `option × option` matrix where `(i, j)` is the cost to
///   move from option `i` to option `j` (fold option `j`'s own traversal cost
///   into the entry if you want the solver to prefer cheaper realizations). Must
///   be square with side `cluster_ids.len()`.
/// - `start[i]` is the cost to begin the tour at option `i`; `end[i]` the cost to
///   finish at it. `None` means a zero vector (no start/end bias).
/// - Non-finite entries mark infeasible edges and are routed around.
/// - `cell_budget` switches exact Held–Karp → heuristic above that many DP cells;
///   pass [`DEFAULT_CELL_BUDGET`] for the standard 16M-cell cap.
///
/// Returns `None` if no feasible tour visits every cluster exactly once.
pub fn solve_matrix(
    cluster_ids: &[usize],
    transition: &[Vec<f64>],
    start: Option<&[f64]>,
    end: Option<&[f64]>,
    cell_budget: usize,
) -> Option<(Vec<usize>, f64)> {
    let prepared = MatrixProblem::new(cluster_ids, transition, start, end);
    let solution = agtsp::solve(&prepared.problem(&prepared.start), cell_budget)?;
    Some((solution.order, solution.cost))
}

/// Multi-start, top-k variant of [`solve_matrix`]: run one solve per starting
/// cluster (each constrained to begin its tour at that cluster) and return the
/// `k` cheapest tours found, ascending by cost. Useful when a single optimum is
/// not enough — e.g. choosing among near-equal orderings by a downstream metric
/// the cost matrix does not capture.
///
/// Arguments are identical to [`solve_matrix`]. The per-start solves are
/// independent and fan out across the rayon pool when the `rayon` feature is
/// enabled. At most one tour is returned per starting cluster, so the result has
/// at most `min(k, n_clusters)` entries.
pub fn solve_matrix_multi_start(
    cluster_ids: &[usize],
    transition: &[Vec<f64>],
    start: Option<&[f64]>,
    end: Option<&[f64]>,
    cell_budget: usize,
    k: usize,
) -> Vec<(Vec<usize>, f64)> {
    if k == 0 {
        return Vec::new();
    }
    let prepared = MatrixProblem::new(cluster_ids, transition, start, end);

    // Force the tour to begin in cluster `c` by leaving only that cluster's
    // start edges finite. One independent solve per starting cluster.
    let solve_from = |c: usize| -> Option<(Vec<usize>, f64)> {
        let start: Vec<f64> = prepared
            .cluster_ids
            .iter()
            .zip(&prepared.start)
            .map(|(&ci, &s)| if ci == c { s } else { f64::INFINITY })
            .collect();
        let solution = agtsp::solve(&prepared.problem(&start), cell_budget)?;
        Some((solution.order, solution.cost))
    };

    let mut tours: Vec<(Vec<usize>, f64)> = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            (0..prepared.n_clusters)
                .into_par_iter()
                .filter_map(solve_from)
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            (0..prepared.n_clusters).filter_map(solve_from).collect()
        }
    };

    tours.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    tours.truncate(k);
    tours
}

/// Owned, validated inputs for the matrix-level AGTSP entry points. Holds the
/// densified cluster labels and the flattened row-major matrix / start / end
/// vectors so a borrowing [`Problem`] can be built (once per solve) against them.
struct MatrixProblem {
    cluster_ids: Vec<usize>,
    n_clusters: usize,
    options: usize,
    transition: Vec<f64>,
    start: Vec<f64>,
    end: Vec<f64>,
}

impl MatrixProblem {
    fn new(
        cluster_ids: &[usize],
        transition: &[Vec<f64>],
        start: Option<&[f64]>,
        end: Option<&[f64]>,
    ) -> Self {
        let options = cluster_ids.len();
        debug_assert_eq!(
            transition.len(),
            options,
            "transition must be square with side cluster_ids.len()"
        );
        debug_assert!(
            transition.iter().all(|row| row.len() == options),
            "transition rows must each have length cluster_ids.len()"
        );
        let (dense, n_clusters) = densify_clusters(cluster_ids);
        let flat = transition
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let resolve = |v: Option<&[f64]>| match v {
            Some(s) => {
                debug_assert_eq!(s.len(), options, "start/end length must equal option count");
                s.to_vec()
            }
            None => vec![0.0; options],
        };
        MatrixProblem {
            cluster_ids: dense,
            n_clusters,
            options,
            transition: flat,
            start: resolve(start),
            end: resolve(end),
        }
    }

    fn problem<'a>(&'a self, start: &'a [f64]) -> Problem<'a> {
        Problem {
            cluster_ids: &self.cluster_ids,
            n_clusters: self.n_clusters,
            transition: &self.transition,
            options: self.options,
            start,
            end: &self.end,
        }
    }
}

/// Remap arbitrary cluster labels onto a dense `0..k` range (in order of first
/// appearance), returning the remapped labels and `k`. The Held–Karp bitmask
/// indexes clusters by bit position, so labels must be dense and zero-based.
fn densify_clusters(cluster_ids: &[usize]) -> (Vec<usize>, usize) {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let dense = cluster_ids
        .iter()
        .map(|&c| {
            let next = map.len();
            *map.entry(c).or_insert(next)
        })
        .collect();
    (dense, map.len())
}

fn run<const N: usize, Cost, Conn>(
    req_paths: &[ReqPath<N>],
    cost: &Cost,
    settings: &MultiPathSettings<N>,
    connect: Conn,
) -> MultipathResult<Vec<SRobotPath<N, f64>>>
where
    Cost: Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> f64,
    Conn: Fn(&SRobotQ<N, f64>, &SRobotQ<N, f64>) -> MultipathResult<SRobotPath<N, f64>> + MaybeSync,
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
