use deke_types::{DekeError, Planner, Retimer, SRobotPath, SRobotQ, Validator};

use crate::reqpath::DirectedOption;

/// Weighted joint-space distance `sqrt(Σ (wᵢ·(aᵢ-bᵢ))²)`. Note this is *not*
/// `SRobotQ::distance`, which is unweighted.
pub fn weighted_distance<const N: usize>(
    a: &SRobotQ<N, f64>,
    b: &SRobotQ<N, f64>,
    w: &SRobotQ<N, f64>,
) -> f64 {
    w.0.iter()
        .zip(a.0.iter())
        .zip(b.0.iter())
        .map(|((&wi, &ai), &bi)| {
            let d = wi * (ai - bi);
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Cost function scoring a transition by the weighted joint-space straight-line
/// distance between the two configurations — the cheap, planner-free metric.
///
/// ```
/// # use deke_multipath::weighted_euclidean;
/// # use deke_types::SRobotQ;
/// let cost = weighted_euclidean(SRobotQ([1.0, 1.0, 0.5]));
/// let d = cost(SRobotQ([0.0; 3]), SRobotQ([1.0, 0.0, 2.0]));
/// ```
pub fn weighted_euclidean<const N: usize>(
    weights: SRobotQ<N, f64>,
) -> impl Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> f64 {
    move |from, to| weighted_distance(&from, &to, &weights)
}

/// Cost function that plans an obstacle-aware connector with `planner` and
/// scores the transition by the arc length of the resulting path. Use this when
/// ordering should account for how far the robot actually has to travel to route
/// around obstacles, not just the straight-line joint distance.
///
/// A transition the planner cannot connect scores `f64::INFINITY`; the ordering
/// then treats that edge as infeasible and routes around it. Each scored
/// transition costs one plan, so this is `O(options²)` planner calls — expensive
/// but exact about reachability.
///
/// `make_waypoints` bridges the planner's associated `Waypoints` type, exactly as
/// for [`crate::TransitionPlanner`] — for the RRT planners that is
/// `|s, e| StartEnd { start: s, end: e }`.
pub fn planned_path_length<'a, 'ctx, const N: usize, P, V, MW>(
    planner: &'a P,
    config: &'a P::Config,
    make_waypoints: MW,
    validator: &'a V,
    ctx: &'a V::Context<'ctx>,
) -> impl Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> f64 + 'a
where
    'ctx: 'a,
    P: Planner<N, f64>,
    V: Validator<N, (), f64>,
    MW: Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> P::Waypoints + 'a,
{
    move |from, to| {
        let waypoints = make_waypoints(from, to);
        match planner
            .plan::<DekeError, _>(config, &waypoints, validator, ctx)
            .0
        {
            Ok(path) => path.arc_length(),
            Err(_) => f64::INFINITY,
        }
    }
}

/// Cost function that plans a connector and then retimes it, scoring the
/// transition by the resulting trajectory duration in seconds. This is the
/// metric to minimise when total cycle time is what matters: it folds in the
/// dynamics (velocity/acceleration limits) the retimer enforces, not just
/// geometric distance.
///
/// A transition that cannot be planned or retimed scores `f64::INFINITY`, so the
/// ordering routes around it. Each scored transition costs one plan plus one
/// retime.
pub fn planned_trajectory_time<'a, 'ctx, const N: usize, P, R, V, MW>(
    planner: &'a P,
    config: &'a P::Config,
    retimer: &'a R,
    constraints: &'a R::Constraints,
    make_waypoints: MW,
    validator: &'a V,
    ctx: &'a V::Context<'ctx>,
) -> impl Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> f64 + 'a
where
    'ctx: 'a,
    P: Planner<N, f64>,
    R: Retimer<N, f64>,
    V: Validator<N, (), f64>,
    MW: Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> P::Waypoints + 'a,
{
    move |from, to| {
        let waypoints = make_waypoints(from, to);
        let path = match planner
            .plan::<DekeError, _>(config, &waypoints, validator, ctx)
            .0
        {
            Ok(path) => path,
            Err(_) => return f64::INFINITY,
        };
        match retimer.retime(constraints, &path, validator, ctx).0 {
            Ok(traj) => traj.duration().as_secs_f64(),
            Err(_) => f64::INFINITY,
        }
    }
}

/// Cost of traversing a path under `cost`: the sum over its segments. For the
/// weighted-joint model this is the weighted arc length.
fn traversal_cost<const N: usize, C>(path: &SRobotPath<N, f64>, cost: &C) -> f64
where
    C: Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> f64,
{
    path.segments().map(|(a, b)| cost(*a, *b)).sum()
}

/// The cost matrices the AGTSP consumes. `transition` is a row-major
/// `options × options` matrix — contiguous rather than a `Vec<Vec<_>>` so the
/// Held–Karp inner loop, which is cache-bound on these reads, avoids a pointer
/// chase per access. Entry `(i, j)` already folds in option `j`'s own traversal
/// cost, so the solver prefers the cheaper realization, not just the cheaper
/// connector.
pub(crate) struct CostMatrices {
    /// Row-major `options × options`: `(i, j)` = move from option `i`'s end to
    /// option `j`'s start, then traverse `j`.
    pub transition: Vec<f64>,
    /// `start[i]` = move from the global start to option `i`'s start, then
    /// traverse `i`.
    pub start: Vec<f64>,
    /// `end[i]` = move from option `i`'s end to the global end (`0.0` when no
    /// end is requested).
    pub end: Vec<f64>,
}

pub(crate) fn build_matrices<const N: usize, C>(
    options: &[DirectedOption<N>],
    cost: &C,
    start_q: &SRobotQ<N, f64>,
    end_q: Option<&SRobotQ<N, f64>>,
) -> CostMatrices
where
    C: Fn(SRobotQ<N, f64>, SRobotQ<N, f64>) -> f64,
{
    let m = options.len();
    let traversal: Vec<f64> = options
        .iter()
        .map(|o| traversal_cost(&o.path, cost))
        .collect();

    let mut transition = vec![0.0_f64; m * m];
    for (i, oi) in options.iter().enumerate() {
        let from = *oi.path.last();
        let row = &mut transition[i * m..(i + 1) * m];
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = cost(from, *options[j].path.first()) + traversal[j];
        }
    }

    let start = options
        .iter()
        .enumerate()
        .map(|(i, o)| cost(*start_q, *o.path.first()) + traversal[i])
        .collect();

    let end = options
        .iter()
        .map(|o| end_q.map_or(0.0, |e| cost(*o.path.last(), *e)))
        .collect();

    CostMatrices {
        transition,
        start,
        end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q<const N: usize>(a: [f64; N]) -> SRobotQ<N, f64> {
        SRobotQ(a)
    }

    #[test]
    fn weighted_distance_applies_weights() {
        let origin = q([0.0, 0.0]);
        let w = q([2.0, 1.0]);
        // Moving the high-weight joint costs twice the low-weight joint.
        assert!((weighted_distance(&origin, &q([1.0, 0.0]), &w) - 2.0).abs() < 1e-12);
        assert!((weighted_distance(&origin, &q([0.0, 1.0]), &w) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn weighted_euclidean_matches_distance() {
        let w = q([2.0, 1.0]);
        let cost = weighted_euclidean(w);
        assert!((cost(q([0.0, 0.0]), q([1.0, 0.0])) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn traversal_sums_weighted_segments() {
        let path = SRobotPath::<2, f64>::try_new(vec![q([0.0, 0.0]), q([1.0, 0.0]), q([1.0, 1.0])])
            .unwrap();
        let cost = weighted_euclidean(q([2.0, 1.0]));
        // 2·1 (joint 0) + 1·1 (joint 1) = 3.
        assert!((traversal_cost(&path, &cost) - 3.0).abs() < 1e-12);
    }
}
