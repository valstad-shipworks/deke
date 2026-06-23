use std::time::Duration;

use deke_multipath::{
    MultiPathSettings, ReqPath, TransitionPlanner, plan_multipath, planned_path_length,
    planned_trajectory_time, solve_matrix, solve_matrix_multi_start, weighted_euclidean,
};
use deke_rrt::{AorrtcPlanner, AorrtcSettings, StartEnd};
use deke_types::{
    DekeError, DekeResult, Retimer, SRobotPath, SRobotQ, SRobotQLike, SRobotTraj, Validator,
};

/// A validator that accepts everything — the planner only has to connect
/// configs in free joint space, which exercises the stitching end to end
/// without pulling in collision geometry.
#[derive(Clone, Debug)]
struct AllowAll;

impl Validator<2, (), f64> for AllowAll {
    type Context<'ctx> = ();

    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<2, E, f64>>(
        &self,
        _q: A,
        _ctx: &(),
    ) -> DekeResult<()> {
        Ok(())
    }

    fn validate_motion<'ctx>(&self, _qs: &[SRobotQ<2, f64>], _ctx: &()) -> DekeResult<()> {
        Ok(())
    }
}

/// A retimer that stamps the path with a fixed per-waypoint step, so its
/// trajectory duration grows with the waypoint count — enough to drive
/// [`planned_trajectory_time`] end to end without a real dynamics solver.
struct UnitRetimer;

impl Retimer<2, f64> for UnitRetimer {
    type Diagnostic = String;
    type Constraints = ();

    fn retime<V: Validator<2, (), f64>>(
        &self,
        _constraints: &(),
        path: &SRobotPath<2, f64>,
        _validator: &V,
        _ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<2, f64>>, String) {
        (
            Ok(SRobotTraj::new(Duration::from_millis(100), path.clone())),
            String::new(),
        )
    }
}

fn q(a: [f64; 2]) -> SRobotQ<2, f64> {
    SRobotQ(a)
}

fn path(points: &[[f64; 2]]) -> SRobotPath<2, f64> {
    SRobotPath::try_new(points.iter().map(|p| q(*p)).collect()).unwrap()
}

#[test]
fn stitches_full_plan_with_rrt() {
    let planner = AorrtcPlanner::<2>::new();
    let config = AorrtcSettings::<2>::new(q([-3.5, -3.5]), q([3.5, 3.5]));
    let validator = AllowAll;
    let ctx = ();

    let req = vec![
        ReqPath::OneWay(path(&[[1.0, 1.0], [1.5, 1.0]])),
        ReqPath::Reversible(path(&[[-1.0, -1.0], [-1.5, -1.0]])),
        ReqPath::OneWay(path(&[[2.0, -2.0], [2.0, -2.5]])),
    ];

    let start = q([0.0, 0.0]);
    let settings = MultiPathSettings::new(start).with_end(start);
    let cost = weighted_euclidean(q([1.0, 1.0]));
    let transition = TransitionPlanner {
        planner: &planner,
        config: &config,
        make_waypoints: |s, e| StartEnd { start: s, end: e },
    };

    let out = plan_multipath(&req, &cost, &settings, &transition, &validator, &ctx).unwrap();

    // At least one segment per required path, plus connectors.
    assert!(out.len() >= req.len());

    // The plan is contiguous: each segment ends where the next begins.
    for w in out.windows(2) {
        assert!(
            w[0].last().distance(w[1].first()) < 1e-6,
            "discontinuity: {:?} -> {:?}",
            w[0].last().0,
            w[1].first().0,
        );
    }

    // It starts and ends at the requested configuration.
    assert!(out.first().unwrap().first().distance(&start) < 1e-6);
    assert!(out.last().unwrap().last().distance(&start) < 1e-6);

    // Every required path's signature endpoint appears exactly once (in either
    // orientation for the reversible one).
    let signatures = [
        ([1.0, 1.0], [1.5, 1.0]),
        ([-1.0, -1.0], [-1.5, -1.0]),
        ([2.0, -2.0], [2.0, -2.5]),
    ];
    for (a, b) in signatures {
        let hits = out
            .iter()
            .filter(|p| {
                let (f, l) = (p.first().0, p.last().0);
                (f == a && l == b) || (f == b && l == a)
            })
            .count();
        assert_eq!(hits, 1, "required path {a:?}->{b:?} appears {hits} times");
    }
}

#[test]
fn solve_matrix_picks_best_option_per_cluster() {
    // Cluster 0 = {opt0, opt1}, cluster 1 = {opt2}. opt1 is the cheap
    // realization of cluster 0 and connects cheaply to opt2.
    let cluster_ids = [0, 0, 1];
    let inf = f64::INFINITY;
    let transition = vec![
        vec![0.0, inf, 9.0],
        vec![inf, 0.0, 1.0],
        vec![9.0, 1.0, 0.0],
    ];
    let start = [5.0, 1.0, 5.0];
    let end = [0.0, 0.0, 0.0];
    let (order, cost) = solve_matrix(
        &cluster_ids,
        &transition,
        Some(&start),
        Some(&end),
        deke_multipath::DEFAULT_CELL_BUDGET,
    )
    .unwrap();
    assert_eq!(order, vec![1, 2]);
    assert!((cost - 2.0).abs() < 1e-9);
}

#[test]
fn solve_matrix_densifies_non_contiguous_clusters() {
    // Same instance but with non-zero-based, non-contiguous cluster labels.
    let cluster_ids = [7, 7, 42];
    let inf = f64::INFINITY;
    let transition = vec![
        vec![0.0, inf, 9.0],
        vec![inf, 0.0, 1.0],
        vec![9.0, 1.0, 0.0],
    ];
    let start = [5.0, 1.0, 5.0];
    let (order, cost) = solve_matrix(
        &cluster_ids,
        &transition,
        Some(&start),
        None,
        deke_multipath::DEFAULT_CELL_BUDGET,
    )
    .unwrap();
    assert_eq!(order, vec![1, 2]);
    assert!((cost - 2.0).abs() < 1e-9);
}

#[test]
fn solve_matrix_multi_start_returns_sorted_top_k() {
    // Three singleton clusters in a line: the tour cost depends on where it
    // starts, so one solve per starting cluster yields distinct tours.
    let cluster_ids = [0, 1, 2];
    let transition = vec![
        vec![0.0, 1.0, 2.0],
        vec![1.0, 0.0, 1.0],
        vec![2.0, 1.0, 0.0],
    ];
    let tours = solve_matrix_multi_start(
        &cluster_ids,
        &transition,
        None,
        None,
        deke_multipath::DEFAULT_CELL_BUDGET,
        2,
    );
    assert_eq!(tours.len(), 2);
    // Each returned tour begins at a distinct cluster and is ordered by cost.
    assert!(tours[0].1 <= tours[1].1);
    let starts: Vec<usize> = tours
        .iter()
        .map(|(order, _)| cluster_ids[order[0]])
        .collect();
    assert_ne!(starts[0], starts[1]);
}

#[test]
fn planner_backed_cost_helpers_score_a_transition() {
    let planner = AorrtcPlanner::<2>::new();
    let config = AorrtcSettings::<2>::new(q([-3.5, -3.5]), q([3.5, 3.5]));
    let validator = AllowAll;
    let ctx = ();

    let length = planned_path_length(
        &planner,
        &config,
        |s, e| StartEnd { start: s, end: e },
        &validator,
        &ctx,
    );
    let by_length = length(q([0.0, 0.0]), q([1.0, 1.0]));
    assert!(by_length.is_finite() && by_length > 0.0);

    let retimer = UnitRetimer;
    let time = planned_trajectory_time(
        &planner,
        &config,
        &retimer,
        &(),
        |s, e| StartEnd { start: s, end: e },
        &validator,
        &ctx,
    );
    let by_time = time(q([0.0, 0.0]), q([1.0, 1.0]));
    assert!(by_time.is_finite() && by_time > 0.0);
}
