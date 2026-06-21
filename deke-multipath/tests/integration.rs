use deke_multipath::{
    MultiPathSettings, ReqPath, TransitionCost, TransitionPlanner, plan_multipath,
};
use deke_rrt::{AorrtcPlanner, AorrtcSettings, StartEnd};
use deke_types::{DekeError, DekeResult, SRobotPath, SRobotQ, SRobotQLike, Validator};

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
    let cost = TransitionCost::JointWeighted(q([1.0, 1.0]));
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
