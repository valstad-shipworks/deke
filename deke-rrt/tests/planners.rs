use deke_rrt::scurve::{JointKinLimits, KinematicLimits};
use deke_rrt::{
    AorrtcPlanner, AorrtcSettings, KrrtcPlanner, KrrtcSettings, RrtTermination, RrtcPlanner,
    RrtcSettings, StartEnd,
};
use deke_types::{DekeError, DekeResult, Planner, SRobotQ, SRobotQLike, Validator};

#[derive(Debug, Clone)]
struct FreeSpace<const N: usize>;

impl<const N: usize> Validator<N, (), f64> for FreeSpace<N> {
    type Context<'ctx> = ();

    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<N, E, f64>>(
        &self,
        _q: A,
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        Ok(())
    }

    fn validate_motion<'ctx>(
        &self,
        _qs: &[SRobotQ<N, f64>],
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        Ok(())
    }
}

/// Rejects any configuration inside an axis-aligned box, so the straight
/// start→goal edge is blocked and the planner must route around it.
#[derive(Debug, Clone)]
struct BoxObstacle<const N: usize> {
    lo: [f64; N],
    hi: [f64; N],
}

impl<const N: usize> BoxObstacle<N> {
    fn inside(&self, q: &SRobotQ<N, f64>) -> bool {
        (0..N).all(|i| q.0[i] >= self.lo[i] && q.0[i] <= self.hi[i])
    }

    /// Inside the box shrunk by `margin` on every face. The planner only checks
    /// collisions at its discrete `resolution`, so a path may clip a corner by
    /// up to ~`resolution` between samples; an independent check must tolerate
    /// that to test "does the path substantially cut through the obstacle"
    /// rather than re-deriving the discretization error.
    fn inside_by(&self, q: &SRobotQ<N, f64>, margin: f64) -> bool {
        (0..N).all(|i| q.0[i] >= self.lo[i] + margin && q.0[i] <= self.hi[i] - margin)
    }
}

impl<const N: usize> Validator<N, (), f64> for BoxObstacle<N> {
    type Context<'ctx> = ();

    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<N, E, f64>>(
        &self,
        q: A,
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        let q = q.to_srobotq().map_err(Into::into)?;
        if self.inside(&q) {
            Err(DekeError::EnvironmentCollision(0, 0))
        } else {
            Ok(())
        }
    }

    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N, f64>],
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        for q in qs {
            if self.inside(q) {
                return Err(DekeError::EnvironmentCollision(0, 0));
            }
        }
        Ok(())
    }
}

fn assert_path_clear<const N: usize>(
    path: &[SRobotQ<N, f64>],
    start: SRobotQ<N, f64>,
    goal: SRobotQ<N, f64>,
    obs: &BoxObstacle<N>,
) {
    assert!(path.len() >= 2, "path needs at least two waypoints");
    assert!(start.distance(&path[0]) < 1e-6, "path must start at start");
    assert!(
        goal.distance(&path[path.len() - 1]) < 1e-6,
        "path must end at goal"
    );
    // Densely re-sample every edge and confirm none of it cuts substantially
    // through the box. The margin matches the planners' default collision-check
    // resolution (0.05); deeper penetration would mean the planner returned a
    // genuinely colliding path.
    for w in path.windows(2) {
        let steps = 64;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let p = w[0] + (w[1] - w[0]) * t;
            assert!(
                !obs.inside_by(&p, 0.05),
                "path cuts through the obstacle at {p:?}"
            );
        }
    }
}

/// A wall spanning x∈[0.4,0.6], y∈[-0.5,0.8] within bounds [-0.5,1.5]². The
/// straight start→goal edge is blocked; a corridor exists above y=0.8.
fn wall() -> BoxObstacle<2> {
    BoxObstacle {
        lo: [0.4, -0.5],
        hi: [0.6, 0.8],
    }
}

const START: SRobotQ<2, f64> = SRobotQ([0.0, 0.0]);
const GOAL: SRobotQ<2, f64> = SRobotQ([1.0, 1.0]);
const LOWER: SRobotQ<2, f64> = SRobotQ([-0.5, -0.5]);
const UPPER: SRobotQ<2, f64> = SRobotQ([1.5, 1.5]);

#[test]
fn rrtc_routes_around_obstacle() {
    let obs = wall();
    let cfg = RrtcSettings::new(LOWER, UPPER);
    let wp = StartEnd::new(START, GOAL).unwrap();
    let (path, diag) = RrtcPlanner::<2>::new().plan::<DekeError, _>(&cfg, &wp, &obs, &());
    let path = path.expect("RRTC should find a path around the wall");
    assert_eq!(diag.termination, RrtTermination::Solved);
    let pts: Vec<_> = path.iter().copied().collect();
    assert_path_clear(&pts, START, GOAL, &obs);
}

#[test]
fn aorrtc_routes_around_obstacle_and_reports_anytime() {
    let obs = wall();
    let mut cfg = AorrtcSettings::new_normalized(LOWER, UPPER);
    cfg.max_iterations = 5_000;
    cfg.stall_iterations = 1_000;
    let wp = StartEnd::new(START, GOAL).unwrap();
    let (path, diag) = AorrtcPlanner::<2>::new().plan::<DekeError, _>(&cfg, &wp, &obs, &());
    let path = path.expect("AORRTC should find a path around the wall");
    let pts: Vec<_> = path.iter().copied().collect();
    assert_path_clear(&pts, START, GOAL, &obs);
    assert!(diag.anytime.is_some(), "AORRTC must report anytime info");
}

#[test]
fn krrtc_solves_free_space() {
    let limits = KinematicLimits {
        joints: [
            JointKinLimits {
                v_max: 1.0,
                a_max: 5.0,
                j_max: 25.0,
            },
            JointKinLimits {
                v_max: 2.0,
                a_max: 8.0,
                j_max: 40.0,
            },
        ],
    };
    let cfg = KrrtcSettings::new(LOWER, UPPER, limits);
    let wp = StartEnd::new(START, GOAL).unwrap();
    let (path, _diag) =
        KrrtcPlanner::<2>::new().plan::<DekeError, _>(&cfg, &wp, &FreeSpace::<2>, &());
    assert!(path.is_ok(), "KRRTC should connect through free space");
}

#[test]
fn free_space_returns_direct_connection() {
    let cfg = RrtcSettings::new(LOWER, UPPER);
    let wp = StartEnd::new(START, GOAL).unwrap();
    let (path, diag) =
        RrtcPlanner::<2>::new().plan::<DekeError, _>(&cfg, &wp, &FreeSpace::<2>, &());
    assert!(path.is_ok());
    assert_eq!(diag.termination, RrtTermination::DirectConnection);
}

#[test]
fn invalid_settings_rejected_before_planning() {
    let mut cfg = RrtcSettings::new(LOWER, UPPER);
    cfg.resolution = 0.0;
    let wp = StartEnd::new(START, GOAL).unwrap();
    let (path, diag) =
        RrtcPlanner::<2>::new().plan::<DekeError, _>(&cfg, &wp, &FreeSpace::<2>, &());
    assert!(path.is_err());
    assert_eq!(diag.termination, RrtTermination::InputInvalid);
}

#[test]
fn zero_rail_weight_rejected() {
    let mut cfg = RrtcSettings::new(LOWER, UPPER);
    cfg.dof_cost_weights = SRobotQ([0.0, 1.0]);
    let wp = StartEnd::new(START, GOAL).unwrap();
    let (path, diag) =
        RrtcPlanner::<2>::new().plan::<DekeError, _>(&cfg, &wp, &FreeSpace::<2>, &());
    assert!(path.is_err());
    assert_eq!(diag.termination, RrtTermination::InputInvalid);
}
