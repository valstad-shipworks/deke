mod common;

use std::time::Duration;

use deke_linear::{
    ConstantSpeedRetimer, JointLimits, LinearConstraints, NoopValidator, PathConditioning,
    PlannerOptions, RailAxis, RailMountedChain, RailOptions, RailRefine, RailYawConfig,
    RailYawPlanner, RedundantAxis, RedundantOptions, TcpLimits, condition,
};
use deke_types::glam::DVec3;
use deke_types::{DekeError, Planner, Retimer, SRobotPath, SRobotQ, SRobotTraj};

const RAIL_V: f64 = 1.0;
const RAIL_A: f64 = 20.0;
const RAIL_J: f64 = 2000.0;
const ARM_V: f64 = 2.0;
const ARM_A: f64 = 8.0;
const ARM_J: f64 = 80.0;
const EPS: f64 = 1.0 + 1e-9;

fn limits7() -> JointLimits<7> {
    JointLimits {
        v_max: SRobotQ::from_array([RAIL_V, ARM_V, ARM_V, ARM_V, ARM_V, ARM_V, ARM_V]),
        a_max: SRobotQ::from_array([RAIL_A, ARM_A, ARM_A, ARM_A, ARM_A, ARM_A, ARM_A]),
        j_max: SRobotQ::from_array([RAIL_J, ARM_J, ARM_J, ARM_J, ARM_J, ARM_J, ARM_J]),
    }
}

fn fd_within_limits(traj: &SRobotTraj<7, f64>, lim: &JointLimits<7>) {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let mut v = [0.0f64; 7];
    let mut a = [0.0f64; 7];
    let mut j = [0.0f64; 7];
    for i in 0..p.len().saturating_sub(1) {
        for (k, vk) in v.iter_mut().enumerate() {
            *vk = vk.max(((p[i + 1].0[k] - p[i].0[k]) / dt).abs());
        }
    }
    for i in 1..p.len().saturating_sub(1) {
        for (k, ak) in a.iter_mut().enumerate() {
            *ak = ak.max(((p[i + 1].0[k] - 2.0 * p[i].0[k] + p[i - 1].0[k]) / (dt * dt)).abs());
        }
    }
    for i in 3..p.len() {
        for (k, jk) in j.iter_mut().enumerate() {
            let d = (p[i].0[k] - 3.0 * p[i - 1].0[k] + 3.0 * p[i - 2].0[k] - p[i - 3].0[k])
                / (dt * dt * dt);
            *jk = jk.max(d.abs());
        }
    }
    for k in 0..7 {
        assert!(
            v[k] <= lim.v_max.0[k] * EPS,
            "joint {k} velocity {} > {}",
            v[k],
            lim.v_max.0[k]
        );
        assert!(
            a[k] <= lim.a_max.0[k] * EPS,
            "joint {k} accel {} > {}",
            a[k],
            lim.a_max.0[k]
        );
        assert!(
            j[k] <= lim.j_max.0[k] * EPS,
            "joint {k} jerk {} > {}",
            j[k],
            lim.j_max.0[k]
        );
    }
}

#[test]
fn rail_composes_with_yaw_planner() {
    let arm = common::ur();
    let tcp = 30.0 * 0.0254 / 60.0;
    let lim = limits7();

    let poses = common::straight(&arm, DVec3::X, 0.30, 24);
    let runs = condition(&poses, &PathConditioning::default()).unwrap();

    let cfg = RailYawConfig::<6, 7> {
        planner: PlannerOptions {
            sample_ds: 5e-4,
            manip_weight: 1.0,
            max_branch_jump: 0.6,
            max_velocity: tcp,
            joint_v_max: lim.v_max,
            reconfig_vel_fraction: 0.9,
        },
        rail: RailOptions {
            axis: RailAxis::PosX,
            window: (-0.3, 0.3),
            samples: 21,
            dp_ds: 5e-3,
            rate_weight: 0.5,
            max_step: 0.05,
            centering_weight: 0.05,
            refine: RailRefine::Pchip,
        },
        yaw: RedundantOptions {
            axis: RedundantAxis::PosZ,
            yaw_window: (-45.0_f64.to_radians(), 45.0_f64.to_radians()),
            yaw_samples: 9,
            dp_ds: 5e-3,
            yaw_rate_weight: 0.2,
            max_yaw_step: 0.6,
        },
    };

    let chain = RailMountedChain::<6, 7, _>::new(&arm, cfg.rail.axis);
    let planner = RailYawPlanner::<6, 7, _>::new(&arm);
    let retimer = ConstantSpeedRetimer::new(&chain);
    let constraints = LinearConstraints {
        joint: limits7(),
        tcp: TcpLimits::speed(tcp),
        output_dt: Duration::from_millis(8),
        forbid_interior_dips: false,
        corner_smoothing: Some(0.01),
    };
    let validator = NoopValidator::<7>;

    let mut all: Vec<SRobotQ<7, f64>> = Vec::new();
    for run in runs.iter() {
        let (path, _diag) = planner.plan::<DekeError, _>(&cfg, run, &validator, &());
        let path = path.expect("rail+yaw plan");
        let (traj, _rd) = retimer.retime(&constraints, &path, &validator, &());
        let traj = traj.expect("rail+yaw retime");
        let samples = traj.path().iter().copied();
        if all.is_empty() {
            all.extend(samples);
        } else {
            all.extend(samples.skip(1));
        }
    }

    let path = SRobotPath::try_new(all).unwrap();
    let traj = SRobotTraj::new(Duration::from_millis(8), path);
    fd_within_limits(&traj, &lim);

    for q in traj.path().iter() {
        assert!(
            q.0[0] >= -0.3 - 1e-9 && q.0[0] <= 0.3 + 1e-9,
            "rail outside window: {}",
            q.0[0]
        );
    }
}
