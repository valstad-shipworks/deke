use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use deke_kin::{DHJoint, JointLimits as KinJointLimits, Kinematics};
use deke_linear::{
    CartesianLinearPlanner, CartesianRun, ConstantSpeedRetimer, JointLimits, LinearConstraints,
    NoopValidator, PathConditioning, PlannerOptions, condition,
};
use deke_types::glam::{DAffine3, DVec3};
use deke_types::{DekeError, FKChain, Planner, Retimer, SRobotPath, SRobotQ};

fn ur() -> Kinematics<6, f64> {
    use std::f64::consts::PI;
    let alpha = [PI / 2.0, 0.0, 0.0, PI / 2.0, -PI / 2.0, 0.0];
    let a = [0.0, -0.612, -0.573, 0.0, 0.0, 0.0];
    let d = [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922];
    Kinematics::from_dh(
        std::array::from_fn(|i| DHJoint {
            a: a[i],
            alpha: alpha[i],
            d: d[i],
            theta_offset: 0.0,
        }),
        KinJointLimits::symmetric(2.0 * PI),
        &[],
    )
}

fn anchor() -> SRobotQ<6, f64> {
    SRobotQ::from_array([0.2, -1.0, 1.2, -1.3, -std::f64::consts::FRAC_PI_2, 0.3])
}

fn straight_run(robot: &Kinematics<6, f64>, len: f64) -> CartesianRun {
    let base = robot.fk_end(&anchor()).unwrap();
    let poses: Vec<DAffine3> = (0..6)
        .map(|i| {
            let f = i as f64 / 5.0;
            DAffine3::from_mat3_translation(base.matrix3, base.translation + DVec3::X * (f * len))
        })
        .collect();
    condition(&poses, &PathConditioning::default())
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
}

fn opts() -> PlannerOptions<6> {
    let tcp_speed = 35.0 * 0.0254 / 60.0;
    PlannerOptions {
        sample_ds: 5e-4,
        manip_weight: 1.0,
        max_branch_jump: 0.6,
        max_velocity: tcp_speed,
        joint_v_max: SRobotQ::splat(2.0),
        reconfig_vel_fraction: 0.9,
    }
}

fn constraints() -> LinearConstraints<6> {
    LinearConstraints {
        joint: JointLimits::symmetric(2.0, 8.0, 80.0),
        tcp_speed: 35.0 * 0.0254 / 60.0,
        output_dt: Duration::from_millis(8),
        forbid_interior_dips: false,
    }
}

fn joint_path(robot: &Kinematics<6, f64>, len: f64) -> SRobotPath<6, f64> {
    let run = straight_run(robot, len);
    let planner = CartesianLinearPlanner::new(robot);
    let noop = NoopValidator::<6>;
    planner
        .plan::<DekeError, _>(&opts(), &run, &noop, &())
        .0
        .unwrap()
}

fn bench(c: &mut Criterion) {
    let robot = ur();
    let cons = constraints();
    let retimer = ConstantSpeedRetimer::new(&robot);
    let noop = NoopValidator::<6>;

    for &len in &[0.04, 0.10, 0.30] {
        let path = joint_path(&robot, len);
        let cm = (len * 100.0) as usize;
        c.bench_function(&format!("retime_{cm}cm"), |b| {
            b.iter(|| black_box(retimer.retime(&cons, black_box(&path), &noop, &()).0).is_ok())
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
