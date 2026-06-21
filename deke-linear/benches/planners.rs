use std::time::Duration;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use deke_kin::{DHJoint, JointLimits as KinJointLimits, Kinematics};
use deke_linear::{
    condition, CartesianLinearPlanner, CartesianRun, FollowConfig, JointLimits, PathConditioning,
    PlannerOptions, RedundantLinearPlanner, RedundantOptions,
};
use deke_types::glam::{DAffine3, DVec3};
use deke_types::{FKChain, IkSolver, SRobotQ};

fn ur() -> Kinematics<6, f64> {
    use std::f64::consts::PI;
    let alpha = [PI / 2.0, 0.0, 0.0, PI / 2.0, -PI / 2.0, 0.0];
    let a = [0.0, -0.612, -0.573, 0.0, 0.0, 0.0];
    let d = [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922];
    Kinematics::from_dh(
        std::array::from_fn(|i| DHJoint { a: a[i], alpha: alpha[i], d: d[i], theta_offset: 0.0 }),
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

fn planner_opts() -> PlannerOptions<6> {
    FollowConfig::weld(35.0, JointLimits::symmetric(2.0, 8.0, 80.0), Duration::from_millis(8)).planner
}

fn bench(c: &mut Criterion) {
    let robot = ur();
    let opts = planner_opts(); // weld preset: sample_ds 0.5 mm, velocity reconfig on
    let ropts = RedundantOptions::default();

    let fixed = CartesianLinearPlanner::new(&robot);
    let red = RedundantLinearPlanner::new(&robot);
    let noop = deke_linear::NoopValidator::<6>;

    // Attribute the per-node cost between IK and the Jacobian (both deke-kin).
    {
        use deke_types::ContinuousFKChain;
        let q = anchor();
        let pose = robot.fk_end(&q).unwrap();
        c.bench_function("ik_call", |b| b.iter(|| black_box(robot.ik(black_box(pose)))));
        c.bench_function("jacobian_call", |b| {
            b.iter(|| black_box(robot.jacobian(black_box(&q))))
        });
    }

    for &len in &[0.04, 0.10] {
        let run = straight_run(&robot, len);
        let cm = (len * 100.0) as usize;
        c.bench_function(&format!("fixed_plan_{cm}cm"), |b| {
            b.iter(|| black_box(fixed.plan_run(black_box(&run), &opts, &noop, &(), None, 0)).is_ok())
        });
        c.bench_function(&format!("redundant_plan_{cm}cm"), |b| {
            b.iter(|| {
                black_box(red.plan_run(black_box(&run), &opts, &ropts, &noop, &(), None, 0)).is_ok()
            })
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
