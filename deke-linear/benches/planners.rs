use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use deke_kin::{DHJoint, JointLimits as KinJointLimits, Kinematics};
use deke_linear::{
    CartesianLinearPlanner, CartesianRun, PathConditioning, PlannerOptions, RedundantConfig,
    RedundantLinearPlanner, RedundantOptions, condition,
};
use deke_types::glam::{DAffine3, DVec3};
use deke_types::{DekeError, FKChain, IkSolver, Planner, SRobotQ};

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

fn planner_opts() -> PlannerOptions<6> {
    // weld preset at 35 IPM: fine sampling, velocity reconfig test on.
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
        c.bench_function("ik_call", |b| {
            b.iter(|| black_box(robot.ik(black_box(pose))))
        });
        c.bench_function("jacobian_call", |b| {
            b.iter(|| black_box(robot.jacobian(black_box(&q))))
        });
        c.bench_function("manipulability_call", |b| {
            b.iter(|| black_box(robot.manipulability(black_box(&q))))
        });
    }

    for &len in &[0.04, 0.10] {
        let run = straight_run(&robot, len);
        let cm = (len * 100.0) as usize;
        c.bench_function(&format!("fixed_plan_{cm}cm"), |b| {
            b.iter(|| {
                black_box(
                    fixed
                        .plan::<DekeError, _>(&opts, black_box(&run), &noop, &())
                        .0,
                )
                .is_ok()
            })
        });
        let rcfg = RedundantConfig {
            planner: opts.clone(),
            redundant: ropts.clone(),
        };
        c.bench_function(&format!("redundant_plan_{cm}cm"), |b| {
            b.iter(|| {
                black_box(
                    red.plan::<DekeError, _>(&rcfg, black_box(&run), &noop, &())
                        .0,
                )
                .is_ok()
            })
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
