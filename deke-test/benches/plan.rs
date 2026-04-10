use criterion::{Criterion, black_box, criterion_group, criterion_main};
use deke_test::m20id12l;
use deke_types::{Planner as _, SRobotQ};

const PROBLEMS: [([f32; 6], [f32; 6]); 5] = [
    (
        [0.0, 0.5, -0.5, 0.0, 0.5, 0.0],
        [2.0, -0.3, 0.8, 1.0, -0.5, 1.5],
    ),
    (
        [-1.0, 0.2, -0.3, 0.5, 0.1, -0.5],
        [1.5, -0.8, 0.4, -1.0, 0.8, 2.0],
    ),
    (
        [0.3, -0.2, 0.7, -0.3, 0.4, 0.1],
        [-0.8, 0.9, -0.4, 0.8, -0.6, 1.2],
    ),
    (
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.5, -0.5, 1.0, -1.0, 1.0],
    ),
    (
        [0.2, 0.8, -0.6, 0.3, -0.2, 0.5],
        [1.8, -0.6, 0.3, -0.7, 0.9, -1.2],
    ),
];

fn make_obstacle_env() -> (wreck::Collider, vamp::Environment) {
    let mut wreck_env = wreck::Collider::default();
    let mut vamp_env = vamp::Environment::new();

    let obstacles: &[(f32, f32, f32, f32)] = &[
        (0.5, 0.0, 0.5, 0.15),
        (-0.3, 0.4, 0.3, 0.12),
        (0.0, -0.5, 0.6, 0.10),
    ];
    for &(x, y, z, r) in obstacles {
        wreck_env.add(wreck::Sphere::new(glam::Vec3::new(x, y, z), r));
        vamp_env.add_sphere(x, y, z, r);
    }

    (wreck_env, vamp_env)
}

fn bench_plan_rrtc(c: &mut Criterion) {
    let (wreck_env, vamp_env) = make_obstacle_env();
    let validator = m20id12l::validator(wreck_env);
    let vamp_robot = vamp::Robot::M20ID12L;

    let mut rrtc_settings = deke_rrt::RrtcSettings::new(
        SRobotQ(m20id12l::JOINT_LOWER),
        SRobotQ(m20id12l::JOINT_UPPER),
    );
    rrtc_settings.range = 2.0;
    rrtc_settings.resolution = 1.0 / 32.0;
    let planner = m20id12l::rrtc(rrtc_settings);
    let vamp_settings = vamp::RRTCSettings::default();

    c.bench_function("deke_rrtc", |b| {
        b.iter(|| {
            for &(start, goal) in &PROBLEMS {
                let mut v = validator.clone();
                let (result, diag) = planner.plan(SRobotQ(start), SRobotQ(goal), &mut v);
                black_box((&result, &diag));
            }
        });
    });

    c.bench_function("vamp_rrtc", |b| {
        b.iter(|| {
            for &(start, goal) in &PROBLEMS {
                let mut rng = vamp::Rng::halton(vamp_robot);
                let result =
                    vamp_robot.plan_rrtc(&start, &goal, &vamp_env, &vamp_settings, &mut rng);
                black_box(&result);
            }
        });
    });
}

fn bench_plan_empty_env(c: &mut Criterion) {
    let validator = m20id12l::validator(wreck::Collider::default());
    let vamp_robot = vamp::Robot::M20ID12L;
    let vamp_env = vamp::Environment::new();

    let mut rrtc_settings = deke_rrt::RrtcSettings::new(
        SRobotQ(m20id12l::JOINT_LOWER),
        SRobotQ(m20id12l::JOINT_UPPER),
    );
    rrtc_settings.range = 2.0;
    rrtc_settings.resolution = 1.0 / 32.0;
    let planner = m20id12l::rrtc(rrtc_settings);
    let vamp_settings = vamp::RRTCSettings::default();

    c.bench_function("deke_rrtc_empty", |b| {
        b.iter(|| {
            for &(start, goal) in &PROBLEMS {
                let mut v = validator.clone();
                let (result, diag) = planner.plan(SRobotQ(start), SRobotQ(goal), &mut v);
                black_box((&result, &diag));
            }
        });
    });

    c.bench_function("vamp_rrtc_empty", |b| {
        b.iter(|| {
            for &(start, goal) in &PROBLEMS {
                let mut rng = vamp::Rng::halton(vamp_robot);
                let result =
                    vamp_robot.plan_rrtc(&start, &goal, &vamp_env, &vamp_settings, &mut rng);
                black_box(&result);
            }
        });
    });
}

fn bench_plan_aorrtc(c: &mut Criterion) {
    let (wreck_env, vamp_env) = make_obstacle_env();
    let validator = m20id12l::validator(wreck_env);
    let vamp_robot = vamp::Robot::M20ID12L;

    let mut aorrtc_settings = deke_rrt::AorrtcSettings::new(
        SRobotQ(m20id12l::JOINT_LOWER),
        SRobotQ(m20id12l::JOINT_UPPER),
    );
    aorrtc_settings.rrtc.range = 2.0;
    aorrtc_settings.rrtc.resolution = 1.0 / 32.0;
    let planner = m20id12l::aorrtc(aorrtc_settings);
    let vamp_settings = vamp::AORRTCSettings::default();

    c.bench_function("deke_aorrtc", |b| {
        b.iter(|| {
            for &(start, goal) in &PROBLEMS {
                let mut v = validator.clone();
                let (result, diag) = planner.plan(SRobotQ(start), SRobotQ(goal), &mut v);
                black_box((&result, &diag));
            }
        });
    });

    c.bench_function("vamp_aorrtc", |b| {
        b.iter(|| {
            for &(start, goal) in &PROBLEMS {
                let mut rng = vamp::Rng::halton(vamp_robot);
                let result =
                    vamp_robot.plan_aorrtc(&start, &goal, &vamp_env, &vamp_settings, &mut rng);
                black_box(&result);
            }
        });
    });
}

fn m20id12l_kin_limits() -> deke_rrt::KinematicLimits<6> {
    // From robot_assets/robots/m20id12l/fanuc_motion_limits.json
    // Middle value (index 10) of each joint's no_payload field, converted deg -> rad.
    deke_rrt::KinematicLimits {
        joints: [
            deke_rrt::JointKinLimits {
                v_max: 210.0_f64.to_radians(),
                a_max: 605.7692_f64.to_radians(),
                j_max: 3494.8225_f64.to_radians(),
            },
            deke_rrt::JointKinLimits {
                v_max: 210.0_f64.to_radians(),
                a_max: 605.7692_f64.to_radians(),
                j_max: 3494.8225_f64.to_radians(),
            },
            deke_rrt::JointKinLimits {
                v_max: 265.0_f64.to_radians(),
                a_max: 764.4231_f64.to_radians(),
                j_max: 4410.1333_f64.to_radians(),
            },
            deke_rrt::JointKinLimits {
                v_max: 420.0_f64.to_radians(),
                a_max: 1211.5385_f64.to_radians(),
                j_max: 6989.645_f64.to_radians(),
            },
            deke_rrt::JointKinLimits {
                v_max: 450.0_f64.to_radians(),
                a_max: 1298.0769_f64.to_radians(),
                j_max: 7488.906_f64.to_radians(),
            },
            deke_rrt::JointKinLimits {
                v_max: 720.0_f64.to_radians(),
                a_max: 2076.923_f64.to_radians(),
                j_max: 11982.249_f64.to_radians(),
            },
        ],
    }
}

fn bench_plan_krrtc(c: &mut Criterion) {
    let (wreck_env, _) = make_obstacle_env();
    let validator = m20id12l::validator(wreck_env);

    let settings = deke_rrt::KrrtcSettings::new(
        SRobotQ(m20id12l::JOINT_LOWER),
        SRobotQ(m20id12l::JOINT_UPPER),
        m20id12l_kin_limits(),
    );
    let planner = m20id12l::krrtc(settings);

    c.bench_function("deke_krrtc", |b| {
        b.iter(|| {
            for &(start, goal) in &PROBLEMS {
                let mut v = validator.clone();
                let (result, diag) = planner.plan(SRobotQ(start), SRobotQ(goal), &mut v);
                black_box((&result, &diag));
            }
        });
    });
}

fn bench_plan_krrtc_empty(c: &mut Criterion) {
    let validator = m20id12l::validator(wreck::Collider::default());

    let settings = deke_rrt::KrrtcSettings::new(
        SRobotQ(m20id12l::JOINT_LOWER),
        SRobotQ(m20id12l::JOINT_UPPER),
        m20id12l_kin_limits(),
    );
    let planner = m20id12l::krrtc(settings);

    c.bench_function("deke_krrtc_empty", |b| {
        b.iter(|| {
            for &(start, goal) in &PROBLEMS {
                let mut v = validator.clone();
                let (result, diag) = planner.plan(SRobotQ(start), SRobotQ(goal), &mut v);
                black_box((&result, &diag));
            }
        });
    });
}

fn bench_plan_aorrtc_empty(c: &mut Criterion) {
    let validator = m20id12l::validator(wreck::Collider::default());
    let vamp_robot = vamp::Robot::M20ID12L;
    let vamp_env = vamp::Environment::new();

    let mut aorrtc_settings = deke_rrt::AorrtcSettings::new(
        SRobotQ(m20id12l::JOINT_LOWER),
        SRobotQ(m20id12l::JOINT_UPPER),
    );
    aorrtc_settings.rrtc.range = 2.0;
    aorrtc_settings.rrtc.resolution = 1.0 / 32.0;
    let planner = m20id12l::aorrtc(aorrtc_settings);
    let vamp_settings = vamp::AORRTCSettings::default();

    c.bench_function("deke_aorrtc_empty", |b| {
        b.iter(|| {
            for &(start, goal) in &PROBLEMS {
                let mut v = validator.clone();
                let (result, diag) = planner.plan(SRobotQ(start), SRobotQ(goal), &mut v);
                black_box((&result, &diag));
            }
        });
    });

    c.bench_function("vamp_aorrtc_empty", |b| {
        b.iter(|| {
            for &(start, goal) in &PROBLEMS {
                let mut rng = vamp::Rng::halton(vamp_robot);
                let result =
                    vamp_robot.plan_aorrtc(&start, &goal, &vamp_env, &vamp_settings, &mut rng);
                black_box(&result);
            }
        });
    });
}

criterion_group!(
    benches,
    bench_plan_rrtc,
    bench_plan_empty_env,
    bench_plan_krrtc,
    bench_plan_krrtc_empty,
    bench_plan_aorrtc,
    bench_plan_aorrtc_empty
);
criterion_main!(benches);
