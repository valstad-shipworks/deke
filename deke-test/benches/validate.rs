use criterion::{Criterion, black_box, criterion_group, criterion_main};
use deke_test::m20id12l;
use deke_types::Validator as _;

fn gen_configs(n: usize) -> Vec<[f32; 6]> {
    let mut rng = 0xDEADBEEFu64;
    let lower = m20id12l::JOINT_LOWER;
    let upper = m20id12l::JOINT_UPPER;
    (0..n)
        .map(|_| {
            let mut q = [0.0f32; 6];
            for j in 0..6 {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let t = (rng >> 33) as f32 / (1u64 << 31) as f32;
                q[j] = lower[j] + t * (upper[j] - lower[j]);
            }
            q
        })
        .collect()
}

fn bench_validate(c: &mut Criterion) {
    let configs = gen_configs(1024);

    let env = wreck::Collider::default();
    let validator = m20id12l::validator(env);
    let vamp_env = vamp::Environment::new();
    let vamp_robot = vamp::Robot::M20ID12L;

    c.bench_function("deke_validate_single", |b| {
        let mut v = validator.clone();
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            let _ = black_box(v.validate(deke_types::SRobotQ(*q)));
        });
    });

    c.bench_function("vamp_validate_single", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            black_box(vamp_robot.validate(q, &vamp_env, true))
        });
    });

    let no_limits = validator.clone().1;

    c.bench_function("deke_validate_no_limits", |b| {
        let mut v = no_limits.clone();
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            let _ = black_box(v.validate(deke_types::SRobotQ(*q)));
        });
    });

    c.bench_function("vamp_validate_no_limits", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            black_box(vamp_robot.validate(q, &vamp_env, false))
        });
    });

    let dyn_validator: deke_wreck::DynamicWreckValidator = validator.1.clone().into();
    let dyn_full: deke_types::ValidatorAnd<
        deke_types::DynamicJointValidator,
        deke_wreck::DynamicWreckValidator,
    > = deke_types::ValidatorAnd(validator.0.clone().into(), dyn_validator);

    c.bench_function("dynamic_validate_single", |b| {
        let mut v = dyn_full.clone();
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            let _ = black_box(v.validate(deke_types::SRobotQ(*q)));
        });
    });

    let dyn_no_limits: deke_wreck::DynamicWreckValidator = no_limits.clone().into();

    c.bench_function("dynamic_validate_no_limits", |b| {
        let mut v = dyn_no_limits.clone();
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            let _ = black_box(v.validate(deke_types::SRobotQ(*q)));
        });
    });
}

criterion_group!(benches, bench_validate);
criterion_main!(benches);
