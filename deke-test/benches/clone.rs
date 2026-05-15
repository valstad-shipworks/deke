use criterion::{Criterion, black_box, criterion_group, criterion_main};
use deke_test::m20id12l;

fn bench_clone(c: &mut Criterion) {
    let validator = m20id12l::validator();
    let no_limits = validator.clone().1;
    let dyn_no_limits: deke_wreck::DynamicWreckValidator = no_limits.clone().into();
    let dyn_full: deke_types::ValidatorAnd<
        deke_types::DynamicJointValidator,
        deke_wreck::DynamicWreckValidator,
    > = deke_types::ValidatorAnd(validator.0.clone().into(), dyn_no_limits.clone());

    c.bench_function("deke_clone_full", |b| {
        b.iter(|| black_box(validator.clone()));
    });

    c.bench_function("deke_clone_no_limits", |b| {
        b.iter(|| black_box(no_limits.clone()));
    });

    c.bench_function("dynamic_clone_full", |b| {
        b.iter(|| black_box(dyn_full.clone()));
    });

    c.bench_function("dynamic_clone_no_limits", |b| {
        b.iter(|| black_box(dyn_no_limits.clone()));
    });
}

criterion_group!(benches, bench_clone);
criterion_main!(benches);
