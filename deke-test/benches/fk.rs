use criterion::{Criterion, black_box, criterion_group, criterion_main};
use deke_test::m20id12l;
use deke_types::SRobotQ;
use deke_types::{DHChain, FKChain, HPChain, HPJoint, URDFChain};

fn make_hp_chain() -> HPChain<6> {
    let dh = m20id12l::DH_JOINTS;
    let hp_joints = dh.map(|j| HPJoint {
        a: j.a,
        alpha: j.alpha,
        beta: 0.0,
        d: j.d,
        theta_offset: j.theta_offset,
    });
    HPChain::new(hp_joints)
}

fn gen_configs(n: usize) -> Vec<SRobotQ<6>> {
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
            SRobotQ(q)
        })
        .collect()
}

fn bench_fk(c: &mut Criterion) {
    let configs = gen_configs(1024);

    let dh = DHChain::new(m20id12l::DH_JOINTS);
    let hp = make_hp_chain();
    let urdf = URDFChain::new(m20id12l::URDF_JOINTS);
    let vamp_robot = vamp::Robot::M20ID12L;

    c.bench_function("fk_dh", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            black_box(dh.fk(q))
        });
    });

    c.bench_function("fk_hp", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            black_box(hp.fk(q))
        });
    });

    c.bench_function("fk_urdf", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            black_box(urdf.fk(q))
        });
    });

    c.bench_function("fk_end_dh", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            black_box(dh.fk_end(q))
        });
    });

    c.bench_function("fk_end_hp", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            black_box(hp.fk_end(q))
        });
    });

    c.bench_function("fk_end_urdf", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            black_box(urdf.fk_end(q))
        });
    });

    c.bench_function("fk_end_vamp_eefk", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &configs[i % configs.len()];
            i += 1;
            black_box(vamp_robot.eefk(&q.0))
        });
    });
}

criterion_group!(benches, bench_fk);
criterion_main!(benches);
