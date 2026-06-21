use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use deke_multipath::{
    plan_multipath_straight, MultiPathSettings, ReqPath, TransitionCost,
};
use deke_types::{DekeError, DekeResult, SRobotPath, SRobotQ, SRobotQLike, Validator};

#[derive(Clone, Debug)]
struct AllowAll;

impl Validator<6, (), f64> for AllowAll {
    type Context<'ctx> = ();

    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<6, E, f64>>(
        &self,
        _q: A,
        _ctx: &(),
    ) -> DekeResult<()> {
        Ok(())
    }

    fn validate_motion<'ctx>(&self, _qs: &[SRobotQ<6, f64>], _ctx: &()) -> DekeResult<()> {
        Ok(())
    }
}

/// Deterministic pseudo-random joint configuration in `[-3, 3]^6` from a seed.
fn cfg(seed: u64) -> SRobotQ<6, f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut a = [0.0_f64; 6];
    for x in a.iter_mut() {
        s ^= s >> 30;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s ^= s >> 27;
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        *x = u * 6.0 - 3.0;
    }
    SRobotQ(a)
}

fn build_reqpaths(clusters: usize, opts: usize) -> Vec<ReqPath<6>> {
    (0..clusters)
        .map(|c| {
            let variants: Vec<SRobotPath<6, f64>> = (0..opts)
                .map(|o| {
                    let base = (c * 97 + o * 13) as u64;
                    SRobotPath::try_new(vec![cfg(base), cfg(base.wrapping_add(7))]).unwrap()
                })
                .collect();
            if opts == 1 {
                ReqPath::OneWay(variants.into_iter().next().unwrap())
            } else {
                ReqPath::ManyWays(variants)
            }
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let validator = AllowAll;
    let ctx = ();
    let weights = SRobotQ([1.0; 6]);
    let start = SRobotQ([0.0; 6]);

    let mut group = c.benchmark_group("plan_multipath_straight");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));

    for &clusters in &[8usize, 14, 18, 24, 36] {
        for &opts in &[1usize, 2, 4] {
            let req = build_reqpaths(clusters, opts);
            let settings = MultiPathSettings::new(start);
            let cost = TransitionCost::JointWeighted(weights);
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("c{clusters}_k{opts}")),
                &(),
                |b, _| {
                    b.iter(|| {
                        let out = plan_multipath_straight(
                            black_box(&req),
                            &cost,
                            &settings,
                            &validator,
                            &ctx,
                        )
                        .unwrap();
                        black_box(out);
                    });
                },
            );
        }
    }
    group.finish();
}

/// Connector planning with a real RRT planner: the connectors between chosen
/// paths are independent point-to-point plans, so the `rayon` feature fans
/// them across the rayon pool. Iteration budgets are capped so each plan is
/// bounded and the benchmark stays quick.
fn bench_rrt_connectors(c: &mut Criterion) {
    use deke_multipath::{plan_multipath, TransitionPlanner};
    use deke_rrt::{AorrtcPlanner, AorrtcSettings, StartEnd};

    let planner = AorrtcPlanner::<6>::new();
    let mut config = AorrtcSettings::<6>::new(SRobotQ([-3.5; 6]), SRobotQ([3.5; 6]));
    config.max_iterations = 4000;
    config.stall_iterations = 1500;
    config.rrtc.max_iterations = 4000;

    let validator = AllowAll;
    let ctx = ();
    let weights = SRobotQ([1.0; 6]);
    let start = SRobotQ([0.0; 6]);

    let mut group = c.benchmark_group("plan_multipath_rrt");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for &clusters in &[8usize, 14] {
        let req = build_reqpaths(clusters, 1);
        let settings = MultiPathSettings::new(start);
        let cost = TransitionCost::JointWeighted(weights);
        let transition = TransitionPlanner {
            planner: &planner,
            config: &config,
            make_waypoints: |s, e| StartEnd { start: s, end: e },
        };
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("c{clusters}")),
            &(),
            |b, _| {
                b.iter(|| {
                    let out =
                        plan_multipath(&req, &cost, &settings, &transition, &validator, &ctx)
                            .unwrap();
                    black_box(out);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench, bench_rrt_connectors);
criterion_main!(benches);
