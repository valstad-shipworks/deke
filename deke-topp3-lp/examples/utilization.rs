//! Limit utilization of the output trajectory. Per output tick the utilization is
//! `max(|Δq|/dt / v_max, |Δ²q|/dt² / a_max, |Δ³q|/dt³ / j_max  over all joints,
//! ‖Δp_tip‖/dt / v_tcp)` — i.e. the max over the joint v/a/j limits AND the TCP
//! velocity cap (this crate caps TCP velocity only). 1.0 = riding a limit.

use std::f64::consts::PI;
use std::time::Duration;

use deke_kin::{DHJoint, JointLimits as KinJointLimits, Kinematics};
use deke_topp3_lp::{Topp3Lp, Topp3LpConstraints, Topp3LpTcp};
use deke_types::glam::DVec3;
use deke_types::{ContinuousFKChain, JointValidator, Retimer, SRobotPath, SRobotQ, SRobotTraj};

fn ur() -> Kinematics<6, f64> {
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

fn path<const N: usize>(wps: &[[f64; N]]) -> SRobotPath<N, f64> {
    SRobotPath::try_new(wps.iter().map(|w| SRobotQ::from_array(*w)).collect()).unwrap()
}

fn wide<const N: usize>() -> JointValidator<N, f64> {
    JointValidator::<N, f64>::new(
        SRobotQ::from_array([-10.0; N]),
        SRobotQ::from_array([10.0; N]),
    )
}

/// (peak utilization, mean utilization, binding quantity at peak).
fn utilization<const N: usize, FK: ContinuousFKChain<N, f64>>(
    traj: &SRobotTraj<N, f64>,
    fk: Option<&FK>,
    v: f64,
    a: f64,
    j: f64,
    tcp_cap: Option<f64>,
) -> (f64, f64, &'static str) {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let n = p.len();
    let tips: Option<Vec<DVec3>> = fk.map(|k| {
        (0..n)
            .map(|i| k.fk_end(&p[i]).unwrap().translation)
            .collect()
    });

    let (mut peak, mut sum, mut binding) = (0.0f64, 0.0f64, "none");
    for i in 1..n {
        let mut u = 0.0f64;
        let mut b = "joint-v";
        for ax in 0..N {
            u = u.max((p[i].0[ax] - p[i - 1].0[ax]).abs() / dt / v);
        }
        if i >= 2 {
            for ax in 0..N {
                let acc =
                    (p[i].0[ax] - 2.0 * p[i - 1].0[ax] + p[i - 2].0[ax]).abs() / (dt * dt) / a;
                if acc > u {
                    u = acc;
                    b = "joint-a";
                }
            }
        }
        if i >= 3 {
            for ax in 0..N {
                let jk = (p[i].0[ax] - 3.0 * p[i - 1].0[ax] + 3.0 * p[i - 2].0[ax]
                    - p[i - 3].0[ax])
                    .abs()
                    / (dt * dt * dt)
                    / j;
                if jk > u {
                    u = jk;
                    b = "joint-j";
                }
            }
        }
        if let (Some(tips), Some(cap)) = (&tips, tcp_cap) {
            let sp = tips[i].distance(tips[i - 1]) / dt / cap;
            if sp > u {
                u = sp;
                b = "tcp-v";
            }
        }
        if u > peak {
            peak = u;
            binding = b;
        }
        sum += u;
    }
    (peak, sum / (n - 1).max(1) as f64, binding)
}

fn dt8() -> Duration {
    Duration::from_millis(8)
}

fn main() {
    let v6 = wide::<6>();
    let v1 = wide::<1>();
    let fk = ur();

    println!("deke-topp3-lp limit utilization (max over joint v/a/j AND TCP v)\n");
    println!("{:<34} {:>9} {:>9}   binds@peak", "case", "peak", "mean");

    let report = |name: &str, (peak, mean, bind): (f64, f64, &str)| {
        println!(
            "{name:<34} {:>8.1}% {:>8.1}%   {bind}",
            peak * 100.0,
            mean * 100.0
        );
    };

    // joint-only cases (no TCP cap)
    let one = path::<1>(&[[0.0], [1.0]]);
    let (t, _) = Topp3Lp::<1>::new().retime(
        &Topp3LpConstraints::symmetric(1.0, 2.0, 200.0, dt8()),
        &one,
        &v1,
        &(),
    );
    report(
        "1-DOF rest-to-rest",
        utilization::<1, Kinematics<1, f64>>(&t.unwrap(), None, 1.0, 2.0, 200.0, None),
    );

    let straight = path::<6>(&[
        [0.0, -1.2, 1.5, -0.3, 0.5, 0.0],
        [0.6, -0.6, 0.9, 0.3, -0.2, 0.8],
    ]);
    let (t, _) = Topp3Lp::<6>::new().retime(
        &Topp3LpConstraints::symmetric(1.5, 8.0, 400.0, dt8()),
        &straight,
        &v6,
        &(),
    );
    report(
        "6-DOF straight (joint)",
        utilization(&t.unwrap(), Some(&fk), 1.5, 8.0, 400.0, None),
    );

    let curved = path::<6>(&[
        [0.0, -1.3, 1.5, 0.0, 0.0, 0.0],
        [0.2, -1.1, 1.3, -0.1, 0.1, 0.1],
        [0.4, -0.9, 1.1, -0.2, 0.2, 0.2],
        [0.6, -0.7, 0.9, -0.3, 0.1, 0.3],
        [0.8, -0.5, 0.7, -0.4, 0.0, 0.4],
    ]);
    let (t, _) = Topp3Lp::<6>::new().retime(
        &Topp3LpConstraints::symmetric(1.5, 4.0, 200.0, dt8()),
        &curved,
        &v6,
        &(),
    );
    report(
        "6-DOF curved 5wp (joint)",
        utilization(&t.unwrap(), Some(&fk), 1.5, 4.0, 200.0, None),
    );

    // TCP-capped case: joints loose so the TCP velocity cap binds.
    let cap = 0.25;
    let c = Topp3LpConstraints::symmetric(5.0, 30.0, 3000.0, dt8()).with_tcp_speed(cap);
    let (t, _) = Topp3LpTcp::new(&fk).retime(&c, &straight, &v6, &());
    report(
        "6-DOF straight + TCP cap",
        utilization(&t.unwrap(), Some(&fk), 5.0, 30.0, 3000.0, Some(cap)),
    );

    let c = Topp3LpConstraints::symmetric(1.5, 8.0, 400.0, dt8()).with_tcp_speed(0.4);
    let (t, _) = Topp3LpTcp::new(&fk).retime(&c, &curved, &v6, &());
    report(
        "6-DOF curved + TCP cap",
        utilization(&t.unwrap(), Some(&fk), 1.5, 8.0, 400.0, Some(0.4)),
    );
}
