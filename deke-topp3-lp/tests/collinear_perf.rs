//! Performance characterization of `Conditioning::Collinear` on a long 7-DOF RTU
//! chord. NOT a pass/fail correctness test — it documents the slowness described
//! in `docs/FUTURE.md` (the "Collinear hang" is pathological re-solve blowup, not
//! an infinite loop). `#[ignore]` because each Collinear case takes seconds; run
//! explicitly with:
//!
//!   cargo test --test collinear_perf -- --ignored --nocapture
//!
//! Observed (M-series, release-ish dev build), real material chain, Path B:
//!   Raw            tcp    ~3.0s     Collinear(0.20) tcp  ~11.8s
//!   Collinear(0.10) tcp   ~6.4s     Collinear(0.05) tcp  ~16.5s
//!   Collinear(0.05) nocap ~5.2s     Raw            nocap  ~1.3s
//! The non-monotonic res→time confirms the cost tracks re-solve COUNT (driven by
//! the FD-verify horizon growth), not knot count.

mod common;

use std::time::{Duration, Instant};

use deke_topp3_lp::{Conditioning, JointLimits, Topp3Lp, Topp3LpConstraints, Topp3LpTcp};
use deke_types::{Retimer, SRobotPath, SRobotQ};

fn limits() -> JointLimits<7> {
    JointLimits {
        v_max: SRobotQ::from_array([
            1.422, 1.099557, 0.942478, 0.890118, 1.256637, 1.256637, 2.094395,
        ]),
        a_max: SRobotQ::from_array([
            3.262729, 3.096281, 2.653955, 2.506513, 3.538607, 3.538607, 5.897679,
        ]),
        j_max: SRobotQ::from_array([
            5.996099, 13.966876, 11.971608, 11.306519, 15.962144, 15.962144, 26.603575,
        ]),
    }
}

fn cons(cond: Conditioning, tcp: bool) -> Topp3LpConstraints<7> {
    let c = Topp3LpConstraints::<7> {
        joint: limits(),
        tcp: deke_topp3_lp::TcpLimits::default(),
        output_dt: Duration::from_secs_f64(0.008),
        conditioning: cond,
        sharp_corner_angle: Some(30.0_f64.to_radians()),
    };
    if tcp { c.with_tcp_speed(2.0) } else { c }
}

fn time_case(label: &str, wps: &[[f64; 7]; 2], cond: Conditioning, tcp: bool) {
    let chain = common::material_7dof();
    let path = SRobotPath::<7, f64>::try_new(wps.iter().map(|w| SRobotQ::from_array(*w)).collect())
        .unwrap();
    let nv = common::wide_validator::<7>();
    let t0 = Instant::now();
    let (ok, n) = if tcp {
        let (r, d) = Topp3LpTcp::new(&chain).retime(&cons(cond, true), &path, &nv, &());
        (r.is_ok(), d.output_samples)
    } else {
        let (r, d) = Topp3Lp::<7>::new().retime(&cons(cond, false), &path, &nv, &());
        (r.is_ok(), d.output_samples)
    };
    println!(
        "  {label}: {} ({} samples) in {:.2}s",
        if ok { "OK" } else { "ERR" },
        n,
        t0.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "perf characterization (seconds per case); see docs/FUTURE.md"]
fn collinear_is_slow_but_terminates() {
    println!("\n--- TCP-capped (Topp3LpTcp, cap 2.0) ---");
    time_case(
        "B Raw            tcp",
        &common::MATERIAL_PATH_B,
        Conditioning::Raw,
        true,
    );
    time_case(
        "B Collinear(0.20) tcp",
        &common::MATERIAL_PATH_B,
        Conditioning::Collinear(0.20),
        true,
    );
    time_case(
        "B Collinear(0.10) tcp",
        &common::MATERIAL_PATH_B,
        Conditioning::Collinear(0.10),
        true,
    );
    time_case(
        "B Collinear(0.05) tcp",
        &common::MATERIAL_PATH_B,
        Conditioning::Collinear(0.05),
        true,
    );
    time_case(
        "A Collinear(0.05) tcp",
        &common::MATERIAL_PATH_A,
        Conditioning::Collinear(0.05),
        true,
    );
    println!("--- joint-only (Topp3Lp, no TCP cap) ---");
    time_case(
        "B Collinear(0.05) nocap",
        &common::MATERIAL_PATH_B,
        Conditioning::Collinear(0.05),
        false,
    );
    time_case(
        "B Raw            nocap",
        &common::MATERIAL_PATH_B,
        Conditioning::Raw,
        false,
    );
}
