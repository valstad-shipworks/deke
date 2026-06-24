mod common;

use std::time::Duration;

use deke_linear::{
    CartesianLinearPlanner, ConstantSpeedRetimer, JointLimits, LinearConstraints, NoopValidator,
    PathConditioning, PlannerOptions, RailAxis, RailConfig, RailLinearPlanner, RailMountedChain,
    RailOptions, RailRefine, RailYawConfig, RailYawPlanner, RedundantAxis, RedundantConfig,
    RedundantLinearPlanner, RedundantOptions, TcpLimits, WeaveOptions, condition,
};
use deke_types::glam::DVec3;
use deke_types::{DekeError, FKChain, Planner, SRobotPath, SRobotQ, SRobotTraj};

/// Peak transverse excursion (along `dir`) and temporal frequency of a TCP stream.
fn measure(tcp: &[DVec3], dir: DVec3, dt: f64) -> (f64, f64) {
    let origin = tcp[0];
    let lat: Vec<f64> = tcp.iter().map(|t| (*t - origin).dot(dir)).collect();
    let amp = lat.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let crossings = (1..lat.len())
        .filter(|&i| lat[i - 1] <= 0.0 && lat[i] > 0.0)
        .count();
    (amp, crossings as f64 / ((tcp.len() - 1) as f64 * dt))
}

/// Sinusoidal weave: 3 mm tip-to-tip transverse oscillation at 2 Hz, travelling a
/// straight seam at 18 inches/min. The weave is a spatial overlay locked to seam
/// arc length (wavelength = travel/frequency); the retimer holds constant travel
/// speed via `retime_weave`, so the 2 Hz / 18 IPM relationship is exact.
#[test]
fn sinusoidal_weave_3mm_2hz_18ipm() {
    let arm = common::ur();

    let amplitude = 3.0e-3; // peak-to-peak (tip-to-tip)
    let freq = 2.0; // Hz
    let travel = 18.0 * 0.0254 / 60.0; // m/s
    let lambda = WeaveOptions::wavelength_for(freq, travel); // ≈ 3.81 mm

    // The weave axis is tool-frame (degeneracy-free), applied as `R·axis`. Pick the
    // tool-frame vector that maps to world +Y, so the weave is transverse to the
    // world-X seam (a flat-position weave) regardless of the torch orientation.
    let base_rot = arm.fk_end(&common::anchor()).unwrap().matrix3;
    let weave_dir = DVec3::Y;
    let weave = WeaveOptions {
        axis: RedundantAxis::Custom(base_rot.inverse() * weave_dir),
        ..WeaveOptions::sine(amplitude, lambda)
    };

    // Straight seam through the anchor, ~20 weave cycles long.
    let seam_len = 20.0 * lambda;
    let poses = common::straight(&arm, DVec3::X, seam_len, 64);
    let runs = condition(&poses, &PathConditioning::default()).unwrap();
    assert_eq!(runs.len(), 1, "a straight seam conditions to one run");

    let joint = JointLimits::symmetric(2.0, 8.0, 80.0);
    // Resolve the weave without aliasing (≥ 15 samples/cycle); the weave is the
    // high-frequency path content, so there is no coarse-grid escape. Pick a
    // sample_ds that divides the run length evenly so the planner's final sample
    // lands exactly at the end (no squished last interval at the boundary).
    let len = runs[0].length();
    let nseg = (len / (weave.max_sample_ds() * 0.5)).round().max(1.0);
    let opts = PlannerOptions::<6> {
        sample_ds: len / nseg,
        ..PlannerOptions::default()
    };
    let cons = LinearConstraints {
        joint: joint.clone(),
        tcp: TcpLimits::speed(travel),
        output_dt: Duration::from_millis(8),
        forbid_interior_dips: false,
        corner_smoothing: Some(0.01),
    };

    let planner = CartesianLinearPlanner::new(&arm);
    let retimer = ConstantSpeedRetimer::new(&arm);

    let mut all: Vec<SRobotQ<6, f64>> = Vec::new();
    for run in &runs {
        let weaving = run.clone().with_weave(weave);
        let (path, _) = planner.plan::<DekeError, _>(&opts, &weaving, &NoopValidator::<6>, &());
        let path = path.expect("weave plan");
        let n = path.len();
        // Seam progress at each planned sample (uniform sample_ds along the seam);
        // this is what the retimer holds at constant travel speed.
        let progress: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * len).collect();
        let (traj, _) = retimer.retime_weave(&cons, &path, &progress, &NoopValidator::<6>, &());
        let traj = traj.expect("weave retime");
        let it = traj.path().iter().copied();
        if all.is_empty() {
            all.extend(it);
        } else {
            all.extend(it.skip(1));
        }
    }
    let traj = SRobotTraj::new(Duration::from_millis(8), SRobotPath::try_new(all).unwrap());
    let p = traj.path();
    let dt = traj.dt().as_secs_f64();
    let eps = 1.0 + 1e-9;

    // Hard limit: finite-difference joint v/a/j stay under the ceilings.
    for i in 1..p.len() {
        for j in 0..6 {
            assert!(
                (p[i].0[j] - p[i - 1].0[j]).abs() / dt <= joint.v_max.0[j] * eps,
                "joint {j} velocity over limit"
            );
        }
    }
    for i in 2..p.len() {
        for j in 0..6 {
            let a = (p[i].0[j] - 2.0 * p[i - 1].0[j] + p[i - 2].0[j]).abs() / (dt * dt);
            assert!(a <= joint.a_max.0[j] * eps, "joint {j} accel over limit");
        }
    }
    for i in 3..p.len() {
        for j in 0..6 {
            let jk = (p[i].0[j] - 3.0 * p[i - 1].0[j] + 3.0 * p[i - 2].0[j] - p[i - 3].0[j]).abs()
                / (dt * dt * dt);
            assert!(jk <= joint.j_max.0[j] * eps, "joint {j} jerk over limit");
        }
    }

    // Executed TCP: transverse offset (world Y, the weave plane) and travel (X).
    let origin = arm.fk_end(&p[0]).unwrap().translation;
    let tcp: Vec<DVec3> = p
        .iter()
        .map(|q| arm.fk_end(q).unwrap().translation)
        .collect();
    let lat: Vec<f64> = tcp.iter().map(|t| (*t - origin).dot(weave_dir)).collect();
    let amp = lat.iter().fold(0.0f64, |m, &v| m.max(v.abs()));

    let mut crossings = 0usize;
    for i in 1..lat.len() {
        if lat[i - 1] <= 0.0 && lat[i] > 0.0 {
            crossings += 1;
        }
    }
    let duration = (p.len() - 1) as f64 * dt;
    let meas_freq = crossings as f64 / duration;
    let travelled = (tcp[tcp.len() - 1] - tcp[0]).dot(DVec3::X);
    let meas_ipm = travelled / duration / 0.0254 * 60.0;

    println!(
        "weave: amp ±{:.3} mm ({:.3} mm tip-to-tip), freq {:.2} Hz, travel {:.1} IPM, λ {:.3} mm, samples {}",
        amp * 1e3,
        2.0 * amp * 1e3,
        meas_freq,
        meas_ipm,
        lambda * 1e3,
        p.len(),
    );

    // ~1.5 mm peak (3 mm tip-to-tip), allowing for the run-end taper pulling the
    // measured peak slightly under the nominal.
    assert!(
        amp > 1.35e-3 && amp < 1.55e-3,
        "weave peak {:.3} mm off the ±1.5 mm target",
        amp * 1e3
    );
    assert!(
        (meas_freq - freq).abs() < 0.15,
        "weave frequency {meas_freq:.2} Hz off 2 Hz"
    );
    assert!(
        (meas_ipm - 18.0).abs() < 1.0,
        "travel speed {meas_ipm:.1} IPM off 18 IPM"
    );
}

/// With no weave overlay the conditioned run and its retiming must be unchanged —
/// the weave field is purely additive.
#[test]
fn no_weave_matches_plain_run() {
    let arm = common::ur();
    let poses = common::straight(&arm, DVec3::X, 0.05, 24);
    let runs = condition(&poses, &PathConditioning::default()).unwrap();
    for s in [0.0, 0.0123, 0.025, 0.05] {
        let pose = runs[0].eval(s);
        let plain = runs[0].clone();
        assert_eq!(pose.translation, plain.eval(s).translation);
    }
}

/// The weave composes with the redundant tool-yaw planner: the yaw is resolved
/// freely while the torch oscillates transversely, all within joint limits.
#[test]
fn weave_with_redundant_yaw() {
    let arm = common::ur();
    let amplitude = 3.0e-3;
    let travel = 18.0 * 0.0254 / 60.0;
    let lambda = WeaveOptions::wavelength_for(2.0, travel);
    let base_rot = arm.fk_end(&common::anchor()).unwrap().matrix3;
    let weave_dir = DVec3::Y;
    let weave = WeaveOptions {
        axis: RedundantAxis::Custom(base_rot.inverse() * weave_dir),
        ..WeaveOptions::sine(amplitude, lambda)
    };
    let poses = common::straight(&arm, DVec3::X, 20.0 * lambda, 64);
    let run = condition(&poses, &PathConditioning::default()).unwrap()[0]
        .clone()
        .with_weave(weave);
    let len = run.length();
    let sample_ds = len / (len / (lambda / 30.0)).round();

    let joint = JointLimits::symmetric(2.0, 8.0, 80.0);
    let cfg = RedundantConfig::<6> {
        planner: PlannerOptions {
            sample_ds,
            max_velocity: travel,
            joint_v_max: joint.v_max,
            ..PlannerOptions::default()
        },
        redundant: RedundantOptions {
            axis: RedundantAxis::PosZ,
            yaw_window: (-30f64.to_radians(), 30f64.to_radians()),
            yaw_samples: 9,
            ..RedundantOptions::default()
        },
    };
    let cons = LinearConstraints {
        joint: joint.clone(),
        tcp: TcpLimits::speed(travel),
        output_dt: Duration::from_millis(8),
        forbid_interior_dips: false,
        corner_smoothing: Some(0.01),
    };
    let planner = RedundantLinearPlanner::new(&arm);
    let retimer = ConstantSpeedRetimer::new(&arm);
    let (path, _) = planner.plan::<DekeError, _>(&cfg, &run, &NoopValidator::<6>, &());
    let path = path.expect("weave+yaw plan");
    let n = path.len();
    let progress: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * len).collect();
    let (traj, _) = retimer.retime_weave(&cons, &path, &progress, &NoopValidator::<6>, &());
    let traj = traj.expect("weave+yaw retime");

    let p = traj.path();
    let dt = traj.dt().as_secs_f64();
    assert_fd_under::<6>(p, &joint, dt);
    let tcp: Vec<DVec3> = p
        .iter()
        .map(|q| arm.fk_end(q).unwrap().translation)
        .collect();
    let (amp, freq) = measure(&tcp, weave_dir, dt);
    println!("weave+yaw: amp ±{:.3} mm, freq {:.2} Hz", amp * 1e3, freq);
    assert!(
        amp > 1.35e-3 && amp < 1.55e-3,
        "weave+yaw amplitude {:.3} mm",
        amp * 1e3
    );
    assert!((freq - 2.0).abs() < 0.2, "weave+yaw freq {freq:.2} Hz");
}

/// The weave composes with the rail planner: the rail carries the arm along the
/// seam while the torch oscillates transversely, all within joint limits.
#[test]
fn weave_with_rail() {
    let arm = common::ur();
    let amplitude = 3.0e-3;
    let travel = 18.0 * 0.0254 / 60.0;
    let lambda = WeaveOptions::wavelength_for(2.0, travel);
    let base_rot = arm.fk_end(&common::anchor()).unwrap().matrix3;
    let weave_dir = DVec3::Y;
    let weave = WeaveOptions {
        axis: RedundantAxis::Custom(base_rot.inverse() * weave_dir),
        ..WeaveOptions::sine(amplitude, lambda)
    };
    let poses = common::straight(&arm, DVec3::X, 20.0 * lambda, 64);
    let run = condition(&poses, &PathConditioning::default()).unwrap()[0]
        .clone()
        .with_weave(weave);
    let len = run.length();
    let sample_ds = len / (len / (lambda / 30.0)).round();

    let lim = JointLimits::<7> {
        v_max: SRobotQ::from_array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        a_max: SRobotQ::from_array([20.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
        j_max: SRobotQ::from_array([2000.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]),
    };
    let cfg = RailConfig::<6, 7> {
        planner: PlannerOptions {
            sample_ds,
            max_velocity: travel,
            joint_v_max: lim.v_max,
            ..PlannerOptions::default()
        },
        rail: RailOptions {
            axis: RailAxis::PosX,
            window: (-0.2, 0.2),
            samples: 21,
            refine: RailRefine::Pchip,
            ..RailOptions::default()
        },
    };
    let chain = RailMountedChain::<6, 7, _>::new(&arm, RailAxis::PosX);
    let cons = LinearConstraints {
        joint: lim.clone(),
        tcp: TcpLimits::speed(travel),
        output_dt: Duration::from_millis(8),
        forbid_interior_dips: false,
        corner_smoothing: Some(0.01),
    };
    let planner = RailLinearPlanner::<6, 7, _>::new(&arm);
    let retimer = ConstantSpeedRetimer::new(&chain);
    let (path, _) = planner.plan::<DekeError, _>(&cfg, &run, &NoopValidator::<7>, &());
    let path = path.expect("weave+rail plan");
    let n = path.len();
    let progress: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * len).collect();
    let (traj, _) = retimer.retime_weave(&cons, &path, &progress, &NoopValidator::<7>, &());
    let traj = traj.expect("weave+rail retime");

    let p = traj.path();
    let dt = traj.dt().as_secs_f64();
    assert_fd_under::<7>(p, &lim, dt);
    let tcp: Vec<DVec3> = p
        .iter()
        .map(|q| chain.fk_end(q).unwrap().translation)
        .collect();
    let (amp, freq) = measure(&tcp, weave_dir, dt);
    println!("weave+rail: amp ±{:.3} mm, freq {:.2} Hz", amp * 1e3, freq);
    assert!(
        amp > 1.35e-3 && amp < 1.55e-3,
        "weave+rail amplitude {:.3} mm",
        amp * 1e3
    );
    assert!((freq - 2.0).abs() < 0.2, "weave+rail freq {freq:.2} Hz");
}

fn assert_fd_under<const N: usize>(
    p: &deke_types::SRobotPath<N, f64>,
    lim: &JointLimits<N>,
    dt: f64,
) {
    let eps = 1.0 + 1e-9;
    for i in 1..p.len() {
        for j in 0..N {
            assert!(
                (p[i].0[j] - p[i - 1].0[j]).abs() / dt <= lim.v_max.0[j] * eps,
                "v j{j}"
            );
        }
    }
    for i in 2..p.len() {
        for j in 0..N {
            let a = (p[i].0[j] - 2.0 * p[i - 1].0[j] + p[i - 2].0[j]).abs() / (dt * dt);
            assert!(a <= lim.a_max.0[j] * eps, "a j{j}");
        }
    }
    for i in 3..p.len() {
        for j in 0..N {
            let jk = (p[i].0[j] - 3.0 * p[i - 1].0[j] + 3.0 * p[i - 2].0[j] - p[i - 3].0[j]).abs()
                / (dt * dt * dt);
            assert!(jk <= lim.j_max.0[j] * eps, "j j{j}");
        }
    }
}

/// The full stack: weave + rail + redundant yaw together (hierarchical rail-then-yaw
/// with the transverse oscillation overlaid), all within joint limits.
#[test]
fn weave_with_rail_and_yaw() {
    let arm = common::ur();
    let amplitude = 3.0e-3;
    let travel = 18.0 * 0.0254 / 60.0;
    let lambda = WeaveOptions::wavelength_for(2.0, travel);
    let base_rot = arm.fk_end(&common::anchor()).unwrap().matrix3;
    let weave_dir = DVec3::Y;
    let weave = WeaveOptions {
        axis: RedundantAxis::Custom(base_rot.inverse() * weave_dir),
        ..WeaveOptions::sine(amplitude, lambda)
    };
    let poses = common::straight(&arm, DVec3::X, 20.0 * lambda, 64);
    let run = condition(&poses, &PathConditioning::default()).unwrap()[0]
        .clone()
        .with_weave(weave);
    let len = run.length();
    let sample_ds = len / (len / (lambda / 30.0)).round();

    let lim = JointLimits::<7> {
        v_max: SRobotQ::from_array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        a_max: SRobotQ::from_array([20.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
        j_max: SRobotQ::from_array([2000.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]),
    };
    let cfg = RailYawConfig::<6, 7> {
        planner: PlannerOptions {
            sample_ds,
            max_velocity: travel,
            joint_v_max: lim.v_max,
            ..PlannerOptions::default()
        },
        rail: RailOptions {
            axis: RailAxis::PosX,
            window: (-0.2, 0.2),
            samples: 21,
            refine: RailRefine::Pchip,
            ..RailOptions::default()
        },
        yaw: RedundantOptions {
            axis: RedundantAxis::PosZ,
            yaw_window: (-30f64.to_radians(), 30f64.to_radians()),
            yaw_samples: 9,
            ..RedundantOptions::default()
        },
    };
    let chain = RailMountedChain::<6, 7, _>::new(&arm, RailAxis::PosX);
    let cons = LinearConstraints {
        joint: lim.clone(),
        tcp: TcpLimits::speed(travel),
        output_dt: Duration::from_millis(8),
        forbid_interior_dips: false,
        corner_smoothing: Some(0.01),
    };
    let planner = RailYawPlanner::<6, 7, _>::new(&arm);
    let (path, _) = planner.plan::<DekeError, _>(&cfg, &run, &NoopValidator::<7>, &());
    let path = path.expect("weave+rail+yaw plan");
    let retimer = ConstantSpeedRetimer::new(&chain);
    let n = path.len();
    let progress: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * len).collect();
    let (traj, _) = retimer.retime_weave(&cons, &path, &progress, &NoopValidator::<7>, &());
    let traj = traj.expect("weave+rail+yaw retime");

    let p = traj.path();
    let dt = traj.dt().as_secs_f64();
    assert_fd_under::<7>(p, &lim, dt);
    let tcp: Vec<DVec3> = p
        .iter()
        .map(|q| chain.fk_end(q).unwrap().translation)
        .collect();
    let (amp, freq) = measure(&tcp, weave_dir, dt);
    println!(
        "weave+rail+yaw: amp ±{:.3} mm, freq {:.2} Hz",
        amp * 1e3,
        freq
    );
    assert!(
        amp > 1.35e-3 && amp < 1.55e-3,
        "weave+rail+yaw amplitude {:.3} mm",
        amp * 1e3
    );
    assert!((freq - 2.0).abs() < 0.2, "weave+rail+yaw freq {freq:.2} Hz");
}
