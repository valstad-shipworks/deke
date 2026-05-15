//! Runs the 6-DOF curved-path demo with verbose diagnostics enabled and prints the retimer
//! report + output trajectory preview. Useful for sanity-checking the solver output.

use std::f64::consts::FRAC_PI_2;

use deke_topp3tcp6::{Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{DHChain, DHJoint, JointValidator, Retimer, SRobotPath, SRobotQ};

fn main() {
    let fk = DHChain::<6, f64>::from_joints([
        DHJoint { a: 0.0, alpha: FRAC_PI_2, d: 0.089, theta_offset: 0.0 },
        DHJoint { a: -0.425, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
        DHJoint { a: -0.392, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
        DHJoint { a: 0.0, alpha: FRAC_PI_2, d: 0.109, theta_offset: 0.0 },
        DHJoint { a: 0.0, alpha: -FRAC_PI_2, d: 0.094, theta_offset: 0.0 },
        DHJoint { a: 0.0, alpha: 0.0, d: 0.082, theta_offset: 0.0 },
    ]);

    let waypoints = vec![
        SRobotQ::<6, f64>::from_array([0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::<6, f64>::from_array([0.2, -1.1, 1.3, -0.1, 0.1, 0.1]),
        SRobotQ::<6, f64>::from_array([0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::<6, f64>::from_array([0.6, -0.7, 0.9, -0.3, 0.1, 0.3]),
        SRobotQ::<6, f64>::from_array([0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<6, f64>::try_new(waypoints).expect("path ok");

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 4.0, 200.0);
    cfg.solver.diagnostics = true;

    let mut validator = JointValidator::<6, f64>::new(
        SRobotQ::from_array([-10.0; 6]),
        SRobotQ::from_array([10.0; 6]),
    );

    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    println!("{}", diag);

    match result {
        Ok(traj) => {
            println!(
                "trajectory: {} samples, duration {:.3}s, dt {:.1}ms",
                traj.len(),
                traj.duration().as_secs_f32(),
                traj.dt().as_secs_f32() * 1_000.0,
            );
            let step = (traj.len() / 10).max(1);
            for (i, q) in traj.iter().enumerate() {
                if i % step == 0 || i + 1 == traj.len() {
                    println!("  t={:>7.4}s q=[{}]", i as f32 * traj.dt().as_secs_f32(),
                        q.0.iter()
                            .map(|v| format!("{:+.3}", v))
                            .collect::<Vec<_>>()
                            .join(", "));
                }
            }
        }
        Err(e) => eprintln!("retime failed: {}", e),
    }
}
