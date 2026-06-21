mod common;

use deke_linear::LinearFollower;

#[test]
fn shallow_corner_stays_one_run_and_keeps_moving() {
    let robot = common::ur();
    let poses = common::corner(&robot, 0.06, 20f64.to_radians(), 4);
    let cfg = common::config(0.04);

    let follower = LinearFollower::new(&robot);
    let (traj, diag) = follower.follow(&poses, &cfg, &common::noop(), &()).expect("follow failed");

    assert_eq!(diag.runs, 1, "a 20° corner is below the sharp threshold → one run");

    let speeds = common::tcp_speeds(&robot, &traj);
    let lo = speeds.len() / 5;
    let hi = speeds.len() * 4 / 5;
    let min_interior = speeds[lo..hi].iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        min_interior > 1e-3,
        "shallow corner should slow but never stop (min {min_interior})"
    );
    println!(
        "shallow: runs={} samples={} vmax={:.3} amax={:.1} min_interior={:.4}",
        diag.runs,
        traj.path().len(),
        common::joint_vel_peak(&traj),
        common::joint_acc_peak(&traj),
        min_interior
    );
}

#[test]
fn sharp_corner_splits_and_stops_at_the_vertex() {
    let robot = common::ur();
    let poses = common::corner(&robot, 0.06, 90f64.to_radians(), 4);
    let cfg = common::config(0.04);

    let follower = LinearFollower::new(&robot);
    let (traj, diag) = follower.follow(&poses, &cfg, &common::noop(), &()).expect("follow failed");

    assert_eq!(diag.runs, 2, "a 90° corner is sharp → two runs");

    let speeds = common::tcp_speeds(&robot, &traj);
    let lo = speeds.len() / 4;
    let hi = speeds.len() * 3 / 4;
    let min_mid = speeds[lo..hi].iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        min_mid < 0.04 * 0.5,
        "sharp corner should decelerate to ~rest at the vertex (min {min_mid})"
    );
    assert!(
        common::joint_vel_peak(&traj) <= 2.0 * 1.05,
        "joint velocity exceeded limit"
    );
    println!(
        "sharp: runs={} samples={} vmax={:.3} amax={:.1} min_mid={:.4}",
        diag.runs,
        traj.path().len(),
        common::joint_vel_peak(&traj),
        common::joint_acc_peak(&traj),
        min_mid
    );
}
