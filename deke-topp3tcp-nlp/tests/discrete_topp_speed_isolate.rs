mod common;

use std::time::Duration;

use deke_topp_speed::{MotionSpec, ToppSolver};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
#[ignore]
fn raw_topp_speed_10wp() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([-1.1967357, 0.6513940, 0.0649984, -0.7458407, -1.0254644, 1.9914096]),
        SRobotQ::from_array([-1.4218939, 0.7337620, 0.3250841, -0.5453823, -0.9866293, 1.6930232]),
        SRobotQ::from_array([-1.5209670, 0.7668492, 0.5091862, -0.3223294, -0.8368485, 1.4805338]),
        SRobotQ::from_array([-1.3353859, 0.6944350, 0.5843410, -0.0834244, -0.9241249, 1.1985058]),
        SRobotQ::from_array([-1.5004166, 0.7278752, 0.6759865, -0.2666964, -0.7699144, 1.4176370]),
        SRobotQ::from_array([-1.3280353, 0.5349262, 0.4342674, -0.4279996, -0.9543195, 1.6290450]),
        SRobotQ::from_array([-1.2149905, 0.3504188, 0.7067139, -0.3298242, -0.9069862, 1.6441734]),
        SRobotQ::from_array([-1.2035334, 0.3318515, 1.0019210, -0.2021268, -0.6941859, 1.5207703]),
        SRobotQ::from_array([-1.3864710, 0.3202283, 1.1828311, -0.2501720, -0.8450851, 1.3589037]),
        SRobotQ::from_array([-1.5619611, 0.5108429, 1.3277226, -0.0363543, -0.6535910, 1.3245343]),
    ])
    .unwrap();

    let mut spec = MotionSpec::<6, f64>::new();
    spec.current_pose = *path.first();
    spec.goal_pose = *path.last();
    spec.waypoint_poses.clear();
    for i in 1..path.len() - 1 {
        if let Some(p) = path.get(i) {
            spec.waypoint_poses.push(*p);
        }
    }
    spec.max_vel = SRobotQ::from_array([1.5; 6]);
    spec.max_accel = SRobotQ::from_array([6.0; 6]);
    spec.max_jerk = SRobotQ::from_array([25.0; 6]);
    spec.max_tcp_speed = Some(2.0);

    let solver = ToppSolver::<6, f64, _>::new(Duration::from_secs_f64(0.008), &fk);
    let v = common::wide_validator::<6>();
    let t = std::time::Instant::now();
    eprintln!("starting topp-speed solve");
    let (r, d) = solver.retime(&spec, &path, &v, &());
    eprintln!("topp-speed: {} in {:?}", r.is_ok(), t.elapsed());
    eprintln!("diag: {}", d);
    if let Ok(t) = r {
        eprintln!("duration: {:?}, samples: {}", t.duration(), t.path().len());
    }
}
