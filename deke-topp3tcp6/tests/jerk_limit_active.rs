mod common;

use deke_topp3tcp6::{SolveStatus, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
fn tight_jerk_limit_increases_total_time() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([1.0]),
    ])
    .unwrap();

    let mut validator = common::wide_validator::<1>();

    let loose = Topp3Tcp6Constraints::<1>::symmetric(1.0, 2.0, 500.0);
    let (r1, d1) = Topp3Tcp6.retime(&loose, &path, &fk, &mut validator, &());
    let t1 = r1.expect("loose-jerk retime failed").duration().as_secs_f64();
    eprintln!("loose jerk:\n{}", d1);

    let tight = Topp3Tcp6Constraints::<1>::symmetric(1.0, 2.0, 4.0);
    let (r2, d2) = Topp3Tcp6.retime(&tight, &path, &fk, &mut validator, &());
    let t2 = r2.expect("tight-jerk retime failed").duration().as_secs_f64();
    eprintln!("tight jerk:\n{}", d2);

    assert_eq!(d1.status, SolveStatus::Success);
    assert_eq!(d2.status, SolveStatus::Success);
    assert!(
        t2 > t1 + 0.02,
        "tight jerk ({}) should take meaningfully longer than loose ({})",
        t2,
        t1
    );
}
