mod common;

use deke_topp3tcp_nlp::discrete::{TcpLimits, Topp3Tcp6Discrete, Topp3Tcp6DiscreteConstraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
#[ignore]
fn tcp_full() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]),
        SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6DiscreteConstraints::<6>::symmetric(5.0, 30.0, 3000.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 1.0,
        a_max: 5.0,
        j_max: 50.0,
    });
    let v = common::wide_validator::<6>();
    let t = std::time::Instant::now();
    let (r, d) = Topp3Tcp6Discrete::new(&fk).retime(&cfg, &path, &v, &());
    eprintln!("tcp_full: {:?} in {:?}\n{}", r.is_ok(), t.elapsed(), d);
}

#[test]
#[ignore]
fn tight_jerk() {
    let fk = common::dh_1dof();
    let path =
        SRobotPath::<1, f64>::try_new(vec![SRobotQ::from_array([0.0]), SRobotQ::from_array([1.0])])
            .unwrap();
    let cfg = Topp3Tcp6DiscreteConstraints::<1>::symmetric(1.0, 2.0, 4.0);
    let v = common::wide_validator::<1>();
    let t = std::time::Instant::now();
    let (r, d) = Topp3Tcp6Discrete::new(&fk).retime(&cfg, &path, &v, &());
    eprintln!("tight_jerk: {:?} in {:?}\n{}", r.is_ok(), t.elapsed(), d);
}

#[test]
#[ignore]
fn captured_10wp_no_seed() {
    let fk = common::dh_6dof();
    let path = make_10wp();
    let mut cfg = Topp3Tcp6DiscreteConstraints::<6>::symmetric(1.5, 6.0, 25.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 2.0,
        a_max: 20.0,
        j_max: 200.0,
    });
    cfg.solver.seed_from_topp_speed = false;
    let v = common::wide_validator::<6>();
    let t = std::time::Instant::now();
    let (r, d) = Topp3Tcp6Discrete::new(&fk).retime(&cfg, &path, &v, &());
    eprintln!(
        "captured_10wp_no_seed: {:?} in {:?}\n{}",
        r.is_ok(),
        t.elapsed(),
        d
    );
}

fn make_10wp() -> SRobotPath<6, f64> {
    SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([
            -1.1967357, 0.6513940, 0.0649984, -0.7458407, -1.0254644, 1.9914096,
        ]),
        SRobotQ::from_array([
            -1.4218939, 0.7337620, 0.3250841, -0.5453823, -0.9866293, 1.6930232,
        ]),
        SRobotQ::from_array([
            -1.5209670, 0.7668492, 0.5091862, -0.3223294, -0.8368485, 1.4805338,
        ]),
        SRobotQ::from_array([
            -1.3353859, 0.6944350, 0.5843410, -0.0834244, -0.9241249, 1.1985058,
        ]),
        SRobotQ::from_array([
            -1.5004166, 0.7278752, 0.6759865, -0.2666964, -0.7699144, 1.4176370,
        ]),
        SRobotQ::from_array([
            -1.3280353, 0.5349262, 0.4342674, -0.4279996, -0.9543195, 1.6290450,
        ]),
        SRobotQ::from_array([
            -1.2149905, 0.3504188, 0.7067139, -0.3298242, -0.9069862, 1.6441734,
        ]),
        SRobotQ::from_array([
            -1.2035334, 0.3318515, 1.0019210, -0.2021268, -0.6941859, 1.5207703,
        ]),
        SRobotQ::from_array([
            -1.3864710, 0.3202283, 1.1828311, -0.2501720, -0.8450851, 1.3589037,
        ]),
        SRobotQ::from_array([
            -1.5619611, 0.5108429, 1.3277226, -0.0363543, -0.6535910, 1.3245343,
        ]),
    ])
    .unwrap()
}

#[test]
#[ignore]
fn captured_10wp_with_seed() {
    let fk = common::dh_6dof();
    let path = make_10wp();
    let mut cfg = Topp3Tcp6DiscreteConstraints::<6>::symmetric(1.5, 6.0, 25.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 2.0,
        a_max: 20.0,
        j_max: 200.0,
    });
    cfg.solver.seed_from_topp_speed = true;
    let v = common::wide_validator::<6>();
    let t = std::time::Instant::now();
    let (r, d) = Topp3Tcp6Discrete::new(&fk).retime(&cfg, &path, &v, &());
    eprintln!(
        "captured_10wp_with_seed: {:?} in {:?}\n{}",
        r.is_ok(),
        t.elapsed(),
        d
    );
}
