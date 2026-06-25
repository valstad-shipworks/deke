use std::f64::consts::PI;
use std::hint::black_box;
use std::time::{Duration, Instant};

use deke_kin::{DHJoint, JointLimits as KinJointLimits, Kinematics};
use deke_topp3_lp::{Topp3Lp, Topp3LpConstraints, Topp3LpTcp};
use deke_types::{JointValidator, Retimer, SRobotPath, SRobotQ};

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

fn bench(name: &str, iters: u32, mut f: impl FnMut() -> usize) {
    let samples = f(); // warm up + capture output size
    for _ in 0..3 {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    let per = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("{name:<42} {per:7.3} ms/call   ({samples} samples)");
}

fn main() {
    let dt8 = Duration::from_millis(8); // 125 Hz
    let dt1 = Duration::from_millis(1); // 1 kHz

    let r6 = Topp3Lp::<6>::new();
    let r1 = Topp3Lp::<1>::new();
    let v6 = wide::<6>();
    let v1 = wide::<1>();
    let fk = ur();
    let rt = Topp3LpTcp::new(&fk);

    let one = path::<1>(&[[0.0], [1.0]]);
    let straight = path::<6>(&[
        [0.0, -1.2, 1.5, -0.3, 0.5, 0.0],
        [0.6, -0.6, 0.9, 0.3, -0.2, 0.8],
    ]);
    let curved = path::<6>(&[
        [0.0, -1.3, 1.5, 0.0, 0.0, 0.0],
        [0.2, -1.1, 1.3, -0.1, 0.1, 0.1],
        [0.4, -0.9, 1.1, -0.2, 0.2, 0.2],
        [0.6, -0.7, 0.9, -0.3, 0.1, 0.3],
        [0.8, -0.5, 0.7, -0.4, 0.0, 0.4],
    ]);
    let corner = path::<6>(&[
        [0.0, -1.0, 1.2, 0.0, 0.0, 0.0],
        [0.3, -1.0, 1.2, 0.0, 0.0, 0.0],
        [0.3, -1.0, 1.2, 0.3, 0.0, 0.0],
        [0.3, -1.0, 1.2, 0.3, 0.3, 0.0],
    ]);

    let c8 = |dt| Topp3LpConstraints::<6>::symmetric(1.5, 8.0, 400.0, dt);
    let c1 = |dt| Topp3LpConstraints::<1>::symmetric(1.0, 2.0, 200.0, dt);
    let ctcp = Topp3LpConstraints::<6>::symmetric(1.5, 8.0, 400.0, dt8).with_tcp_speed(0.3);

    println!("deke-topp3-lp retime() — release, single-threaded\n");
    bench("1-DOF rest-to-rest          @125Hz", 2000, || {
        r1.retime(&c1(dt8), &one, &v1, &()).0.unwrap().len()
    });
    bench("6-DOF straight (joint)      @125Hz", 2000, || {
        r6.retime(&c8(dt8), &straight, &v6, &()).0.unwrap().len()
    });
    bench("6-DOF curved 5wp (joint)    @125Hz", 1000, || {
        r6.retime(&c8(dt8), &curved, &v6, &()).0.unwrap().len()
    });
    bench("6-DOF sharp corner (split)  @125Hz", 1000, || {
        r6.retime(&c8(dt8), &corner, &v6, &()).0.unwrap().len()
    });
    bench("6-DOF curved + TCP cap      @125Hz", 1000, || {
        rt.retime(&ctcp, &curved, &v6, &()).0.unwrap().len()
    });
    bench("6-DOF curved 5wp (joint)    @1kHz ", 500, || {
        r6.retime(&c8(dt1), &curved, &v6, &()).0.unwrap().len()
    });
}
