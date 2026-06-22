mod common;

use std::time::Duration;

use common::Cfg;
use deke_kin::{DHJoint, JointLimits as KinJointLimits, Kinematics};
use deke_linear::{JointLimits, RedundantAxis, RedundantOptions};
use deke_types::glam::{DAffine3, DMat3, DVec3};
use deke_types::{IkOutcome, IkSolver};

/// UR10-ish, but with realistic non-wrapping joint limits (±150°) so that tool
/// orientation actually constrains reachability — the regime where the free
/// tool-axis yaw matters.
fn ur_tight() -> Kinematics<6, f64> {
    use std::f64::consts::PI;
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
        KinJointLimits::symmetric(150.0_f64.to_radians()),
        &[],
    )
}

/// Orthonormal frame whose local +Z points along `z`.
fn frame_from_z(z: DVec3) -> DMat3 {
    let z = z.normalize();
    let up = if z.dot(DVec3::Z).abs() > 0.95 {
        DVec3::X
    } else {
        DVec3::Z
    };
    let x = up.cross(z).normalize();
    let y = z.cross(x);
    DMat3::from_cols(x, y, z)
}

fn ik_reachable(robot: &Kinematics<6, f64>, pose: DAffine3) -> bool {
    matches!(robot.ik(pose), Ok(IkOutcome::Solved(s)) if !s.is_empty())
}

#[test]
fn free_yaw_succeeds_where_fixed_orientation_fails() {
    let robot = ur_tight();

    // Find a near-reach position + tool-Z direction where the nominal (yaw = 0)
    // orientation is unreachable, but some yaw about the tool axis is reachable.
    let radii = [1.0, 1.1, 1.2];
    let thetas = (0..12).map(|k| k as f64 * std::f64::consts::TAU / 12.0);
    let zs = [-0.1, 0.2, 0.5];

    let mut found = None;
    'search: for r in radii {
        for theta in thetas.clone() {
            for z in zs {
                let p = DVec3::new(r * theta.cos(), r * theta.sin(), z);
                // Tool pointing roughly back toward the base axis (a plausible weld attitude).
                for tilt in [0.0_f64, 0.5, 1.0] {
                    let zdir =
                        (DVec3::new(-p.x, -p.y, 0.0).normalize() + DVec3::Z * tilt).normalize();
                    let r0 = frame_from_z(zdir);
                    let nominal = DAffine3::from_mat3_translation(r0, p);
                    if ik_reachable(&robot, nominal) {
                        continue; // yaw=0 already works → not a discriminator
                    }
                    // Sweep yaw about the tool Z.
                    let good_yaw = (1..36)
                        .map(|m| m as f64 * std::f64::consts::TAU / 36.0)
                        .find(|&psi| {
                            let rot = DMat3::from_cols(r0.x_axis, r0.y_axis, r0.z_axis)
                                * DMat3::from_rotation_z(psi);
                            ik_reachable(&robot, DAffine3::from_mat3_translation(rot, p))
                        });
                    if let Some(psi) = good_yaw {
                        found = Some((p, r0, psi));
                        break 'search;
                    }
                }
            }
        }
    }

    let (p, r0, good_psi) = found.expect("no near-reach orientation-limited pose found");
    println!(
        "DISCRIMINATOR pose at p={:?}, nominal yaw unreachable, yaw {:.0}° reachable",
        p,
        good_psi.to_degrees()
    );

    // Build a short weld move through that pose (1 cm along the frame's X), fixed
    // nominal orientation.
    let tangent = r0.x_axis;
    let poses: Vec<DAffine3> = (0..5)
        .map(|i| {
            let f = (i as f64 / 4.0 - 0.5) * 0.01;
            DAffine3::from_mat3_translation(r0, p + tangent * f)
        })
        .collect();

    let joint = JointLimits::symmetric(2.0, 8.0, 80.0);
    let dt = Duration::from_millis(8);

    let fixed = Cfg::weld(30.0, joint.clone(), dt);
    let redundant = fixed.clone().with_redundancy(RedundantOptions {
        axis: RedundantAxis::PosZ,
        yaw_window: (-std::f64::consts::PI, std::f64::consts::PI),
        yaw_samples: 36,
        ..RedundantOptions::default()
    });

    let fixed_res = common::follow(&robot, &poses, &fixed, &common::noop(), &());
    assert!(
        fixed_res.is_err(),
        "fixed-orientation planner should fail on an orientation-unreachable weld"
    );
    println!("  fixed planner: {:?}", fixed_res.unwrap_err());

    let (traj, diag) = common::follow(&robot, &poses, &redundant, &common::noop(), &())
        .expect("free-yaw planner should solve it");
    let rd = &diag.redundant[0];
    println!(
        "  free-yaw planner: ok, {} samples, min_manip {:.3e}, yaw ∈ [{:.0}°,{:.0}°]",
        traj.path().len(),
        rd.min_manipulability,
        rd.yaw_range.0.to_degrees(),
        rd.yaw_range.1.to_degrees()
    );
    assert!(rd.min_manipulability > 0.0);
}
