deke_cricket::cricket!(
    name = "R2000IC270F",
    urdf = "../robot_assets/robots/r2000ic270f/spherized.urdf",
    srdf = "../robot_assets/robots/r2000ic270f/r2000ic270f.srdf",
    end_effector = "flange",
    forced_end_effector_collision = ["base_link", "link_1", "link_2", "link_3"],
    ignored_environment_collision = ["base_link"],
);

fn main() {
    use r2000ic270f::*;
    use deke_types::{FKChain as _, Planner as _, SRobotQ};
    use deke_viz::{LinkMesh, RobotMeshes, affine_from_xyz_rpy};

    let asset_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../robot_assets/robots/r2000ic270f/assets");

    let load_stl = |name: &str| -> Vec<u8> {
        std::fs::read(format!("{asset_dir}/{name}.stl")).expect(&format!("failed to read {name}.stl"))
    };

    let meshes: RobotMeshes<{ DOF }> = RobotMeshes {
        base: Some(LinkMesh {
            stl_data: load_stl("base_link"),
            visual_origin: affine_from_xyz_rpy([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        }),
        links: [
            LinkMesh {
                stl_data: load_stl("link_1"),
                visual_origin: affine_from_xyz_rpy([0.0, 0.0, -0.2605], [0.0, 0.0, 0.0]),
            },
            LinkMesh {
                stl_data: load_stl("link_2"),
                visual_origin: affine_from_xyz_rpy([-0.312, 0.67, -0.108], [1.5708, 0.0, 0.0]),
            },
            LinkMesh {
                stl_data: load_stl("link_3"),
                visual_origin: affine_from_xyz_rpy([-0.312, -1.745, 0.1645], [-1.5708, 0.0, 0.0]),
            },
            LinkMesh {
                stl_data: load_stl("link_4"),
                visual_origin: affine_from_xyz_rpy([0.0, 1.97, 1.7766], [1.5708, 1.5708, 0.0]),
            },
            LinkMesh {
                stl_data: load_stl("link_5"),
                visual_origin: affine_from_xyz_rpy([-2.042, -1.97, -0.0664], [-1.5708, 0.0, 0.0]),
            },
            LinkMesh {
                stl_data: load_stl("link_6"),
                visual_origin: affine_from_xyz_rpy([0.0, 1.97, 2.2234], [1.5708, 1.5708, 0.0]),
            },
        ],
    };

    println!("=== deke K-RRTC planner ===");
    println!("robot: r2000ic270f ({DOF} DOF)");

    let rec = rerun::RecordingStreamBuilder::new("deke-viz-krrtc")
        .spawn()
        .expect("failed to start rerun viewer");

    let pi_2 = std::f32::consts::FRAC_PI_2;
    let start = SRobotQ([pi_2, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let goal = SRobotQ([-pi_2, 0.0, 0.0, 0.0, 0.0, 0.0]);

    println!("start: {:?}", start.0);
    println!("goal:  {:?}", goal.0);

    let mut env = wreck::Collider::default();
    env.add(wreck::Sphere::new(glam::Vec3::new(1.5, 0.0, 1.75), 0.5));
    let mut validator = validator(env);

    println!("environment: 1 sphere at (1.5, 0.0, 1.75) r=0.5");

    deke_viz::log_collider(&rec, "obstacle", validator.1.environment())
        .expect("failed to log obstacle");

    let kin_limits = deke_rrt::KinematicLimits {
        joints: JOINT_VELOCITY.map(|v| {
            let v_max = v as f64;
            deke_rrt::JointKinLimits {
                v_max,
                a_max: v_max * 10.0,
                j_max: v_max * 80.0,
            }
        }),
    };
    println!("kinematic limits (per joint):");
    for (i, jl) in kin_limits.joints.iter().enumerate() {
        println!("  [{}] v_max={:.2} a_max={:.2} j_max={:.2}", i, jl.v_max, jl.a_max, jl.j_max);
    }

    let settings = deke_rrt::KrrtcSettings::new(
        SRobotQ(JOINT_LOWER),
        SRobotQ(JOINT_UPPER),
        kin_limits,
    );
    let planner = krrtc(settings);

    println!("planning...");
    let (result, diag) = planner.plan(start, goal, &mut validator);
    println!("{diag}");

    let path = match result {
        Ok(p) => p,
        Err(e) => {
            eprintln!("planning failed: {e:?}");
            return;
        }
    };

    let fk = deke_types::URDFChain::new(URDF_JOINTS);

    println!("path has {} waypoints:", path.len());
    for (i, q) in path.iter().enumerate() {
        let sq = deke_types::SRobotQ::<{ DOF }>::try_from(q.as_slice().unwrap()).unwrap();
        let tcp = fk.fk(&sq).unwrap()[DOF - 1].translation;
        println!("  [{i}] joints={:?}  tcp=({:.3}, {:.3}, {:.3})", q.as_slice().unwrap(), tcp.x, tcp.y, tcp.z);
    }

    let slow_velocity = JOINT_VELOCITY.map(|v| v * 0.1);
    let seg_times = deke_viz::segment_times(&path, &slow_velocity);
    let total_time: f64 = seg_times.iter().sum();
    println!("estimated playback time: {:.2}s ({} segments)", total_time, seg_times.len());

    println!("logging tcp trace...");
    deke_viz::log_path_tcp::<{ DOF }>(&rec, "path/tcp_trace", &path, &fk)
        .expect("failed to log tcp path");

    println!("logging waypoints...");
    deke_viz::log_waypoints::<{ DOF }>(&rec, "path/waypoints", &path, &fk)
        .expect("failed to log waypoints");

    println!("logging realtime playback...");
    deke_viz::log_path_realtime::<{ DOF }>(
        &rec,
        "robot",
        &path,
        &fk,
        Some(&meshes),
        &slow_velocity,
    )
    .expect("failed to log realtime path");

    println!("done.");
}
