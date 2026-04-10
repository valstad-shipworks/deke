deke_cricket::cricket!(
    name = "R2000IC270F",
    urdf = "../robot_assets/robots/r2000ic270f/spherized.urdf",
    srdf = "../robot_assets/robots/r2000ic270f/r2000ic270f.srdf",
    end_effector = "flange",
    forced_end_effector_collision = ["base_link", "link_1", "link_2", "link_3"],
    ignored_environment_collision = ["base_link"],
);

fn main() {
    use deke_types::{FKChain as _, Planner as _, SRobotQ};
    use deke_viz::{LinkMesh, RobotMeshes, affine_from_xyz_rpy};
    use r2000ic270f::*;

    let asset_dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../robot_assets/robots/r2000ic270f/assets"
    );

    let load_stl = |name: &str| -> Vec<u8> {
        std::fs::read(format!("{asset_dir}/{name}.stl"))
            .expect(&format!("failed to read {name}.stl"))
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

    println!("=== deke AO-RRTC planner ===");
    println!("robot: r2000ic270f ({DOF} DOF)");

    let rec = rerun::RecordingStreamBuilder::new("deke-viz-aorrtc")
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

    let mut settings = deke_rrt::AorrtcSettings::new(SRobotQ(JOINT_LOWER), SRobotQ(JOINT_UPPER));
    settings.rrtc.max_iterations = 100000;
    settings.rrtc.max_samples = 100000;
    settings.rrtc.dynamic_domain = true;
    settings.rrtc.range = 1.8;
    settings.rrtc.balance = false;
    let dof_weights = SRobotQ(std::array::from_fn(|i| {
        let v = JOINT_VELOCITY[i] as f64;
        let a = v * 10.0;
        let j = v * 80.0;
        let w = (100.0 / v.powf(1.2)) + (100.0 / a) + (100.0 / j.powf(0.8));
        let scale = if i > 2 { 0.1 } else { 3.0 };
        (w * scale * 100.0) as f32
    }));
    // settings.rrtc.dof_cost_weights = dof_weights;
    settings.dof_cost_weights = dof_weights;
    settings.max_iterations = 100000;
    settings.max_samples = 100000;
    settings.use_phs = false;
    settings.simplify_bspline_steps = 3;
    settings.simplify_reduce_max_steps = 25;
    settings.simplify_reduce_range_ratio = 0.8;
    settings.static_dof_penalty = 100.0;
    settings.penalize_static_dof = true;
    let planner = aorrtc(settings);

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
        println!(
            "  [{i}] joints={:?}  tcp=({:.3}, {:.3}, {:.3})",
            q.as_slice().unwrap(),
            tcp.x,
            tcp.y,
            tcp.z
        );
    }

    let slow_velocity = JOINT_VELOCITY.map(|v| v * 0.1);
    let seg_times = deke_viz::segment_times(&path, &slow_velocity);
    let total_time: f64 = seg_times.iter().sum();
    println!(
        "estimated playback time: {:.2}s ({} segments)",
        total_time,
        seg_times.len()
    );

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
