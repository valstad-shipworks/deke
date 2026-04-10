deke_cricket::cricket!(
    name = "R2000IC270F",
    urdf = "../robot_assets/robots/r2000ic270f/spherized.urdf",
    srdf = "../robot_assets/robots/r2000ic270f/r2000ic270f.srdf",
    end_effector = "flange",
    forced_end_effector_collision = ["base_link", "link_1", "link_2", "link_3"],
    ignored_environment_collision = ["base_link"],
);

fn main() {
    use deke_types::FKChain as _;
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

    println!("=== vamp AO-RRTC planner ===");
    println!("robot: r2000ic270f ({DOF} DOF)");

    let rec = rerun::RecordingStreamBuilder::new("deke-viz-vamp-aorrtc")
        .spawn()
        .expect("failed to start rerun viewer");

    let pi_2 = std::f32::consts::FRAC_PI_2;
    let start = [pi_2, 0.0, 0.0, 0.0, 0.0, 0.0];
    let goal = [-pi_2, 0.0, 0.0, 0.0, 0.0, 0.0];

    println!("start: {:?}", start);
    println!("goal:  {:?}", goal);

    let mut vamp_env = vamp::Environment::new();
    vamp_env.add_sphere(1.5, 0.0, 1.75, 0.5);

    let mut wreck_env = wreck::Collider::default();
    wreck_env.add(wreck::Sphere::new(glam::Vec3::new(1.5, 0.0, 1.75), 0.5));

    println!("environment: 1 sphere at (1.5, 0.0, 1.75) r=0.5");

    let robot = vamp::Robot::R2000IC270F;
    let mut rng = vamp::Rng::halton(robot);
    let mut settings = vamp::AORRTCSettings::default();
    settings.rrtc_max_iterations = 400000;
    settings.rrtc_max_samples = 200000;
    settings.rrtc_dynamic_domain = true;
    settings.rrtc_range = 1.8;
    settings.rrtc_balance = false;
    settings.max_iterations = 400000;
    settings.optimize = true;
    settings.max_samples = 200000;
    settings.simplify_intermediate = true;
    settings.cost_bound_resample = true;
    settings.max_internal_iterations = 5000000;
    settings.use_phs = false;
    // settings.dof_cost_weights = std::array::from_fn(|i| {
    //     let v = JOINT_VELOCITY[i] as f64;
    //     let a = v * 10.0;
    //     let j = v * 80.0;
    //     let w = (100.0 / v.powf(1.2)) + (100.0 / a) + (100.0 / j.powf(0.8));
    //     let scale = if i > 2 { 0.1 } else { 3.0 };
    //     (w * scale) as f32
    // });
    println!("dof_cost_weights: {:?}", settings.dof_cost_weights);

    println!("planning...");
    let result = robot.plan_aorrtc(&start, &goal, &vamp_env, &settings, &mut rng);

    let vamp_path = match result {
        Ok(success) => {
            println!(
                "planning succeeded: cost={:.4}, iterations={}, time={:.2}ms, {} waypoints",
                success.cost,
                success.iterations,
                success.nanoseconds as f64 / 1_000_000.0,
                success.path.len(),
            );
            success.path
        }
        Err(failure) => {
            eprintln!(
                "planning failed after {} iterations ({:.2}ms)",
                failure.iterations,
                failure.nanoseconds as f64 / 1_000_000.0
            );
            return;
        }
    };

    println!("simplifying (shortcut -> bspline -> shortcut -> reduce)...");
    let mut simplify_settings = vamp::SimplifySettings::default();
    simplify_settings.operations = vec![
        vamp::SimplifyRoutine::Shortcut,
        vamp::SimplifyRoutine::BSpline,
        vamp::SimplifyRoutine::Shortcut,
        vamp::SimplifyRoutine::Reduce,
    ];
    simplify_settings.bspline_max_steps = 3;
    simplify_settings.reduce_max_steps = 25;
    simplify_settings.reduce_range_ratio = 0.8;
    rng.reset();
    let vamp_path = match vamp_path.simplify(&vamp_env, &simplify_settings, &mut rng) {
        Ok(success) => {
            println!(
                "simplification: cost={:.4}, iterations={}, time={:.2}ms, {} waypoints",
                success.cost,
                success.iterations,
                success.nanoseconds as f64 / 1_000_000.0,
                success.path.len(),
            );
            success.path
        }
        Err(_) => {
            eprintln!("simplification failed, using raw path");
            vamp_path
        }
    };

    let waypoints: Vec<deke_types::SRobotQ<{ DOF }>> = vamp_path
        .iter()
        .map(|cfg| {
            let mut arr = [0.0f32; DOF];
            arr.copy_from_slice(&cfg[..DOF]);
            deke_types::SRobotQ(arr)
        })
        .collect();
    let path = deke_types::SRobotPath::new(waypoints).expect("invalid path");

    let wreck_validator = validator(wreck_env);
    let fk = deke_types::URDFChain::new(URDF_JOINTS);

    println!("path has {} waypoints:", path.len());
    for (i, sq) in path.iter().enumerate() {
        let tcp = fk.fk(sq).unwrap()[DOF - 1].translation;
        println!(
            "  [{i}] joints={:?}  tcp=({:.3}, {:.3}, {:.3})",
            sq.0,
            tcp.x,
            tcp.y,
            tcp.z
        );
    }

    let slow_velocity = JOINT_VELOCITY.map(|v| v * 0.1);
    let rp = path.to_robot_path();
    let seg_times = deke_viz::segment_times(&rp, &slow_velocity);
    let total_time: f64 = seg_times.iter().sum();
    println!(
        "estimated playback time: {:.2}s ({} segments)",
        total_time,
        seg_times.len()
    );

    deke_viz::log_collider(&rec, "obstacle", wreck_validator.1.environment())
        .expect("failed to log obstacle");

    println!("logging tcp trace...");
    deke_viz::log_path_tcp::<{ DOF }>(&rec, "path/tcp_trace", &rp, &fk)
        .expect("failed to log tcp path");

    println!("logging waypoints...");
    deke_viz::log_waypoints::<{ DOF }>(&rec, "path/waypoints", &rp, &fk)
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
