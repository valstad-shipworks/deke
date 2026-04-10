deke_cricket::cricket!(
    name = "M20ID12L",
    urdf = "../robot_assets/robots/m20id12l/spherized.urdf",
    srdf = "../robot_assets/robots/m20id12l/m20id12l.srdf",
    end_effector = "flange",
    forced_end_effector_collision = ["base_link", "link_1", "link_2", "link_3", "link_4"],
    ignored_environment_collision = ["base_link"],
);

#[cfg(test)]
mod tests {
    use super::m20id12l::*;
    use deke_types::Validator as _;

    #[test]
    fn smoke() {
        assert_eq!(DOF, 6);
        assert_eq!(END_EFFECTOR, "flange");
        println!("lower: {:?}", JOINT_LOWER);
        println!("upper: {:?}", JOINT_UPPER);

        let env = wreck::Collider::default();
        let _v = validator(env);

        let rrtc_settings = deke_rrt::RrtcSettings::new(
            deke_types::SRobotQ(JOINT_LOWER),
            deke_types::SRobotQ(JOINT_UPPER),
        );
        let _p = rrtc(rrtc_settings);

        let aorrtc_settings = deke_rrt::AorrtcSettings::new(
            deke_types::SRobotQ(JOINT_LOWER),
            deke_types::SRobotQ(JOINT_UPPER),
        );
        let _p = aorrtc(aorrtc_settings);
    }

    #[test]
    fn validate_matches_vamp() {
        let env = wreck::Collider::default();
        let mut deke_validator = validator(env);
        let vamp_env = vamp::Environment::new();
        let vamp_robot = vamp::Robot::M20ID12L;

        let mut rng = 0x12345678u64;
        let mut mismatches = 0;
        let mut deke_extra = 0u32;
        let mut vamp_extra = 0u32;
        let mut deke_extra_reasons: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();

        for _ in 0..1000 {
            let mut q = [0.0f32; 6];
            for j in 0..6 {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                q[j] = ((rng >> 33) as f32 / (1u64 << 31) as f32) * 8.0 - 4.0;
            }

            let deke_result = deke_validator.validate_motion(&[deke_types::SRobotQ(q)]);
            let deke_ok = deke_result.is_ok();
            let vamp_ok = vamp_robot.validate(&q, &vamp_env, true);

            if deke_ok != vamp_ok {
                mismatches += 1;
                if !deke_ok && vamp_ok {
                    deke_extra += 1;
                    let reason = format!("{:?}", deke_result.unwrap_err());
                    *deke_extra_reasons.entry(reason).or_default() += 1;
                } else {
                    vamp_extra += 1;
                }
            }
        }

        println!("{mismatches} mismatches out of 1000");
        println!("  deke_only (deke=err, vamp=ok): {deke_extra}");
        println!("  vamp_only   (deke=ok, vamp=err): {vamp_extra}");
        if !deke_extra_reasons.is_empty() {
            println!("  deke collision reasons:");
            let mut reasons: Vec<_> = deke_extra_reasons.iter().collect();
            reasons.sort_by(|a, b| b.1.cmp(a.1));
            for (reason, count) in &reasons {
                println!("    {count:>4}x {reason}");
            }
        }

        use deke_types::FKChain;
        let fk = deke_types::URDFChain::<6>::new(super::m20id12l::URDF_JOINTS);
        let test_configs: &[[f32; 6]] = &[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.3, -0.2, 0.1, 0.4, -0.3],
            [1.0, -0.5, 1.0, 0.5, -0.5, 1.0],
        ];
        println!("\nFK comparison:");
        for q in test_configs {
            let transforms = fk.fk(&deke_types::SRobotQ(*q)).unwrap();
            let vamp_ee = vamp_robot.eefk(q);
            let dx = transforms[5].translation.x - vamp_ee[12];
            let dy = transforms[5].translation.y - vamp_ee[13];
            let dz = transforms[5].translation.z - vamp_ee[14];
            let err = (dx * dx + dy * dy + dz * dz).sqrt();
            println!("  q={:.2?} err={err:.6}m", q);
        }

        assert!(
            mismatches < 100,
            "{mismatches} mismatches out of 1000 exceeds tolerance",
        );
    }

    #[test]
    fn validate_position() {
        let q: [f32; 6] = [
            -0.9705478549003601,
            0.7127375602722168,
            -0.15622738003730774,
            1.0232568979263306,
            -1.1581602096557617,
            -1.4832807779312134,
        ];

        let env = wreck::Collider::default();
        let mut deke_validator = validator(env);
        let deke_result = deke_validator.validate(deke_types::SRobotQ(q));
        assert!(
            deke_result.is_ok(),
            "deke rejected position: {:?}",
            deke_result.unwrap_err()
        );

        let vamp_env = vamp::Environment::new();
        let vamp_robot = vamp::Robot::M20ID12L;
        assert!(
            vamp_robot.validate(&q, &vamp_env, true),
            "vamp rejected position"
        );
    }

    #[test]
    fn plan_rrtc_bench_problems() {
        use deke_types::Planner as _;

        const PROBLEMS: [([f32; 6], [f32; 6]); 5] = [
            (
                [0.0, 0.5, -0.5, 0.0, 0.5, 0.0],
                [1.5, -0.3, 0.8, 1.0, -0.5, 1.5],
            ),
            (
                [-1.0, 0.2, -0.3, 0.5, 0.1, -0.5],
                [1.5, -0.8, 0.4, -1.0, 0.8, 2.0],
            ),
            (
                [0.3, -0.2, 0.7, -0.3, 0.4, 0.1],
                [-0.8, 0.9, -0.4, 0.8, -0.6, 1.2],
            ),
            (
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.5, -0.5, 1.0, -1.0, 1.0],
            ),
            (
                [0.2, 0.8, -0.6, 0.3, -0.2, 0.5],
                [1.5, -0.6, 0.3, -0.7, 0.9, -1.2],
            ),
        ];

        let vamp_lower: [f32; 6] = [-1.6, -2.7925301, -1.6, -1.8, -3.9269900, -6.2831898];
        let vamp_upper: [f32; 6] = [1.6, 1.7453299, 1.6, 3.4906600, 3.9269900, 6.2831898];

        let mut wreck_env = wreck::Collider::default();
        let mut vamp_env = vamp::Environment::new();
        for &(x, y, z, r) in &[
            (0.5f32, 0.0, 0.5, 0.15),
            (-0.3, 0.4, 0.3, 0.12),
            (0.0, -0.5, 0.6, 0.10),
        ] {
            wreck_env.add(wreck::Sphere::new(glam::Vec3::new(x, y, z), r));
            vamp_env.add_sphere(x, y, z, r);
        }

        let v = deke_types::ValidatorAnd(
            deke_types::JointValidator::new(
                deke_types::SRobotQ(vamp_lower),
                deke_types::SRobotQ(vamp_upper),
            ),
            validator(wreck_env).1,
        );
        let vamp_robot = vamp::Robot::M20ID12L;

        let rrtc_settings = deke_rrt::RrtcSettings::new(
            deke_types::SRobotQ(vamp_lower),
            deke_types::SRobotQ(vamp_upper),
        );
        let planner = rrtc(rrtc_settings);
        let vamp_settings = vamp::RRTCSettings::default();

        println!(
            "\n{:<8} {:>6} {:>6} {:>8} {:>8}",
            "", "r_wps", "v_wps", "r_cost", "v_cost"
        );

        for (i, &(start, goal)) in PROBLEMS.iter().enumerate() {
            let mut rv = v.clone();
            let (r_result, _) = planner.plan(
                deke_types::SRobotQ(start),
                deke_types::SRobotQ(goal),
                &mut rv,
            );

            let mut rng = vamp::Rng::halton(vamp_robot);
            let v_result = vamp_robot.plan_rrtc(&start, &goal, &vamp_env, &vamp_settings, &mut rng);

            let r_ok = r_result.is_ok();
            let v_ok = v_result.is_ok();

            let (r_wps, r_cost) = match &r_result {
                Ok(path) => (path.len(), path.arc_length()),
                Err(_) => (0, 0.0),
            };
            let (v_wps, v_cost) = match &v_result {
                Ok(s) => (s.path.len(), s.cost),
                Err(_) => (0, 0.0),
            };

            println!(
                "prob {i}   {:>6} {:>6} {:>8.3} {:>8.3}  {}",
                r_wps,
                v_wps,
                r_cost,
                v_cost,
                match (r_ok, v_ok) {
                    (true, true) => "both ok",
                    (true, false) => "vamp FAIL",
                    (false, true) => "deke FAIL",
                    (false, false) => "BOTH FAIL",
                }
            );

            assert_eq!(
                r_ok, v_ok,
                "prob {i}: success mismatch (deke={r_ok}, vamp={v_ok})"
            );

            if r_ok && v_ok {
                let r_path = r_result.unwrap();
                let r_start: [f32; 6] = r_path[0].into();
                let r_goal: [f32; 6] = r_path[r_path.len() - 1].into();

                assert!(
                    r_start
                        .iter()
                        .zip(start.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-3),
                    "prob {i}: deke start mismatch"
                );
                assert!(
                    r_goal
                        .iter()
                        .zip(goal.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-3),
                    "prob {i}: deke goal mismatch"
                );

                let straight_line: f32 = start
                    .iter()
                    .zip(goal.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    r_cost >= straight_line * 0.99,
                    "prob {i}: deke cost {r_cost} below straight line {straight_line}"
                );

                let dense = r_path.densify(0.02);
                let mut check_v = v.clone();
                for (wi, sq) in dense.iter().enumerate() {
                    let rv_ok = check_v.validate(*sq);
                    assert!(
                        rv_ok.is_ok(),
                        "prob {i}: deke path waypoint {wi}/{} collides (deke): {:?}",
                        dense.len(),
                        rv_ok.unwrap_err()
                    );
                    let arr: [f32; 6] = (*sq).into();
                    assert!(
                        vamp_robot.validate(&arr, &vamp_env, true),
                        "prob {i}: deke path waypoint {wi}/{} rejected by vamp: {:?}",
                        dense.len(),
                        vamp_robot.validate_with_reason(&arr, &vamp_env, true)
                    );
                }
            }
        }
    }

    #[test]
    fn plan_aorrtc_smoke() {
        use deke_types::Planner as _;

        let start = [-1.0f32, 0.2, -0.3, 0.5, 0.1, -0.5];
        let goal = [1.5, -0.8, 0.4, -1.0, 0.8, 2.0];

        let mut env = wreck::Collider::default();
        for &(x, y, z, r) in &[
            (0.5f32, 0.0, 0.5, 0.15),
            (-0.3, 0.4, 0.3, 0.12),
            (0.0, -0.5, 0.6, 0.10),
        ] {
            env.add(wreck::Sphere::new(glam::Vec3::new(x, y, z), r));
        }
        let v = validator(env);

        let mut aorrtc_settings = deke_rrt::AorrtcSettings::new(
            deke_types::SRobotQ(JOINT_LOWER),
            deke_types::SRobotQ(JOINT_UPPER),
        );
        aorrtc_settings.max_iterations = 100;
        aorrtc_settings.max_samples = 10_000;
        let planner = aorrtc(aorrtc_settings);

        let t0 = std::time::Instant::now();
        let mut rv = v.clone();
        let (result, diag) = planner.plan(
            deke_types::SRobotQ(start),
            deke_types::SRobotQ(goal),
            &mut rv,
        );
        let elapsed = t0.elapsed();

        println!("aorrtc: {elapsed:?}  diag={diag}");
        match &result {
            Ok(path) => println!(
                "  path: {} waypoints, cost={:.3}",
                path.len(),
                path.arc_length()
            ),
            Err(e) => println!("  failed: {e:?}"),
        }

        assert!(elapsed.as_secs() < 10, "aorrtc took too long: {elapsed:?}");
        assert!(result.is_ok(), "aorrtc failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn plan_krrtc_nanopanel_welder() {
        use deke_types::Planner as _;

        // From robot_assets/robots/m20id12l/fanuc_motion_limits.json
        // Middle value (index 10) of each joint's no_payload field, converted deg -> rad.
        let kin_limits = deke_rrt::KinematicLimits {
            joints: [
                deke_rrt::JointKinLimits {
                    v_max: 210.0_f64.to_radians(),
                    a_max: 605.7692_f64.to_radians(),
                    j_max: 3494.8225_f64.to_radians(),
                },
                deke_rrt::JointKinLimits {
                    v_max: 210.0_f64.to_radians(),
                    a_max: 605.7692_f64.to_radians(),
                    j_max: 3494.8225_f64.to_radians(),
                },
                deke_rrt::JointKinLimits {
                    v_max: 265.0_f64.to_radians(),
                    a_max: 764.4231_f64.to_radians(),
                    j_max: 4410.1333_f64.to_radians(),
                },
                deke_rrt::JointKinLimits {
                    v_max: 420.0_f64.to_radians(),
                    a_max: 1211.5385_f64.to_radians(),
                    j_max: 6989.645_f64.to_radians(),
                },
                deke_rrt::JointKinLimits {
                    v_max: 450.0_f64.to_radians(),
                    a_max: 1298.0769_f64.to_radians(),
                    j_max: 7488.906_f64.to_radians(),
                },
                deke_rrt::JointKinLimits {
                    v_max: 720.0_f64.to_radians(),
                    a_max: 2076.923_f64.to_radians(),
                    j_max: 11982.249_f64.to_radians(),
                },
            ],
        };

        const PROBLEMS: [([f32; 6], [f32; 6]); 5] = [
            (
                [0.0, 0.5, -0.5, 0.0, 0.5, 0.0],
                [1.5, -0.3, 0.8, 1.0, -0.5, 1.5],
            ),
            (
                [-1.0, 0.2, -0.3, 0.5, 0.1, -0.5],
                [1.5, -0.8, 0.4, -1.0, 0.8, 2.0],
            ),
            (
                [0.3, -0.2, 0.7, -0.3, 0.4, 0.1],
                [-0.8, 0.9, -0.4, 0.8, -0.6, 1.2],
            ),
            (
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.5, -0.5, 1.0, -1.0, 1.0],
            ),
            (
                [0.2, 0.8, -0.6, 0.3, -0.2, 0.5],
                [1.5, -0.6, 0.3, -0.7, 0.9, -1.2],
            ),
        ];

        let mut env = wreck::Collider::default();
        for &(x, y, z, r) in &[
            (0.5f32, 0.0, 0.5, 0.15),
            (-0.3, 0.4, 0.3, 0.12),
            (0.0, -0.5, 0.6, 0.10),
        ] {
            env.add(wreck::Sphere::new(glam::Vec3::new(x, y, z), r));
        }

        let v = validator(env);

        let settings = deke_rrt::KrrtcSettings::new(
            deke_types::SRobotQ(JOINT_LOWER),
            deke_types::SRobotQ(JOINT_UPPER),
            kin_limits,
        );
        let planner = krrtc(settings);

        println!(
            "\n{:<8} {:>6} {:>10} {:>10}",
            "", "wps", "cost(s)", "time(ms)"
        );

        let mut solved = 0;
        for (i, &(start, goal)) in PROBLEMS.iter().enumerate() {
            let mut rv = v.clone();
            let (result, diag) = planner.plan(
                deke_types::SRobotQ(start),
                deke_types::SRobotQ(goal),
                &mut rv,
            );

            let (wps, cost) = match &result {
                Ok(path) => (path.len(), diag.path_cost),
                Err(_) => (0, 0.0),
            };

            println!(
                "prob {i}   {:>6} {:>10.4} {:>10.3}  {}",
                wps,
                cost,
                diag.elapsed_ns as f64 / 1e6,
                if result.is_ok() { "ok" } else { "FAIL" },
            );

            if result.is_err() {
                continue;
            }
            solved += 1;

            let path = result.unwrap();
            let r_start: [f32; 6] = path[0].into();
            let r_goal: [f32; 6] = path[path.len() - 1].into();

            assert!(
                r_start
                    .iter()
                    .zip(start.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-3),
                "prob {i}: start mismatch",
            );
            assert!(
                r_goal
                    .iter()
                    .zip(goal.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-3),
                "prob {i}: goal mismatch",
            );

            assert!(cost > 0.0, "prob {i}: path cost should be positive");
        }

        assert!(
            solved >= 3,
            "expected at least 3 solved problems, got {solved}"
        );
    }

    #[test]
    fn plan_rrtc_vs_vamp() {
        use deke_types::Planner as _;

        let wreck_env = wreck::Collider::default();
        let vamp_env = vamp::Environment::new();

        let mut rv_check = validator(wreck_env.clone());
        let vamp_robot = vamp::Robot::M20ID12L;

        let mut rng = 0xABCD1234u64;
        let mut valid_configs = Vec::new();
        while valid_configs.len() < 20 {
            let mut q = [0.0f32; 6];
            for j in 0..6 {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let t = (rng >> 33) as f32 / (1u64 << 31) as f32;
                q[j] = JOINT_LOWER[j] + t * (JOINT_UPPER[j] - JOINT_LOWER[j]);
            }
            if rv_check.validate(deke_types::SRobotQ(q)).is_ok()
                && vamp_robot.validate(&q, &vamp_env, true)
            {
                valid_configs.push(q);
            }
        }

        let mut problems = Vec::new();
        for i in 0..valid_configs.len() / 2 {
            problems.push((valid_configs[i * 2], valid_configs[i * 2 + 1]));
        }
        println!(
            "{} planning problems from {} valid configs",
            problems.len(),
            valid_configs.len()
        );

        let v = validator(wreck_env);
        let rrtc_settings = deke_rrt::RrtcSettings::new(
            deke_types::SRobotQ(JOINT_LOWER),
            deke_types::SRobotQ(JOINT_UPPER),
        );
        let planner = rrtc(rrtc_settings);

        let vamp_settings = vamp::RRTCSettings::default();

        println!(
            "\n{:<8} {:>6} {:>6} {:>8} {:>8} {:>10} {:>10}",
            "", "r_wps", "v_wps", "r_cost", "v_cost", "r_ms", "v_ms"
        );

        for (i, (start, goal)) in problems.iter().enumerate() {
            let (start, goal) = (*start, *goal);
            let mut rv = v.clone();
            let (result, diag) = planner.plan(
                deke_types::SRobotQ(start),
                deke_types::SRobotQ(goal),
                &mut rv,
            );

            let mut rng = vamp::Rng::halton(vamp_robot);
            let vamp_result =
                vamp_robot.plan_rrtc(&start, &goal, &vamp_env, &vamp_settings, &mut rng);

            let (r_wps, r_cost, r_ms, r_ok) = match &result {
                Ok(path) => (
                    path.len(),
                    path.arc_length(),
                    diag.elapsed_ns as f64 / 1e6,
                    true,
                ),
                Err(_) => (0, 0.0, diag.elapsed_ns as f64 / 1e6, false),
            };

            let (v_wps, v_cost, v_ms, v_ok) = match &vamp_result {
                Ok(s) => (s.path.len(), s.cost, s.nanoseconds as f64 / 1e6, true),
                Err(f) => (0, 0.0, f.nanoseconds as f64 / 1e6, false),
            };

            println!(
                "prob {i}   {:>6} {:>6} {:>8.3} {:>8.3} {:>9.3} {:>9.3}  {}",
                r_wps,
                v_wps,
                r_cost,
                v_cost,
                r_ms,
                v_ms,
                match (r_ok, v_ok) {
                    (true, true) => "both ok",
                    (true, false) => "vamp FAIL",
                    (false, true) => "deke FAIL",
                    (false, false) => "BOTH FAIL",
                },
            );

            if r_ok && v_ok {
                let r_path = result.unwrap();
                let v_path = &vamp_result.as_ref().unwrap().path;

                let r_start: [f32; 6] = r_path[0].into();
                let r_goal: [f32; 6] = r_path[r_path.len() - 1].into();
                let v_start = v_path.get_config(0).unwrap();
                let v_goal = v_path.get_config(v_path.len() - 1).unwrap();

                let start_match = r_start
                    .iter()
                    .zip(start.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-3);
                let goal_match = r_goal
                    .iter()
                    .zip(goal.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-3);
                let v_start_match = v_start
                    .iter()
                    .zip(start.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-3);
                let v_goal_match = v_goal
                    .iter()
                    .zip(goal.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-3);

                let straight_line: f32 = start
                    .iter()
                    .zip(goal.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();

                let v_arc: f32 = (0..v_path.len() - 1)
                    .map(|j| {
                        let a = v_path.get_config(j).unwrap();
                        let b = v_path.get_config(j + 1).unwrap();
                        a.iter()
                            .zip(b.iter())
                            .map(|(x, y)| (x - y) * (x - y))
                            .sum::<f32>()
                            .sqrt()
                    })
                    .sum();

                println!(
                    "         endpoints: deke={start_match}/{goal_match} vamp={v_start_match}/{v_goal_match}"
                );
                println!(
                    "         straight_line={straight_line:.3}  deke_arc={r_cost:.3}  vamp_arc={v_arc:.3}  r/opt={:.3}  v/opt={:.3}",
                    r_cost / straight_line,
                    v_arc / straight_line
                );
            }
        }
    }
}
