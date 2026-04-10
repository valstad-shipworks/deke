use glam::Affine3A;
use rerun::{RecordingStream, RecordingStreamResult};

pub trait ToRerun {
    fn log_to(&self, rec: &RecordingStream, entity_path: &str) -> RecordingStreamResult<()>;
}

impl ToRerun for wreck::Sphere {
    fn log_to(&self, rec: &RecordingStream, entity_path: &str) -> RecordingStreamResult<()> {
        rec.log(
            entity_path,
            &rerun::Ellipsoids3D::from_centers_and_radii(
                [[self.center.x, self.center.y, self.center.z]],
                [self.radius],
            ),
        )
    }
}

impl ToRerun for wreck::Cylinder {
    fn log_to(&self, rec: &RecordingStream, entity_path: &str) -> RecordingStreamResult<()> {
        let p2 = self.p2();
        let center = (self.p1 + p2) * 0.5;
        let length = self.dir.length();
        let dir = if length > f32::EPSILON {
            self.dir / length
        } else {
            glam::Vec3::Y
        };
        let rotation = rotation_from_y_axis(dir);
        rec.log(
            entity_path,
            &rerun::Cylinders3D::from_lengths_and_radii([length], [self.radius])
                .with_centers([[center.x, center.y, center.z]])
                .with_quaternions([rotation]),
        )
    }
}

impl ToRerun for wreck::Cuboid {
    fn log_to(&self, rec: &RecordingStream, entity_path: &str) -> RecordingStreamResult<()> {
        let he = self.half_extents;
        let full = [he[0] * 2.0, he[1] * 2.0, he[2] * 2.0];
        let rotation = rotation_from_axes(self.axes);
        rec.log(
            entity_path,
            &rerun::Boxes3D::from_sizes([full])
                .with_centers([[self.center.x, self.center.y, self.center.z]])
                .with_quaternions([rotation]),
        )
    }
}

impl ToRerun for wreck::Capsule {
    fn log_to(&self, rec: &RecordingStream, entity_path: &str) -> RecordingStreamResult<()> {
        let p2 = self.p2();
        let center = (self.p1 + p2) * 0.5;
        let length = self.dir.length();
        let dir = if length > f32::EPSILON {
            self.dir / length
        } else {
            glam::Vec3::Y
        };
        let rotation = rotation_from_y_axis(dir);
        rec.log(
            entity_path,
            &rerun::Capsules3D::from_lengths_and_radii([length], [self.radius])
                .with_translations([[center.x, center.y, center.z]])
                .with_quaternions([rotation]),
        )
    }
}

impl ToRerun for wreck::Collider {
    fn log_to(&self, rec: &RecordingStream, entity_path: &str) -> RecordingStreamResult<()> {
        for (i, sphere) in self.spheres().iter().enumerate() {
            sphere.log_to(rec, &format!("{entity_path}/spheres/{i}"))?;
        }
        for (i, cuboid) in self.cuboids().iter().enumerate() {
            cuboid.log_to(rec, &format!("{entity_path}/cuboids/{i}"))?;
        }
        for (i, cylinder) in self.cylinders().iter().enumerate() {
            cylinder.log_to(rec, &format!("{entity_path}/cylinders/{i}"))?;
        }
        for (i, capsule) in self.capsules().iter().enumerate() {
            capsule.log_to(rec, &format!("{entity_path}/capsules/{i}"))?;
        }
        Ok(())
    }
}

/// Log an entire `Collider` to rerun at the given entity path.
pub fn log_collider(
    rec: &RecordingStream,
    entity_path: &str,
    collider: &wreck::Collider,
) -> RecordingStreamResult<()> {
    collider.log_to(rec, entity_path)
}

impl ToRerun for wreck::soa::SpheresSoA {
    fn log_to(&self, rec: &RecordingStream, entity_path: &str) -> RecordingStreamResult<()> {
        let centers: Vec<[f32; 3]> = self
            .iter()
            .map(|s| [s.center.x, s.center.y, s.center.z])
            .collect();
        let radii: Vec<f32> = self.iter().map(|s| s.radius).collect();
        rec.log(
            entity_path,
            &rerun::Ellipsoids3D::from_centers_and_radii(centers, radii),
        )
    }
}

/// Per-link mesh data: the raw STL bytes and the visual origin offset within the link frame.
pub struct LinkMesh {
    pub stl_data: Vec<u8>,
    pub visual_origin: Affine3A,
}

/// Pre-loaded robot meshes for rerun visualization.
pub struct RobotMeshes<const N: usize> {
    pub base: Option<LinkMesh>,
    pub links: [LinkMesh; N],
}

impl<const N: usize> RobotMeshes<N> {
    /// Log the static base mesh (it never moves).
    pub fn log_base_static(
        &self,
        rec: &RecordingStream,
        entity_path: &str,
    ) -> RecordingStreamResult<()> {
        if let Some(base) = &self.base {
            let tf = base.visual_origin;
            log_mesh_at(rec, entity_path, &base.stl_data, tf)?;
        }
        Ok(())
    }

    /// Log all link meshes at the given FK transforms.
    pub fn log_links(
        &self,
        rec: &RecordingStream,
        entity_path: &str,
        fk_transforms: &[Affine3A; N],
    ) -> RecordingStreamResult<()> {
        for (i, (link, fk_tf)) in self.links.iter().zip(fk_transforms.iter()).enumerate() {
            let world_tf = *fk_tf * link.visual_origin;
            log_transform(rec, &format!("{entity_path}/link_{i}"), world_tf)?;
        }
        Ok(())
    }

    /// Log the STL assets as static (the geometry itself doesn't change, only the transforms do).
    pub fn log_link_assets_static(
        &self,
        rec: &RecordingStream,
        entity_path: &str,
    ) -> RecordingStreamResult<()> {
        for (i, link) in self.links.iter().enumerate() {
            rec.log_static(
                format!("{entity_path}/link_{i}"),
                &rerun::Asset3D::from_file_contents(
                    link.stl_data.clone(),
                    Some(rerun::MediaType::STL),
                ),
            )?;
        }
        Ok(())
    }
}

/// Log a `RobotPath` as an end-effector line strip using `fk()` to get the
/// TCP position through all intermediate link origins at each waypoint.
pub fn log_path_tcp<const N: usize>(
    rec: &RecordingStream,
    entity_path: &str,
    path: &deke_types::RobotPath,
    fk: &impl deke_types::FKChain<N>,
) -> RecordingStreamResult<()> {
    let points: Vec<[f32; 3]> = path
        .rows()
        .into_iter()
        .filter_map(|row| {
            let sq = deke_types::SRobotQ::<N>::try_from(row.as_slice()?).ok()?;
            let transforms = fk.fk(&sq).ok()?;
            let t = transforms[N - 1].translation;
            Some([t.x, t.y, t.z])
        })
        .collect();
    rec.log(entity_path, &rerun::LineStrips3D::new([points]))
}

/// Log the path waypoints as 3D points at TCP positions.
pub fn log_waypoints<const N: usize>(
    rec: &RecordingStream,
    entity_path: &str,
    path: &deke_types::RobotPath,
    fk: &impl deke_types::FKChain<N>,
) -> RecordingStreamResult<()> {
    let points: Vec<[f32; 3]> = path
        .rows()
        .into_iter()
        .filter_map(|row| {
            let sq = deke_types::SRobotQ::<N>::try_from(row.as_slice()?).ok()?;
            let transforms = fk.fk(&sq).ok()?;
            let t = transforms[N - 1].translation;
            Some([t.x, t.y, t.z])
        })
        .collect();
    rec.log_static(entity_path, &rerun::Points3D::new(points))
}

/// Log the kinematic chain as a line strip through all link origins for a single config.
pub fn log_chain_line<const N: usize>(
    rec: &RecordingStream,
    entity_path: &str,
    fk: &impl deke_types::FKChain<N>,
    q: deke_types::SRobotQ<N>,
) -> RecordingStreamResult<()> {
    let Ok(transforms) = fk.fk(&q) else {
        return Ok(());
    };
    let mut points: Vec<[f32; 3]> = Vec::with_capacity(N + 1);
    points.push([0.0, 0.0, 0.0]);
    for tf in &transforms {
        let t = tf.translation;
        points.push([t.x, t.y, t.z]);
    }
    rec.log(entity_path, &rerun::LineStrips3D::new([points]))
}

/// Log the collision spheres of an `InlinedWreckValidator` at a given joint config.
pub fn log_robot_spheres<const N: usize>(
    rec: &RecordingStream,
    entity_path: &str,
    validator: &deke_wreck::InlinedWreckValidator<N>,
    q: deke_types::SRobotQ<N>,
) -> RecordingStreamResult<()> {
    let spheres = validator.spheres(q);
    spheres.log_to(rec, entity_path)
}

/// Log the collision spheres at each waypoint in a path using `set_time_sequence`.
pub fn log_path_spheres<const N: usize>(
    rec: &RecordingStream,
    entity_path: &str,
    path: &deke_types::RobotPath,
    validator: &deke_wreck::InlinedWreckValidator<N>,
) -> RecordingStreamResult<()> {
    for (i, row) in path.rows().into_iter().enumerate() {
        let Some(slice) = row.as_slice() else {
            continue;
        };
        let Ok(sq) = deke_types::SRobotQ::<N>::try_from(slice) else {
            continue;
        };
        rec.set_time_sequence("step", i as i64);
        log_robot_spheres(rec, entity_path, validator, sq)?;
    }
    Ok(())
}

/// Compute the time each segment takes given per-joint max velocities (rad/s).
/// Each segment's duration is the slowest joint: `max_j(|dq_j| / v_max_j)`.
pub fn segment_times(path: &deke_types::RobotPath, joint_velocities: &[f32]) -> Vec<f64> {
    (0..path.nrows().saturating_sub(1))
        .map(|i| {
            let a = path.row(i);
            let b = path.row(i + 1);
            a.iter()
                .zip(b.iter())
                .zip(joint_velocities.iter())
                .map(|((ai, bi), &v)| (bi - ai).abs() as f64 / v as f64)
                .fold(0.0, f64::max)
        })
        .collect()
}

/// Log meshes and chain line along a path in real-time.
///
/// Expects an `SRobotPath` so the path can be densified and iterated as typed waypoints.
pub fn log_path_realtime<const N: usize>(
    rec: &RecordingStream,
    entity_path: &str,
    path: &deke_types::SRobotPath<N>,
    fk: &impl deke_types::FKChain<N>,
    meshes: Option<&RobotMeshes<N>>,
    joint_velocities: &[f32; N],
) -> RecordingStreamResult<()> {
    let dense = path.densify(0.02);
    let dense_rp = dense.to_robot_path();
    let times = segment_times(&dense_rp, joint_velocities);

    if let Some(m) = meshes {
        m.log_link_assets_static(rec, &format!("{entity_path}/mesh"))?;
        m.log_base_static(rec, &format!("{entity_path}/mesh/base"))?;
    }

    let mut t = 0.0_f64;
    let wall_start = std::time::Instant::now();

    for (i, sq) in dense.iter().enumerate() {
        let wall_target = std::time::Duration::from_secs_f64(t);
        let elapsed = wall_start.elapsed();
        if wall_target > elapsed {
            std::thread::sleep(wall_target - elapsed);
        }

        rec.set_duration_secs("time", t);

        if let Ok(fk_transforms) = fk.fk(sq) {
            if let Some(m) = meshes {
                m.log_links(rec, &format!("{entity_path}/mesh"), &fk_transforms)?;
            }
            let mut chain_pts: Vec<[f32; 3]> = Vec::with_capacity(N + 1);
            chain_pts.push([0.0, 0.0, 0.0]);
            for tf in &fk_transforms {
                let p = tf.translation;
                chain_pts.push([p.x, p.y, p.z]);
            }
            rec.log(
                format!("{entity_path}/chain"),
                &rerun::LineStrips3D::new([chain_pts]),
            )?;
        }

        if i < times.len() {
            t += times[i];
        }
    }
    Ok(())
}

/// Build an `Affine3A` from URDF-style xyz + rpy.
pub fn affine_from_xyz_rpy(xyz: [f64; 3], rpy: [f64; 3]) -> Affine3A {
    let [roll, pitch, yaw] = rpy;
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    let m = glam::Mat3A::from_cols(
        glam::Vec3A::new((cy * cp) as f32, (sy * cp) as f32, (-sp) as f32),
        glam::Vec3A::new(
            (cy * sp * sr - sy * cr) as f32,
            (sy * sp * sr + cy * cr) as f32,
            (cp * sr) as f32,
        ),
        glam::Vec3A::new(
            (cy * sp * cr + sy * sr) as f32,
            (sy * sp * cr - cy * sr) as f32,
            (cp * cr) as f32,
        ),
    );
    Affine3A {
        matrix3: m,
        translation: glam::Vec3A::new(xyz[0] as f32, xyz[1] as f32, xyz[2] as f32),
    }
}

fn log_mesh_at(
    rec: &RecordingStream,
    entity_path: &str,
    stl_data: &[u8],
    tf: Affine3A,
) -> RecordingStreamResult<()> {
    log_transform(rec, entity_path, tf)?;
    rec.log_static(
        entity_path,
        &rerun::Asset3D::from_file_contents(stl_data.to_vec(), Some(rerun::MediaType::STL)),
    )
}

fn log_transform(
    rec: &RecordingStream,
    entity_path: &str,
    tf: Affine3A,
) -> RecordingStreamResult<()> {
    let t = tf.translation;
    let quat = glam::Quat::from_mat3a(&tf.matrix3);
    rec.log(
        entity_path,
        &rerun::Transform3D::from_translation_rotation(
            [t.x, t.y, t.z],
            rerun::Quaternion::from_xyzw([quat.x, quat.y, quat.z, quat.w]),
        ),
    )
}

fn rotation_from_y_axis(dir: glam::Vec3) -> rerun::Quaternion {
    let from = glam::Vec3::Y;
    let quat = glam::Quat::from_rotation_arc(from, dir);
    rerun::Quaternion::from_xyzw([quat.x, quat.y, quat.z, quat.w])
}

fn rotation_from_axes(axes: [glam::Vec3; 3]) -> rerun::Quaternion {
    let mat = glam::Mat3::from_cols(axes[0], axes[1], axes[2]);
    let quat = glam::Quat::from_mat3(&mat);
    rerun::Quaternion::from_xyzw([quat.x, quat.y, quat.z, quat.w])
}
