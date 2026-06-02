//! 6R analytical IK with kinematic-class detection.

use glam::{DMat3, DMat4, DVec3};
use crate::ik_geo::subproblems::{
    subproblem1, subproblem2, subproblem3, subproblem4, subproblem5, subproblem6,
};

use crate::ik_geo::subproblems::auxiliary::{rot as axis_angle, rot_vec};
use crate::kinematics::{create_normal_vector, inverse_homogeneous, reverse_chain};
use crate::remodel::{calc_intersection, do_axes_intersect, is_point_on_axis, remodel_kinematics};
use crate::solver::Chain6;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum Class6 {
    ThreeInnerParallel,
    ThreeParallelTwoIntersecting,
    SphericalFirstTwoParallel,
    SphericalSecondTwoParallel,
    SphericalFirstTwoIntersecting,
    SphericalSecondTwoIntersecting,
    SphericalNoParallelNoIntersecting,
    Reversed,
    Unknown,
}

impl Class6 {
    fn name(self) -> &'static str {
        use Class6::*;
        match self {
            ThreeInnerParallel => "6R-THREE_INNER_PARALLEL",
            ThreeParallelTwoIntersecting => "6R-THREE_PARALLEL_TWO_INTERSECTING",
            SphericalFirstTwoParallel => "6R-SPHERICAL_FIRST_TWO_PARALLEL",
            SphericalSecondTwoParallel => "6R-SPHERICAL_SECOND_TWO_PARALLEL",
            SphericalFirstTwoIntersecting => "6R-SPHERICAL_FIRST_TWO_INTERSECTING",
            SphericalSecondTwoIntersecting => "6R-SPHERICAL_SECOND_TWO_INTERSECTING",
            SphericalNoParallelNoIntersecting => "6R-SPHERICAL_NO_PARALLEL_NO_INTERSECTING",
            Reversed => "6R-REVERSED",
            Unknown => "6R-Unknown Kinematic Class",
        }
    }

    fn is_spherical(self) -> bool {
        use Class6::*;
        matches!(
            self,
            SphericalFirstTwoParallel
                | SphericalSecondTwoParallel
                | SphericalFirstTwoIntersecting
                | SphericalSecondTwoIntersecting
                | SphericalNoParallelNoIntersecting
        )
    }
}

pub struct R6 {
    chain: Chain6,
    zero_thresh: f64,
    class: Class6,
    reversed: Option<Box<R6>>,
}

impl R6 {
    pub fn new(h: &[DVec3], p: &[DVec3], zero_thresh: f64, axis_thresh: f64) -> Self {
        let (class, reversed) = classify(h, p, zero_thresh, axis_thresh);
        Self {
            chain: Chain6::from_slices(h, p),
            zero_thresh,
            class,
            reversed,
        }
    }

    pub fn has_known_decomposition(&self) -> bool {
        match (&self.class, &self.reversed) {
            (Class6::Reversed, Some(rev)) => rev.has_known_decomposition(),
            (Class6::Unknown, _) => false,
            _ => true,
        }
    }

    pub fn is_spherical(&self) -> bool {
        match (&self.class, &self.reversed) {
            (Class6::Reversed, Some(rev)) => rev.is_spherical(),
            (class, _) => class.is_spherical(),
        }
    }

    pub fn kinematic_family(&self) -> String {
        match (&self.class, &self.reversed) {
            (Class6::Reversed, Some(rev)) => rev.kinematic_family(),
            _ => self.class.name().to_string(),
        }
    }

    pub fn solve(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        match self.class {
            Class6::ThreeParallelTwoIntersecting => self.solve_3par_2int(pose),
            Class6::ThreeInnerParallel => self.solve_three_inner_parallel(pose),
            Class6::SphericalFirstTwoParallel => self.solve_sph_12_parallel(pose),
            Class6::SphericalSecondTwoParallel => self.solve_sph_23_parallel(pose),
            Class6::SphericalFirstTwoIntersecting => self.solve_sph_12_intersecting(pose),
            Class6::SphericalSecondTwoIntersecting => self.solve_sph_23_intersecting(pose),
            Class6::SphericalNoParallelNoIntersecting => self.solve_sph_general(pose),
            Class6::Reversed => {
                if let Some(rev) = &self.reversed {
                    let inv = inverse_homogeneous(pose);
                    let mut sols = rev.solve(&inv);
                    for q in sols.iter_mut() {
                        q.0[..6].reverse();
                    }
                    sols
                } else {
                    Vec::new()
                }
            }
            Class6::Unknown => Vec::new(),
        }
    }

    fn pose_decompose(&self, pose: &DMat4) -> (DVec3, DMat3) {
        let p_t = pose.w_axis.truncate();
        let r_06 = DMat3::from_cols(pose.x_axis.truncate(), pose.y_axis.truncate(), pose.z_axis.truncate());
        let p_16 = p_t - self.chain.p[0] - r_06 * self.chain.p[6];
        (p_16, r_06)
    }

    fn solve_3par_2int(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_16, r_06) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let h5 = self.chain.h[5];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];
        let p4 = self.chain.p[4];

        let d = h0.dot(p_16 - p1 - p2 - p3);
        let set4_q4 = subproblem4(&h0, &p4, &h3, d);

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        for q4 in set4_q4.get_all() {
            let r_34 = axis_angle(&h3, q4);
            let set4_q6 =
                subproblem4(&h4, &(r_06.transpose() * h0), &h5, h4.dot(r_34.transpose() * h0));
            for q6 in set4_q6.get_all() {
                let r_56 = axis_angle(&h5, q6);
                let set4_q03 =
                    subproblem4(&h3, &(r_06 * r_56.transpose() * h4), &(-h0), h3.dot(h4));
                for q03 in set4_q03.get_all() {
                    let r_03 = axis_angle(&h0, q03);
                    let hn = create_normal_vector(&h4);
                    let Some(q5) = subproblem1(
                        &(r_56 * hn),
                        &(r_34.transpose() * r_03.transpose() * r_06 * hn),
                        &h4,
                    ) else {
                        continue;
                    };
                    let delta = p3 + r_34 * p4;
                    let set3_q2 = subproblem3(&p2, &(-p1), &h1, (p_16 - r_03 * delta).length());
                    for q2 in set3_q2.get_all() {
                        let r_12 = axis_angle(&h1, q2);
                        let Some(q1) =
                            subproblem1(&(p1 + r_12 * p2), &(p_16 - r_03 * delta), &h0)
                        else {
                            continue;
                        };
                        let r_01 = axis_angle(&h0, q1);
                        let hn3 = create_normal_vector(&h2);
                        if let Some(q3) = subproblem1(
                            &hn3,
                            &(r_12.transpose() * r_01.transpose() * r_03 * hn3),
                            &h2,
                        ) {
                            out.push(crate::solver::pack(&[q1, q2, q3, q4, q5, q6]));
                        }
                    }
                }
            }
        }
        out
    }

    fn solve_three_inner_parallel(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_16, r_06) = self.pose_decompose(pose);
        let zt = self.zero_thresh;
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h4 = self.chain.h[4];
        let h5 = self.chain.h[5];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];
        let p4 = self.chain.p[4];
        let p5 = self.chain.p[5];

        let mut theta1: Vec<f64> = Vec::new();
        let mut theta5: Vec<f64> = Vec::new();

        let h45_cross = h4.cross(h5);
        if h45_cross.length() >= zt && h45_cross.dot(p5).abs() < zt {
            let p15 = p1 + p2 + p3 + p4;
            let set4_q1 = subproblem4(&h1, &p_16, &(-h0), h1.dot(p15));
            for q1 in set4_q1.get_all() {
                let r_01 = axis_angle(&h0, q1);
                let set4_q5 = subproblem4(&h1, &h5, &h4, h1.dot(r_01.transpose() * r_06 * h5));
                for q5 in set4_q5.get_all() {
                    theta1.push(q1);
                    theta5.push(q5);
                }
            }
        } else {
            let d1 = h1.dot(p2 + p3 + p4 + p1);
            let h_arr = [h1, h1, h1, h1];
            let k_arr = [-h0, h4, -h0, h4];
            let p_arr = [p_16, -p5, r_06 * h5, -h5];
            let set6 = subproblem6(&h_arr, &k_arr, &p_arr, d1, 0.0);
            for (q1, q5) in set6.get_all() {
                theta1.push(q1);
                theta5.push(q5);
            }
        }

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        for i in 0..theta1.len() {
            let q1 = theta1[i];
            let q5 = theta5[i];
            let r_01 = axis_angle(&h0, q1);
            let r_45 = axis_angle(&h4, q5);
            let Some(q14) = subproblem1(&(r_45 * h5), &(r_01.transpose() * r_06 * h5), &h1) else {
                continue;
            };
            let Some(q6) =
                subproblem1(&(r_45.transpose() * h1), &(r_06.transpose() * r_01 * h1), &(-h5))
            else {
                continue;
            };

            let r_14 = axis_angle(&h1, q14);
            let d_inner = r_01.transpose() * p_16 - p1 - r_14 * r_45 * p5 - r_14 * p4;
            let d = d_inner.length();
            let set3 = subproblem3(&(-p3), &p2, &h1, d);

            for q3 in set3.get_all() {
                let rot = axis_angle(&h1, q3);
                if let Some(q2) = subproblem1(&(p2 + rot * p3), &d_inner, &h1) {
                    let mut q4 = q14 - q2 - q3;
                    q4 = q4.sin().atan2(q4.cos());
                    out.push(crate::solver::pack(&[q1, q2, q3, q4, q5, q6]));
                }
            }
        }
        out
    }

    fn solve_sph_12_parallel(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_16, r_06) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let mut position_solutions = Vec::new();
        let set4 = subproblem4(&h0, &p3, &h2, h0.dot(p_16 - p1 - p2));
        let nh0 = -h0;
        for q3 in set4.get_all() {
            let rot_3_p3 = rot_vec(&h2, q3, p3);
            let shifted = rot_3_p3 + p2;
            let set3 = subproblem3(&p_16, &p1, &(-h0), shifted.length());
            for q1 in set3.get_all() {
                let rot1_p16 = rot_vec(&nh0, q1, p_16);
                if let Some(q2) = subproblem1(&shifted, &(rot1_p16 - p1), &h1) {
                    position_solutions.push([q1, q2, q3]);
                }
            }
        }
        self.spherical_wrist_orientation(&position_solutions, &r_06)
    }

    fn solve_sph_23_parallel(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_16, r_06) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let mut position_solutions = Vec::new();
        let set4 = subproblem4(&h1, &p_16, &(-h0), h1.dot(p1 + p2 + p3));
        let nh0 = -h0;
        for q1 in set4.get_all() {
            let target = rot_vec(&nh0, q1, -p_16) + p1;
            let set3 = subproblem3(&(-p3), &p2, &h2, target.length());
            for q3 in set3.get_all() {
                let rot_3_p3 = rot_vec(&h2, q3, p3);
                if let Some(q2) = subproblem1(&(-p2 - rot_3_p3), &target, &h1) {
                    position_solutions.push([q1, q2, q3]);
                }
            }
        }
        self.spherical_wrist_orientation(&position_solutions, &r_06)
    }

    fn solve_sph_12_intersecting(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_16, r_06) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let mut position_solutions = Vec::new();
        let set3 = subproblem3(&p3, &(-p2), &h2, p_16.length());
        for q3 in set3.get_all() {
            let rot_3_p3 = rot_vec(&h2, q3, p3);
            let set2 = subproblem2(&p_16, &(p2 + rot_3_p3), &(-h0), &h1);
            for (q1, q2) in set2.get_all() {
                position_solutions.push([q1, q2, q3]);
            }
        }
        self.spherical_wrist_orientation(&position_solutions, &r_06)
    }

    fn solve_sph_23_intersecting(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_16, r_06) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let p1 = self.chain.p[1];
        let p3 = self.chain.p[3];

        let mut position_solutions = Vec::new();
        let set3 = subproblem3(&p_16, &p1, &(-h0), p3.length());
        let nh0 = -h0;
        for q1 in set3.get_all() {
            let rot1_p16 = rot_vec(&nh0, q1, p_16);
            let set2 = subproblem2(&(rot1_p16 - p1), &p3, &(-h1), &h2);
            for (q2, q3) in set2.get_all() {
                position_solutions.push([q1, q2, q3]);
            }
        }
        self.spherical_wrist_orientation(&position_solutions, &r_06)
    }

    fn solve_sph_general(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_16, r_06) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let set5 = subproblem5(&(-p1), &p_16, &p2, &p3, &(-h0), &h1, &h2);
        let mut position_solutions = Vec::new();
        for (q1, q2, q3) in set5.get_all() {
            position_solutions.push([q1, q2, q3]);
        }
        self.spherical_wrist_orientation(&position_solutions, &r_06)
    }

    fn spherical_wrist_orientation(
        &self,
        position_solutions: &[[f64; 3]],
        r_06: &DMat3,
    ) -> Vec<crate::solver::Joints> {
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let h5 = self.chain.h[5];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        for qpos in position_solutions {
            let q1 = qpos[0];
            let q2 = qpos[1];
            let q3 = qpos[2];
            let rot_1_inv = axis_angle(&(-h0), q1);
            let rot_2_inv = axis_angle(&(-h1), q2);
            let rot_3_inv = axis_angle(&(-h2), q3);
            let r_36 = rot_3_inv * rot_2_inv * rot_1_inv * r_06;
            let r36_h5 = r_36 * h5;
            let r36t_h3 = r_36.transpose() * h3;

            let set4 = subproblem4(&h3, &h5, &h4, h3.dot(r36_h5));
            for q5 in set4.get_all() {
                let rot_5 = axis_angle(&h4, q5);
                let Some(q4) = subproblem1(&(rot_5 * h5), &r36_h5, &h3) else {
                    continue;
                };
                let Some(q6) = subproblem1(&(rot_5.transpose() * h3), &r36t_h3, &(-h5)) else {
                    continue;
                };
                out.push(crate::solver::pack(&[q1, q2, q3, q4, q5, q6]));
            }
        }
        out
    }
}

fn classify(
    h: &[DVec3],
    p: &[DVec3],
    zt: f64,
    at: f64,
) -> (Class6, Option<Box<R6>>) {
    let h0 = h[0];
    let h1 = h[1];
    let h2 = h[2];
    let h3 = h[3];
    let h4 = h[4];
    let h5 = h[5];
    let p0 = p[0];
    let p1 = p[1];
    let p2 = p[2];
    let p3 = p[3];
    let p4 = p[4];
    let p5 = p[5];

    let make_reversed = |remodel: bool| {
        let (h_rev, p_rev) = reverse_chain(h, p);
        let p_used = if remodel {
            remodel_kinematics(&h_rev, &p_rev, zt, at)
        } else {
            p_rev.clone()
        };
        let (class, nested) = classify(&h_rev, &p_used, zt, at);
        Box::new(R6 {
            chain: Chain6::from_slices(&h_rev, &p_used),
            zero_thresh: zt,
            class,
            reversed: nested,
        })
    };

    if h0.cross(h1).length() < zt
        && h0.cross(h2).length() < zt
        && h1.cross(h2).length() < zt
        && h4.cross(h5).length() >= zt
        && h4.cross(h5).dot(p5).abs() < zt
    {
        return (Class6::ThreeParallelTwoIntersecting, None);
    }

    if h1.cross(h2).length() < zt && h1.cross(h3).length() < zt && h2.cross(h3).length() < zt {
        return (Class6::ThreeInnerParallel, None);
    }

    if h2.cross(h3).length() < zt && h2.cross(h4).length() < zt && h3.cross(h4).length() < zt {
        return (Class6::Reversed, Some(make_reversed(true)));
    }

    if h3.cross(h4).length() < zt
        && h3.cross(h5).length() < zt
        && h4.cross(h5).length() < zt
        && h0.cross(h1).length() >= zt
        && h0.cross(h1).dot(p1).abs() < zt
    {
        return (Class6::Reversed, Some(make_reversed(false)));
    }

    if do_axes_intersect(&h3, &h4, &p4, zt, zt) {
        let p04 = p0 + p1 + p2 + p3;
        let intersection = calc_intersection(&h3, &h4, &p04, &p4, zt);
        if is_point_on_axis(&h5, &(p04 + p4 + p5), &intersection, zt) {
            if h0.cross(h1).length() < zt {
                return (Class6::SphericalFirstTwoParallel, None);
            }
            if h1.cross(h2).length() < zt {
                return (Class6::SphericalSecondTwoParallel, None);
            }
            if h0.cross(h1).dot(p1).abs() < zt {
                return (Class6::SphericalFirstTwoIntersecting, None);
            }
            if h1.cross(h2).dot(p2).abs() < zt {
                return (Class6::SphericalSecondTwoIntersecting, None);
            }
            return (Class6::SphericalNoParallelNoIntersecting, None);
        }
    }

    if do_axes_intersect(&h0, &h1, &p1, zt, zt) {
        let intersection = calc_intersection(&h0, &h1, &p0, &p1, zt);
        if is_point_on_axis(&h2, &(p0 + p1 + p2), &intersection, zt) {
            return (Class6::Reversed, Some(make_reversed(true)));
        }
    }

    (Class6::Unknown, None)
}
