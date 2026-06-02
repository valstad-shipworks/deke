//! 5R analytical IK with kinematic-class detection.

use glam::{DMat3, DMat4, DVec3};
use crate::ik_geo::subproblems::{subproblem1, subproblem2, subproblem3, subproblem4, subproblem5, subproblem6};

use crate::ik_geo::subproblems::auxiliary::rot as axis_angle;
use crate::kinematics::{create_normal_vector, inverse_homogeneous, reverse_chain};
use crate::remodel::{calc_intersection, do_axes_intersect, is_point_on_axis, remodel_kinematics};
use crate::solver::Chain5;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum Class5 {
    FourthFifthIntersecting,
    FourthFifthIntersectingSecondThirdIntersecting,
    FourthFifthIntersectingFirstSecondIntersecting,
    FourthFifthIntersectingFirstSecondParallel,
    FourthFifthIntersectingSecondThirdParallel,
    SphericalWrist,
    SphericalWristFirstSecondIntersecting,
    ThirdFourthIntersectingSecondThirdParallel,
    ThirdFourthIntersectingSecondThirdParallelFourthFifthParallel,
    FirstSecondThirdParallel,
    FirstSecondThirdParallelFourthFifthParallel,
    SecondThirdFourthParallel,
    Reversed,
    Unknown,
}

impl Class5 {
    fn name(self) -> &'static str {
        use Class5::*;
        match self {
            FourthFifthIntersecting => "5R-FOURTH_FITH_INTERSECTING",
            FourthFifthIntersectingSecondThirdIntersecting => {
                "5R-FOURTH_FITH_INTERSECTING_SECOND_THIRD_INTERSECTING"
            }
            FourthFifthIntersectingFirstSecondIntersecting => {
                "5R-FOURTH_FITH_INTERSECTING_FIRST_SECOND_INTERSECTING"
            }
            FourthFifthIntersectingFirstSecondParallel => {
                "5R-FOURTH_FITH_INTERSECTING_FIRST_SECOND_PARALLEL"
            }
            FourthFifthIntersectingSecondThirdParallel => {
                "5R-FOURTH_FITH_INTERSECTING_SECOND_THIRD_PARALLEL"
            }
            SphericalWrist => "5R-SPHERICAL_WRIST",
            SphericalWristFirstSecondIntersecting => "5R-SPHERICAL_WRIST_FIRST_SECOND_INTERSECTING",
            ThirdFourthIntersectingSecondThirdParallel => {
                "5R-THIRD_FOURTH_INTERSECTING_SECOND_THIRD_PARALLEL"
            }
            ThirdFourthIntersectingSecondThirdParallelFourthFifthParallel => {
                "5R-THIRD_FOURTH_INTERSECTING_SECOND_THIRD_PARALLEL_FOURTH_FITH_PARALLEL"
            }
            FirstSecondThirdParallel => "5R-FIRST_SECOND_THIRD_PARALLEL",
            FirstSecondThirdParallelFourthFifthParallel => {
                "5R-FIRST_SECOND_THIRD_PARALLEL_FOURTH_FITH_PARALLEL"
            }
            SecondThirdFourthParallel => "5R-SECOND_THIRD_FOURTH_PARALLEL",
            Reversed => "5R-REVERSED",
            Unknown => "5R-Unknown Kinematic Class",
        }
    }
}

pub struct R5 {
    chain: Chain5,
    #[allow(dead_code)]
    zero_thresh: f64,
    class: Class5,
    reversed: Option<Box<R5>>,
}

impl R5 {
    pub fn new(h: &[DVec3], p: &[DVec3], zero_thresh: f64, axis_thresh: f64) -> Self {
        let (class, reversed) = classify(h, p, zero_thresh, axis_thresh);
        Self {
            chain: Chain5::from_slices(h, p),
            zero_thresh,
            class,
            reversed,
        }
    }

    pub fn has_known_decomposition(&self) -> bool {
        !matches!(self.class, Class5::Unknown)
    }

    pub fn kinematic_family(&self) -> String {
        match (&self.class, &self.reversed) {
            (Class5::Reversed, Some(rev)) => rev.kinematic_family(),
            _ => self.class.name().to_string(),
        }
    }

    pub fn solve(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        match self.class {
            Class5::FourthFifthIntersecting => self.solve_45_intersecting(pose),
            Class5::FourthFifthIntersectingSecondThirdIntersecting => {
                self.solve_45int_23int(pose)
            }
            Class5::FourthFifthIntersectingFirstSecondIntersecting => {
                self.solve_45int_12int(pose)
            }
            Class5::FourthFifthIntersectingFirstSecondParallel => self.solve_45int_12par(pose),
            Class5::FourthFifthIntersectingSecondThirdParallel => self.solve_45int_23par(pose),
            Class5::SphericalWrist => self.solve_spherical_wrist(pose),
            Class5::SphericalWristFirstSecondIntersecting => self.solve_sphwrist_12int(pose),
            Class5::ThirdFourthIntersectingSecondThirdParallel => self.solve_34int_23par(pose),
            Class5::ThirdFourthIntersectingSecondThirdParallelFourthFifthParallel => {
                self.solve_34int_23par_45par(pose)
            }
            Class5::FirstSecondThirdParallel => self.solve_123par(pose),
            Class5::FirstSecondThirdParallelFourthFifthParallel => self.solve_123par_45par(pose),
            Class5::SecondThirdFourthParallel => self.solve_234par(pose),
            Class5::Reversed => {
                if let Some(rev) = &self.reversed {
                    let inv = inverse_homogeneous(pose);
                    let mut sols = rev.solve(&inv);
                    for q in sols.iter_mut() {
                        q.0[..5].reverse();
                    }
                    sols
                } else {
                    Vec::new()
                }
            }
            Class5::Unknown => Vec::new(),
        }
    }

    fn pose_decompose(&self, pose: &DMat4) -> (DVec3, DMat3) {
        let p_t = pose.w_axis.truncate();
        let r_05 = DMat3::from_cols(pose.x_axis.truncate(), pose.y_axis.truncate(), pose.z_axis.truncate());
        let p_15 = p_t - self.chain.p[0] - r_05 * self.chain.p[5];
        (p_15, r_05)
    }

    fn solve_45_intersecting(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set5 = subproblem5(&(-p1), &p_15, &p2, &p3, &(-h0), &h1, &h2);
        for (q1, q2, q3) in set5.get_all() {
            let r_01 = axis_angle(&h0, q1);
            let r_12 = axis_angle(&h1, q2);
            let r_23 = axis_angle(&h2, q3);
            let hn = (r_05 * h4).cross(r_01 * r_12 * r_23 * h3);
            let set = subproblem2(
                &(r_05.transpose() * hn),
                &(r_23.transpose() * r_12.transpose() * r_01.transpose() * hn),
                &h4,
                &(-h3),
            );
            for (q5, q4) in set.get_all() {
                out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
            }
        }
        out
    }

    fn solve_45int_23int(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p1 = self.chain.p[1];
        let p3 = self.chain.p[3];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set3 = subproblem3(&p_15, &p1, &(-h0), p3.length());
        for q1 in set3.get_all() {
            let r_01 = axis_angle(&h0, q1);
            let set23 = subproblem2(&(r_01.transpose() * p_15 - p1), &p3, &(-h1), &h2);
            for (q2, q3) in set23.get_all() {
                let r_12 = axis_angle(&h1, q2);
                let r_23 = axis_angle(&h2, q3);
                let hn = (r_05 * h4).cross(r_01 * r_12 * r_23 * h3);
                let set45 = subproblem2(
                    &(r_05.transpose() * hn),
                    &(r_23.transpose() * r_12.transpose() * r_01.transpose() * hn),
                    &h4,
                    &(-h3),
                );
                for (q5, q4) in set45.get_all() {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                }
            }
        }
        out
    }

    fn solve_45int_12int(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set3 = subproblem3(&p3, &(-p2), &h2, p_15.length());
        for q3 in set3.get_all() {
            let r_23 = axis_angle(&h2, q3);
            let set12 = subproblem2(&p_15, &(p2 + r_23 * p3), &(-h0), &h1);
            for (q1, q2) in set12.get_all() {
                let r_01 = axis_angle(&h0, q1);
                let r_12 = axis_angle(&h1, q2);
                let hn = (r_05 * h4).cross(r_01 * r_12 * r_23 * h3);
                let set45 = subproblem2(
                    &(r_05.transpose() * hn),
                    &(r_23.transpose() * r_12.transpose() * r_01.transpose() * hn),
                    &h4,
                    &(-h3),
                );
                for (q5, q4) in set45.get_all() {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                }
            }
        }
        out
    }

    fn solve_45int_12par(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set4 = subproblem4(&h0, &p3, &h2, h0.dot(p_15 - p1 - p2));
        for q3 in set4.get_all() {
            let r_23 = axis_angle(&h2, q3);
            let set3 = subproblem3(&p_15, &p1, &(-h0), (p2 + r_23 * p3).length());
            for q1 in set3.get_all() {
                let r_01 = axis_angle(&h0, q1);
                let Some(q2) =
                    subproblem1(&(p2 + r_23 * p3), &(r_01.transpose() * p_15 - p1), &h1)
                else {
                    continue;
                };
                let r_12 = axis_angle(&h1, q2);
                let hn = (r_05 * h4).cross(r_01 * r_12 * r_23 * h3);
                let set45 = subproblem2(
                    &(r_05.transpose() * hn),
                    &(r_23.transpose() * r_12.transpose() * r_01.transpose() * hn),
                    &h4,
                    &(-h3),
                );
                for (q5, q4) in set45.get_all() {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                }
            }
        }
        out
    }

    fn solve_45int_23par(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set4 = subproblem4(&h1, &p_15, &(-h0), h1.dot(p1 + p2 + p3));
        for q1 in set4.get_all() {
            let r_01 = axis_angle(&h0, q1);
            let placeholder = r_01.transpose() * p_15 - p1;
            let set3a = subproblem3(&p3, &(-p2), &h2, placeholder.length());
            for q3 in set3a.get_all() {
                let r_23 = axis_angle(&h2, q3);
                let set3b = subproblem3(&placeholder, &p2, &(-h1), p3.length());
                for q2 in set3b.get_all() {
                    let r_12 = axis_angle(&h1, q2);
                    let hn = (r_05 * h4).cross(r_01 * r_12 * r_23 * h3);
                    let set45 = subproblem2(
                        &(r_05.transpose() * hn),
                        &(r_23.transpose() * r_12.transpose() * r_01.transpose() * hn),
                        &h4,
                        &(-h3),
                    );
                    for (q5, q4) in set45.get_all() {
                        out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                    }
                }
            }
        }
        out
    }

    fn solve_spherical_wrist(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set3 = subproblem3(&p1, &p_15, &h0, p2.length());
        for q1 in set3.get_all() {
            let r_10 = axis_angle(&(-h0), q1);
            let Some(q2) = subproblem1(&p2, &(r_10 * p_15 - p1), &h1) else {
                continue;
            };
            let r_21 = axis_angle(&(-h1), q2);
            let set = subproblem2(&(r_21 * r_10 * r_05 * h4), &h4, &(-h2), &h3);
            for (q3, q4) in set.get_all() {
                let r_32 = axis_angle(&(-h2), q3);
                let r_43 = axis_angle(&(-h3), q4);
                let r_40 = r_43 * r_32 * r_21 * r_10;
                let hn = create_normal_vector(&(r_40.transpose() * h4));
                if let Some(q5) =
                    subproblem1(&(r_43 * r_32 * r_21 * r_10 * hn), &(r_05.transpose() * hn), &(-h4))
                {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                }
            }
        }
        out
    }

    fn solve_sphwrist_12int(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p2 = self.chain.p[2];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set12 = subproblem2(&p_15, &p2, &(-h0), &h1);
        for (q1, q2) in set12.get_all() {
            let r_10 = axis_angle(&(-h0), q1);
            let r_21 = axis_angle(&(-h1), q2);
            let set34 = subproblem2(&(r_21 * r_10 * r_05 * h4), &h4, &(-h2), &h3);
            for (q3, q4) in set34.get_all() {
                let r_32 = axis_angle(&(-h2), q3);
                let r_43 = axis_angle(&(-h3), q4);
                let r_40 = r_43 * r_32 * r_21 * r_10;
                let hn = create_normal_vector(&(r_40.transpose() * h4));
                if let Some(q5) = subproblem1(&(r_40 * hn), &(r_05.transpose() * hn), &(-h4)) {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                }
            }
        }
        out
    }

    fn solve_34int_23par(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p4 = self.chain.p[4];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let h_arr = [h1, h1, h1, h1];
        let k_arr = [-h0, h3, -h0, h3];
        let p_arr = [p_15, -p4, r_05 * h4, -h4];
        let d1 = h1.dot(p1 + p2);
        let set6 = subproblem6(&h_arr, &k_arr, &p_arr, d1, 0.0);

        for (q1, q4) in set6.get_all() {
            let r_01 = axis_angle(&h0, q1);
            let r_34 = axis_angle(&h3, q4);
            let set3 =
                subproblem3(&(r_34 * p4), &(-p2), &h2, (r_01.transpose() * p_15 - p1).length());
            for q3 in set3.get_all() {
                let r_23 = axis_angle(&h2, q3);
                let Some(q2) = subproblem1(
                    &(p2 + r_23 * r_34 * p4),
                    &(r_01.transpose() * p_15 - p1),
                    &h1,
                ) else {
                    continue;
                };
                let r_12 = axis_angle(&h1, q2);
                let hn = create_normal_vector(&h4);
                if let Some(q5) = subproblem1(
                    &hn,
                    &(r_34.transpose() * r_23.transpose() * r_12.transpose() * r_01.transpose() * r_05 * hn),
                    &h4,
                ) {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                }
            }
        }
        out
    }

    fn solve_34int_23par_45par(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p4 = self.chain.p[4];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set4a = subproblem4(&h1, &(r_05 * h4), &(-h0), h1.dot(h4));
        for q1 in set4a.get_all() {
            let r_01 = axis_angle(&h0, q1);
            let set4b = subproblem4(&h1, &p4, &h3, h1.dot(r_01.transpose() * p_15 - p1 - p2));
            for q4 in set4b.get_all() {
                let r_34 = axis_angle(&h3, q4);
                let set3 =
                    subproblem3(&(r_34 * p4), &(-p2), &h2, (r_01.transpose() * p_15 - p1).length());
                for q3 in set3.get_all() {
                    let r_23 = axis_angle(&h2, q3);
                    let Some(q2) = subproblem1(
                        &(p2 + r_23 * r_34 * p4),
                        &(r_01.transpose() * p_15 - p1),
                        &h1,
                    ) else {
                        continue;
                    };
                    let r_12 = axis_angle(&h1, q2);
                    let hn = create_normal_vector(&(r_05 * h4));
                    if let Some(q5) = subproblem1(
                        &(r_05.transpose() * hn),
                        &(r_34.transpose() * r_23.transpose() * r_12.transpose() * r_01.transpose() * hn),
                        &h4,
                    ) {
                        out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                    }
                }
            }
        }
        out
    }

    fn solve_123par(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];
        let p4 = self.chain.p[4];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set2 = subproblem2(&(r_05.transpose() * h0), &h0, &h4, &(-h3));
        for (q5, q4) in set2.get_all() {
            let r_34 = axis_angle(&h3, q4);
            let r_45 = axis_angle(&h4, q5);
            let hn = create_normal_vector(&h0);
            let Some(q03) =
                subproblem1(&hn, &(r_05 * r_45.transpose() * r_34.transpose() * hn), &h0)
            else {
                continue;
            };
            let r_03 = axis_angle(&h0, q03);
            let set3 = subproblem3(&p2, &(-p1), &h1, (p_15 - r_03 * (p3 + r_34 * p4)).length());
            for q2 in set3.get_all() {
                let r_12 = axis_angle(&h1, q2);
                let Some(q1) =
                    subproblem1(&(p1 + r_12 * p2), &(p_15 - r_03 * (p3 + r_34 * p4)), &h0)
                else {
                    continue;
                };
                let r_01 = axis_angle(&h0, q1);
                if let Some(q3) = subproblem1(
                    &(r_12.transpose() * r_01.transpose() * hn),
                    &(r_03.transpose() * hn),
                    &(-h2),
                ) {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                }
            }
        }
        out
    }

    fn solve_123par_45par(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];
        let p4 = self.chain.p[4];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set4 = subproblem4(&h0, &p4, &h3, h0.dot(p_15 - p1 - p2 - p3));
        for q4 in set4.get_all() {
            let r_34 = axis_angle(&h3, q4);
            let Some(q5) = subproblem1(&(r_05.transpose() * h0), &(r_34.transpose() * h0), &h4)
            else {
                continue;
            };
            let r_45 = axis_angle(&h4, q5);
            let hn = create_normal_vector(&h0);
            let Some(q03) =
                subproblem1(&hn, &(r_05 * r_45.transpose() * r_34.transpose() * hn), &h0)
            else {
                continue;
            };
            let r_03 = axis_angle(&h0, q03);
            let set3 = subproblem3(&p2, &(-p1), &h1, (p_15 - r_03 * (p3 + r_34 * p4)).length());
            for q2 in set3.get_all() {
                let r_12 = axis_angle(&h1, q2);
                let Some(q1) =
                    subproblem1(&(p1 + r_12 * p2), &(p_15 - r_03 * (p3 + r_34 * p4)), &h0)
                else {
                    continue;
                };
                let r_01 = axis_angle(&h0, q1);
                if let Some(q3) = subproblem1(
                    &(r_12.transpose() * r_01.transpose() * hn),
                    &(r_03.transpose() * hn),
                    &(-h2),
                ) {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                }
            }
        }
        out
    }

    fn solve_234par(&self, pose: &DMat4) -> Vec<crate::solver::Joints> {
        let (p_15, r_05) = self.pose_decompose(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let h4 = self.chain.h[4];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];
        let p4 = self.chain.p[4];

        let mut out: Vec<crate::solver::Joints> = Vec::with_capacity(8);
        let set4 = subproblem4(&h1, &p_15, &(-h0), h2.dot(p1 + p2 + p3 + p4));
        for q1 in set4.get_all() {
            let r_01 = axis_angle(&h0, q1);
            let Some(q5) = subproblem1(&(r_05.transpose() * r_01 * h1), &h1, &h4) else {
                continue;
            };
            let r_45 = axis_angle(&h4, q5);
            let hn = create_normal_vector(&h1);
            let Some(q14) =
                subproblem1(&hn, &(r_01.transpose() * r_05 * r_45.transpose() * hn), &h1)
            else {
                continue;
            };
            let r_14 = axis_angle(&h1, q14);
            let r_40 = (r_01 * r_14).transpose();
            let placeholder = r_40 * p_15 - r_14.transpose() * p1 - p4;
            let set3 = subproblem3(&p2, &(-p3), &(-h2), placeholder.length());
            for q3 in set3.get_all() {
                let r_32 = axis_angle(&(-h2), q3);
                let Some(q4) = subproblem1(&(r_32 * p2 + p3), &placeholder, &(-h3)) else {
                    continue;
                };
                let r_43 = axis_angle(&(-h3), q4);
                if let Some(q2) = subproblem1(&hn, &(r_14 * r_43 * r_32 * hn), &h1) {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4, q5]));
                }
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
) -> (Class5, Option<Box<R5>>) {
    let h0 = h[0];
    let h1 = h[1];
    let h2 = h[2];
    let h3 = h[3];
    let h4 = h[4];
    let p0 = p[0];
    let p1 = p[1];
    let p2 = p[2];
    let p3 = p[3];
    let p4 = p[4];

    let make_reversed = || {
        let (h_rev, p_rev) = reverse_chain(h, p);
        let p_rev_remodeled = remodel_kinematics(&h_rev, &p_rev, zt, at);
        let (class, nested) = classify(&h_rev, &p_rev_remodeled, zt, at);
        Box::new(R5 {
            chain: Chain5::from_slices(&h_rev, &p_rev_remodeled),
            zero_thresh: zt,
            class,
            reversed: nested,
        })
    };

    if h0.cross(h1).length() < zt && h1.cross(h2).length() < zt {
        if h3.cross(h4).length() < zt {
            return (Class5::FirstSecondThirdParallelFourthFifthParallel, None);
        }
        return (Class5::FirstSecondThirdParallel, None);
    } else if h2.cross(h3).length() < zt && h3.cross(h4).length() < zt {
        return (Class5::Reversed, Some(make_reversed()));
    }

    if h1.cross(h2).length() < zt && h2.cross(h3).length() < zt {
        return (Class5::SecondThirdFourthParallel, None);
    }

    if do_axes_intersect(&h3, &h4, &p4, zt, zt) {
        if do_axes_intersect(&h2, &h3, &p3, zt, zt) {
            let p03 = p0 + p1 + p2;
            let intersection = calc_intersection(&h2, &h3, &p03, &p3, zt);
            if is_point_on_axis(&h4, &(p03 + p3 + p4), &intersection, zt) {
                if do_axes_intersect(&h0, &h1, &p1, zt, zt) {
                    return (Class5::SphericalWristFirstSecondIntersecting, None);
                }
                return (Class5::SphericalWrist, None);
            }
        }
        if do_axes_intersect(&h0, &h1, &p1, zt, zt) {
            let intersection = calc_intersection(&h0, &h1, &p0, &p1, zt);
            if is_point_on_axis(&h2, &(p0 + p1 + p2), &intersection, zt) {
                return (Class5::Reversed, Some(make_reversed()));
            }
            return (
                Class5::FourthFifthIntersectingFirstSecondIntersecting,
                None,
            );
        }
        if do_axes_intersect(&h1, &h2, &p2, zt, zt) {
            return (
                Class5::FourthFifthIntersectingSecondThirdIntersecting,
                None,
            );
        }
        if h0.cross(h1).length() < zt {
            return (Class5::FourthFifthIntersectingFirstSecondParallel, None);
        }
        if h1.cross(h2).length() < zt {
            return (Class5::FourthFifthIntersectingSecondThirdParallel, None);
        }
        return (Class5::FourthFifthIntersecting, None);
    } else if do_axes_intersect(&h0, &h1, &p1, zt, zt) {
        return (Class5::Reversed, Some(make_reversed()));
    }

    if do_axes_intersect(&h2, &h3, &p3, zt, zt) && h1.cross(h2).length() < zt {
        if h3.cross(h4).length() < zt {
            return (
                Class5::ThirdFourthIntersectingSecondThirdParallelFourthFifthParallel,
                None,
            );
        }
        return (Class5::ThirdFourthIntersectingSecondThirdParallel, None);
    } else if do_axes_intersect(&h1, &h2, &p2, zt, zt) && h2.cross(h3).length() < zt {
        return (Class5::Reversed, Some(make_reversed()));
    }

    (Class5::Unknown, None)
}
