//! 4R analytical IK with kinematic-class detection.

use glam::{DMat3, DMat4, DVec3};

use crate::ik_geo::subproblems::auxiliary::rot as axis_angle;
use crate::ik_geo::subproblems::{subproblem1, subproblem2, subproblem3, subproblem4, subproblem5};
use crate::kinematics::{create_normal_vector, inverse_homogeneous, reverse_chain};
use crate::remodel::{calc_intersection, is_point_on_axis, remodel_kinematics};
use crate::solver::{Chain4, Joints};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum Class4 {
    ThirdFourthIntersecting,
    SecondThirdIntersecting,
    FirstSecondParallel,
    SecondThirdParallel,
    NoneParallelNoneIntersecting,
    FirstTwoLastTwoIntersecting,
    SphericalWrist,
    Reversed,
    Unknown,
}

impl Class4 {
    fn name(self) -> &'static str {
        match self {
            Class4::ThirdFourthIntersecting => "4R-THIRD_FOURTH_INTERSECTING",
            Class4::SecondThirdIntersecting => "4R-SECOND_THIRD_INTERSECTING",
            Class4::FirstSecondParallel => "4R-FIRST_SECOND_PARALLEL",
            Class4::SecondThirdParallel => "4R-SECOND_THIRD_PARALLEL",
            Class4::NoneParallelNoneIntersecting => "4R-NONE_PARALLEL_NONE_INTERSECTING",
            Class4::FirstTwoLastTwoIntersecting => "4R-FIRST_TWO_LAST_TWO_INTERSECTING",
            Class4::SphericalWrist => "4R-SPHERICAL_WRIST",
            Class4::Reversed => "4R-REVERSED",
            Class4::Unknown => "4R-Unknown Kinematic Class",
        }
    }
}

pub struct R4 {
    chain: Chain4,
    #[allow(dead_code)]
    zero_thresh: f64,
    class: Class4,
    reversed: Option<Box<R4>>,
}

impl R4 {
    pub fn new(h: &[DVec3], p: &[DVec3], zero_thresh: f64, axis_thresh: f64) -> Self {
        let (class, reversed) = classify(h, p, zero_thresh, axis_thresh);
        Self {
            chain: Chain4::from_slices(h, p),
            zero_thresh,
            class,
            reversed,
        }
    }

    pub fn has_known_decomposition(&self) -> bool {
        !matches!(self.class, Class4::Unknown)
    }

    pub fn kinematic_family(&self) -> String {
        match (&self.class, &self.reversed) {
            (Class4::Reversed, Some(rev)) => rev.kinematic_family(),
            _ => self.class.name().to_string(),
        }
    }

    pub fn solve(&self, pose: &DMat4) -> Vec<Joints> {
        match self.class {
            Class4::ThirdFourthIntersecting => self.solve_third_fourth_intersecting(pose),
            Class4::SecondThirdIntersecting => self.solve_second_third_intersecting(pose),
            Class4::FirstSecondParallel => self.solve_first_second_parallel(pose),
            Class4::SecondThirdParallel => self.solve_second_third_parallel(pose),
            Class4::NoneParallelNoneIntersecting => self.solve_none_parallel(pose),
            Class4::FirstTwoLastTwoIntersecting => self.solve_first_two_last_two(pose),
            Class4::SphericalWrist => self.solve_spherical_wrist(pose),
            Class4::Reversed => {
                if let Some(rev) = &self.reversed {
                    let inv = inverse_homogeneous(pose);
                    let mut sols = rev.solve(&inv);
                    for q in sols.iter_mut() {
                        q.0[..4].reverse();
                    }
                    sols
                } else {
                    Vec::new()
                }
            }
            Class4::Unknown => Vec::new(),
        }
    }

    fn p14(&self, pose: &DMat4) -> (DVec3, DMat3) {
        let p_t = pose.w_axis.truncate();
        let r_04 = DMat3::from_cols(
            pose.x_axis.truncate(),
            pose.y_axis.truncate(),
            pose.z_axis.truncate(),
        );
        let p_14 = p_t - self.chain.p[0] - r_04 * self.chain.p[4];
        (p_14, r_04)
    }

    fn solve_third_fourth_intersecting(&self, pose: &DMat4) -> Vec<Joints> {
        let (p_14, r_04) = self.p14(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];

        let mut out: Vec<Joints> = Vec::with_capacity(8);
        let sp3_q1 = subproblem3(&p_14, &p1, &(-h0), p2.length());
        for q1 in sp3_q1.get_all() {
            let sp3_q2 = subproblem3(&p2, &(-p1), &h1, p_14.length());
            for q2 in sp3_q2.get_all() {
                let r_21 = axis_angle(&(-h1), q2);
                let r_10 = axis_angle(&(-h0), q1);
                let hn = create_normal_vector(&h2);
                let set = subproblem2(&(r_21 * r_10 * r_04 * hn), &hn, &(-h2), &h3);
                for (t3, t4) in set.get_all() {
                    out.push(crate::solver::pack(&[q1, q2, t3, t4]));
                }
            }
        }
        out
    }

    fn solve_second_third_intersecting(&self, pose: &DMat4) -> Vec<Joints> {
        let (p_14, r_04) = self.p14(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let p1 = self.chain.p[1];
        let p3 = self.chain.p[3];

        let mut out: Vec<Joints> = Vec::with_capacity(8);
        let sp3 = subproblem3(&p1, &p_14, &h0, p3.length());
        for q1 in sp3.get_all() {
            let r_10 = axis_angle(&(-h0), q1);
            let set = subproblem2(&(r_10 * p_14 - p1), &p3, &(-h1), &h2);
            let hn = create_normal_vector(&h3);
            for (q2, q3) in set.get_all() {
                let r_12 = axis_angle(&h1, q2);
                let r_23 = axis_angle(&h2, q3);
                if let Some(q4) = subproblem1(
                    &hn,
                    &(r_04.transpose() * r_10.transpose() * r_12 * r_23 * hn),
                    &(-h3),
                ) {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4]));
                }
            }
        }
        out
    }

    fn solve_first_second_parallel(&self, pose: &DMat4) -> Vec<Joints> {
        let (p_14, r_04) = self.p14(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let mut out: Vec<Joints> = Vec::with_capacity(8);
        let sp4_q3 = subproblem4(&h0, &p3, &h2, h0.dot(p_14 - p1 - p2));
        for q3 in sp4_q3.get_all() {
            let r_23 = axis_angle(&h2, q3);
            let sp3_q1 = subproblem3(&p_14, &p1, &(-h0), (p2 + r_23 * p3).length());
            for q1 in sp3_q1.get_all() {
                let r_01 = axis_angle(&h0, q1);
                let Some(q2) =
                    subproblem1(&(p2 + r_23 * p3), &(r_01.transpose() * p_14 - p1), &h1)
                else {
                    continue;
                };
                let r_12 = axis_angle(&h1, q2);
                let hn = create_normal_vector(&h3);
                if let Some(q4) = subproblem1(
                    &hn,
                    &(r_04.transpose() * r_01 * r_12 * r_23 * hn),
                    &(-h3),
                ) {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4]));
                }
            }
        }
        out
    }

    fn solve_second_third_parallel(&self, pose: &DMat4) -> Vec<Joints> {
        let (p_14, r_04) = self.p14(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let mut out: Vec<Joints> = Vec::with_capacity(8);
        let sp4 = subproblem4(&h1, &p_14, &(-h0), h1.dot(p1 + p2 + p3));
        for q1 in sp4.get_all() {
            let r_10 = axis_angle(&(-h0), q1);
            let sp3 = subproblem3(&p3, &(-p2), &h2, (r_10 * p_14 - p1).length());
            for q3 in sp3.get_all() {
                let r_23 = axis_angle(&h2, q3);
                let Some(q2) = subproblem1(&(p2 + r_23 * p3), &(r_10 * p_14 - p1), &h1) else {
                    continue;
                };
                let r_12 = axis_angle(&h1, q2);
                let hn = create_normal_vector(&h3);
                if let Some(q4) = subproblem1(
                    &hn,
                    &(r_04.transpose() * r_10.transpose() * r_12 * r_23 * hn),
                    &(-h3),
                ) {
                    out.push(crate::solver::pack(&[q1, q2, q3, q4]));
                }
            }
        }
        out
    }

    fn solve_none_parallel(&self, pose: &DMat4) -> Vec<Joints> {
        let (p_14, r_04) = self.p14(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let mut out: Vec<Joints> = Vec::with_capacity(8);
        let set = subproblem5(&(-p1), &p_14, &p2, &p3, &(-h0), &h1, &h2);
        let hn = create_normal_vector(&h3);
        for (q1, q2, q3) in set.get_all() {
            let r_01 = axis_angle(&h0, q1);
            let r_12 = axis_angle(&h1, q2);
            let r_23 = axis_angle(&h2, q3);
            if let Some(q4) = subproblem1(
                &hn,
                &(r_23.transpose() * r_12.transpose() * r_01.transpose() * r_04 * hn),
                &h3,
            ) {
                out.push(crate::solver::pack(&[q1, q2, q3, q4]));
            }
        }
        out
    }

    fn solve_first_two_last_two(&self, pose: &DMat4) -> Vec<Joints> {
        let (p_14, r_04) = self.p14(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let p2 = self.chain.p[2];

        let mut out: Vec<Joints> = Vec::with_capacity(8);
        let set12 = subproblem2(&p_14, &p2, &(-h0), &h1);
        for (q1, q2) in set12.get_all() {
            let r_01 = axis_angle(&h0, q1);
            let r_12 = axis_angle(&h1, q2);
            let hn = h3.cross(r_04.transpose() * r_01 * r_12 * h2);
            let set34 =
                subproblem2(&(r_12.transpose() * r_01.transpose() * r_04 * hn), &hn, &(-h2), &h3);
            for (q3, q4) in set34.get_all() {
                out.push(crate::solver::pack(&[q1, q2, q3, q4]));
            }
        }
        out
    }

    fn solve_spherical_wrist(&self, pose: &DMat4) -> Vec<Joints> {
        let (p_14, r_04) = self.p14(pose);
        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let h3 = self.chain.h[3];
        let p1 = self.chain.p[1];

        let mut out: Vec<Joints> = Vec::with_capacity(8);
        let Some(q1) = subproblem1(&p1, &p_14, &h0) else {
            return out;
        };
        let r_01 = axis_angle(&h0, q1);
        let set = subproblem2(&(r_01.transpose() * r_04 * h3), &h3, &(-h1), &h2);
        let hn = create_normal_vector(&h3);
        for (q2, q3) in set.get_all() {
            let r_21 = axis_angle(&(-h1), q2);
            let r_32 = axis_angle(&(-h2), q3);
            if let Some(q4) = subproblem1(&hn, &(r_32 * r_21 * r_01.transpose() * r_04 * hn), &h3) {
                out.push(crate::solver::pack(&[q1, q2, q3, q4]));
            }
        }
        out
    }
}

fn classify(h: &[DVec3], p: &[DVec3], zt: f64, at: f64) -> (Class4, Option<Box<R4>>) {
    let h0 = h[0];
    let h1 = h[1];
    let h2 = h[2];
    let h3 = h[3];
    let p0 = p[0];
    let p1 = p[1];
    let p2 = p[2];
    let p3 = p[3];

    let make_reversed = || {
        let (h_rev, p_rev) = reverse_chain(h, p);
        let p_rev_remodeled = remodel_kinematics(&h_rev, &p_rev, zt, at);
        let (class, nested) = classify(&h_rev, &p_rev_remodeled, zt, at);
        Box::new(R4 {
            chain: Chain4::from_slices(&h_rev, &p_rev_remodeled),
            zero_thresh: zt,
            class,
            reversed: nested,
        })
    };

    if h0.cross(h1).length() >= zt && h0.cross(h1).dot(p1).abs() < zt {
        let intersection = calc_intersection(&h0, &h1, &p0, &p1, zt);
        if is_point_on_axis(&h2, &(p0 + p1 + p2), &intersection, zt) {
            return (Class4::Reversed, Some(make_reversed()));
        } else if h2.cross(h3).length() >= zt && h2.cross(h3).dot(p3).abs() < zt {
            return (Class4::FirstTwoLastTwoIntersecting, None);
        }
        return (Class4::Reversed, Some(make_reversed()));
    }

    if h1.cross(h2).length() >= zt && h1.cross(h2).dot(p2).abs() < zt {
        let p02 = p0 + p1;
        let intersection = calc_intersection(&h1, &h2, &p02, &p2, zt);
        if is_point_on_axis(&h3, &(p02 + p2 + p3), &intersection, zt) {
            return (Class4::SphericalWrist, None);
        }
        return (Class4::SecondThirdIntersecting, None);
    }

    if h2.cross(h3).length() >= zt && h2.cross(h3).dot(p3).abs() < zt {
        return (Class4::ThirdFourthIntersecting, None);
    }

    if h0.cross(h1).length() > zt {
        if h1.cross(h2).length() > zt {
            if h2.cross(h3).length() < zt {
                return (Class4::Reversed, Some(make_reversed()));
            }
            return (Class4::NoneParallelNoneIntersecting, None);
        }
        return (Class4::SecondThirdParallel, None);
    } else if h1.cross(h2).length() > zt {
        return (Class4::FirstSecondParallel, None);
    } else {
        return (Class4::Reversed, Some(make_reversed()));
    }
}
