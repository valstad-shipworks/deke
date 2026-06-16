//! 3R analytical IK.

use glam::{DMat3, DMat4, DVec3};

use crate::ik_geo::subproblems::auxiliary::rot as axis_angle;
use crate::ik_geo::subproblems::{subproblem1, subproblem2, subproblem3, subproblem4};
use crate::kinematics::create_normal_vector;
use crate::solver::{Chain3, Solutions};

pub struct R3 {
    chain: Chain3,
    zero_thresh: f64,
}

impl R3 {
    pub fn new(h: &[DVec3], p: &[DVec3], zero_thresh: f64) -> Self {
        Self {
            chain: Chain3::from_slices(h, p),
            zero_thresh,
        }
    }

    pub fn solve(&self, pose: &DMat4) -> Solutions {
        let zt = self.zero_thresh;
        let p_t = pose.w_axis.truncate();
        let r_03 = DMat3::from_cols(
            pose.x_axis.truncate(),
            pose.y_axis.truncate(),
            pose.z_axis.truncate(),
        );

        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let h2 = self.chain.h[2];
        let p0 = self.chain.p[0];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];
        let p3 = self.chain.p[3];

        let p_13 = p_t - p0 - r_03 * p3;

        let mut solution_t_12: Vec<[f64; 2]> = Vec::new();
        let mut out: Solutions = Solutions::new();

        let h12_cross = h1.cross(h2);
        let h01_cross = h0.cross(h1);

        if h12_cross.dot(p2).abs() < zt && h12_cross.length() > zt {
            if p1.length() < zt && p2.length() < zt {
                let d = h0.dot(r_03 * h2);
                let set = subproblem4(&h0, &h2, &h1, d);
                for q2 in set.get_all() {
                    let r12 = axis_angle(&h1, q2);
                    let Some(q1) = subproblem1(&(r_03 * h2), &(r12 * h2), &(-h0)) else {
                        continue;
                    };
                    let r01 = axis_angle(&h0, q1);
                    let hn = create_normal_vector(&h2);
                    let Some(q3) =
                        subproblem1(&hn, &(r12.transpose() * r01.transpose() * r_03 * hn), &h2)
                    else {
                        continue;
                    };
                    out.push(crate::solver::pack(&[q1, q2, q3]));
                }
                return out;
            }
            let Some(q1) = subproblem1(&p1, &p_13, &h0) else {
                return out;
            };
            let r01 = axis_angle(&h0, q1);
            let Some(q2) = subproblem1(&h2, &(r01.transpose() * r_03 * h2), &h1) else {
                return out;
            };
            solution_t_12.push([q1, q2]);
        } else if h01_cross.length() < zt {
            let set = subproblem3(&p2, &(-p1), &h1, p_13.length());
            for q2 in set.get_all() {
                let r12 = axis_angle(&h1, q2);
                if let Some(q1) = subproblem1(&p_13, &(p1 + r12 * p2), &(-h0)) {
                    solution_t_12.push([q1, q2]);
                }
            }
        } else if h01_cross.dot(p1).abs() < zt {
            let set = subproblem2(&p_13, &p2, &(-h0), &h1);
            for (q1, q2) in set.get_all() {
                solution_t_12.push([q1, q2]);
            }
        } else {
            let set = subproblem3(&p2, &(-p1), &h1, p_13.length());
            for q2 in set.get_all() {
                let r12 = axis_angle(&h1, q2);
                if let Some(q1) = subproblem1(&p_13, &(p1 + r12 * p2), &(-h0)) {
                    solution_t_12.push([q1, q2]);
                }
            }
        }

        for [q1, q2] in solution_t_12 {
            let r12 = axis_angle(&h1, q2);
            let r01 = axis_angle(&h0, q1);
            let hn = create_normal_vector(&h2);
            if let Some(q3) =
                subproblem1(&hn, &(r12.transpose() * r01.transpose() * r_03 * hn), &h2)
            {
                out.push(crate::solver::pack(&[q1, q2, q3]));
            }
        }
        out
    }
}
