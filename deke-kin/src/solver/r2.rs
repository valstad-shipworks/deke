//! 2R analytical IK.

use glam::{DMat4, DVec3};

use crate::ik_geo::subproblems::auxiliary::rot as axis_angle;
use crate::ik_geo::subproblems::{subproblem1, subproblem2, subproblem3};
use crate::remodel::do_axes_intersect;
use crate::solver::{Chain2, Solutions};

pub struct R2 {
    chain: Chain2,
    zero_thresh: f64,
}

impl R2 {
    pub fn new(h: &[DVec3], p: &[DVec3], zero_thresh: f64) -> Self {
        Self {
            chain: Chain2::from_slices(h, p),
            zero_thresh,
        }
    }

    pub fn solve(&self, pose: &DMat4) -> Solutions {
        let p_t = pose.w_axis.truncate();
        let p_1ee = p_t - self.chain.p[0];

        let h0 = self.chain.h[0];
        let h1 = self.chain.h[1];
        let p1 = self.chain.p[1];
        let p2 = self.chain.p[2];

        let mut out: Solutions = Solutions::new();

        if do_axes_intersect(&h0, &h1, &p1, self.zero_thresh, self.zero_thresh) {
            let set = subproblem2(&p_1ee, &p2, &(-h0), &h1);
            for (t1, t2) in set.get_all() {
                out.push(crate::solver::pack(&[t1, t2]));
            }
        } else {
            let set3 = subproblem3(&p2, &(-p1), &h1, p_1ee.length());
            for q2 in set3.get_all() {
                let r12 = axis_angle(&h1, q2);
                if let Some(q1) = subproblem1(&(p1 + r12 * p2), &p_1ee, &h0) {
                    out.push(crate::solver::pack(&[q1, q2]));
                }
            }
        }
        out
    }
}
