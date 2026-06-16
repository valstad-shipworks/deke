//! 1R analytical IK.

use glam::{DMat4, DVec3};

use smallvec::smallvec;

use crate::ik_geo::subproblems::subproblem1;
use crate::solver::{Chain1, Solutions};

pub struct R1 {
    chain: Chain1,
}

impl R1 {
    pub fn new(h: &[DVec3], p: &[DVec3]) -> Self {
        Self {
            chain: Chain1::from_slices(h, p),
        }
    }

    pub fn solve(&self, pose: &DMat4) -> Solutions {
        let p_t = pose.w_axis.truncate();
        let p_1t = p_t - self.chain.p[0];
        match subproblem1(&self.chain.p[1], &p_1t, &self.chain.h[0]) {
            Some(theta) => smallvec![crate::solver::pack(&[theta])],
            None => Solutions::new(),
        }
    }
}
