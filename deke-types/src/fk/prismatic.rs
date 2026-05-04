use glam::{Affine3A, Vec3A};

use crate::SRobotQ;

use super::FKChain;

/// Wraps an `FKChain<N>` and prepends a prismatic (linear) joint, producing
/// an `FKChain<M>` where `M = N + 1`.
///
/// The prismatic joint always acts first in the kinematic chain — it
/// translates the entire arm along `axis` (world frame).  The
/// `q_index_first` flag only controls where the prismatic value is read
/// from in `SRobotQ<M>`: when `true` it is `q[0]`, when `false` it is
/// `q[M-1]`.
///
/// Jacobian columns for the prismatic joint are `[axis; 0]` (pure linear,
/// no angular contribution).  Because the prismatic uniformly shifts all
/// positions, the revolute Jacobian columns are identical to the inner
/// chain's.
#[derive(Debug, Clone)]
pub struct PrismaticFK<const M: usize, const N: usize, FK: FKChain<N>> {
    inner: FK,
    axis: Vec3A,
    q_index_first: bool,
}

impl<const M: usize, const N: usize, FK: FKChain<N>> PrismaticFK<M, N, FK> {
    pub const fn new(inner: FK, axis: Vec3A, q_index_first: bool) -> Self {
        const { assert!(M == N + 1, "M must equal N + 1") };
        Self {
            inner,
            axis,
            q_index_first,
        }
    }

    pub fn inner(&self) -> &FK {
        &self.inner
    }

    pub fn axis(&self) -> Vec3A {
        self.axis
    }

    pub fn q_index_first(&self) -> bool {
        self.q_index_first
    }

    fn split_q(&self, q: &SRobotQ<M>) -> (f32, SRobotQ<N>) {
        let mut inner = [0.0f32; N];
        if self.q_index_first {
            inner.copy_from_slice(&q.0[1..M]);
            (q.0[0], SRobotQ(inner))
        } else {
            inner.copy_from_slice(&q.0[..N]);
            (q.0[M - 1], SRobotQ(inner))
        }
    }

    fn prismatic_col(&self) -> usize {
        if self.q_index_first { 0 } else { N }
    }

    fn revolute_offset(&self) -> usize {
        if self.q_index_first { 1 } else { 0 }
    }
}

impl<const M: usize, const N: usize, FK: FKChain<N>> FKChain<M> for PrismaticFK<M, N, FK> {
    type Error = FK::Error;

    fn base_tf(&self) -> Affine3A {
        self.inner.base_tf()
    }

    fn fk(&self, q: &SRobotQ<M>) -> Result<[Affine3A; M], Self::Error> {
        let (q_p, inner_q) = self.split_q(q);
        let offset = q_p * self.axis;
        let inner_frames = self.inner.fk(&inner_q)?;
        let mut out = [Affine3A::IDENTITY; M];

        out[0] = Affine3A::from_translation(offset.into());
        for i in 0..N {
            let mut f = inner_frames[i];
            f.translation += offset;
            out[i + 1] = f;
        }

        Ok(out)
    }

    fn fk_end(&self, q: &SRobotQ<M>) -> Result<Affine3A, Self::Error> {
        let (q_p, inner_q) = self.split_q(q);
        let mut end = self.inner.fk_end(&inner_q)?;
        end.translation += q_p * self.axis;
        Ok(end)
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<M>,
    ) -> Result<([Vec3A; M], [Vec3A; M], Vec3A), Self::Error> {
        let (q_p, inner_q) = self.split_q(q);
        let offset = q_p * self.axis;
        let (inner_axes, inner_pos, inner_p_ee) = self.inner.joint_axes_positions(&inner_q)?;

        let mut axes = [Vec3A::ZERO; M];
        let mut positions = [Vec3A::ZERO; M];

        axes[0] = self.axis;
        for i in 0..N {
            axes[i + 1] = inner_axes[i];
            positions[i + 1] = inner_pos[i] + offset;
        }

        Ok((axes, positions, inner_p_ee + offset))
    }

    fn jacobian(&self, q: &SRobotQ<M>) -> Result<[[f32; M]; 6], Self::Error> {
        let (_q_p, inner_q) = self.split_q(q);
        let inner_j = self.inner.jacobian(&inner_q)?;
        let p_col = self.prismatic_col();
        let r_off = self.revolute_offset();

        let mut j = [[0.0f32; M]; 6];
        j[0][p_col] = self.axis.x;
        j[1][p_col] = self.axis.y;
        j[2][p_col] = self.axis.z;

        for row in 0..6 {
            for col in 0..N {
                j[row][col + r_off] = inner_j[row][col];
            }
        }

        Ok(j)
    }

    fn jacobian_dot(
        &self,
        q: &SRobotQ<M>,
        qdot: &SRobotQ<M>,
    ) -> Result<[[f32; M]; 6], Self::Error> {
        let (_q_p, inner_q) = self.split_q(q);
        let (_qdot_p, inner_qdot) = self.split_q(qdot);
        let inner_jd = self.inner.jacobian_dot(&inner_q, &inner_qdot)?;
        let r_off = self.revolute_offset();

        let mut jd = [[0.0f32; M]; 6];
        for row in 0..6 {
            for col in 0..N {
                jd[row][col + r_off] = inner_jd[row][col];
            }
        }

        Ok(jd)
    }

    fn jacobian_ddot(
        &self,
        q: &SRobotQ<M>,
        qdot: &SRobotQ<M>,
        qddot: &SRobotQ<M>,
    ) -> Result<[[f32; M]; 6], Self::Error> {
        let (_q_p, inner_q) = self.split_q(q);
        let (_qdot_p, inner_qdot) = self.split_q(qdot);
        let (_qddot_p, inner_qddot) = self.split_q(qddot);
        let inner_jdd = self
            .inner
            .jacobian_ddot(&inner_q, &inner_qdot, &inner_qddot)?;
        let r_off = self.revolute_offset();

        let mut jdd = [[0.0f32; M]; 6];
        for row in 0..6 {
            for col in 0..N {
                jdd[row][col + r_off] = inner_jdd[row][col];
            }
        }

        Ok(jdd)
    }
}
