use glam_traits_ext::{FloatAffine, FloatVec, TAffine3, TVec3};

use crate::SRobotQ;

use super::{AAffine3, AVec3, FKChain, FKScalar};

/// Wraps an `FKChain<N, F>` and prepends a prismatic (linear) joint, producing
/// an `FKChain<M, F>` where `M = N + 1`.
///
/// The prismatic joint always acts first in the kinematic chain — it
/// translates the entire arm along `axis` (world frame).  The
/// `q_index_first` flag only controls where the prismatic value is read
/// from in `SRobotQ<M, F>`: when `true` it is `q[0]`, when `false` it is
/// `q[M-1]`.
///
/// Jacobian columns for the prismatic joint are `[axis; 0]` (pure linear,
/// no angular contribution).  Because the prismatic uniformly shifts all
/// positions, the revolute Jacobian columns are identical to the inner
/// chain's.
#[derive(Debug, Clone)]
pub struct PrismaticFK<const M: usize, const N: usize, F: FKScalar, FK: FKChain<N, F>> {
    inner: FK,
    axis: AVec3<F>,
    q_index_first: bool,
}

impl<const M: usize, const N: usize, F: FKScalar, FK: FKChain<N, F>> PrismaticFK<M, N, F, FK> {
    pub fn new(inner: FK, axis: AVec3<F>, q_index_first: bool) -> Self {
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

    pub fn axis(&self) -> AVec3<F> {
        self.axis
    }

    pub fn q_index_first(&self) -> bool {
        self.q_index_first
    }

    fn split_q(&self, q: &SRobotQ<M, F>) -> (F, SRobotQ<N, F>) {
        let mut inner = [F::zero(); N];
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

impl<const M: usize, const N: usize, F: FKScalar, FK: FKChain<N, F>> FKChain<M, F>
    for PrismaticFK<M, N, F, FK>
{
    type Error = FK::Error;

    fn base_tf(&self) -> AAffine3<F> {
        self.inner.base_tf()
    }

    fn fk(&self, q: &SRobotQ<M, F>) -> Result<[AAffine3<F>; M], Self::Error> {
        let (q_p, inner_q) = self.split_q(q);
        let offset = self.axis * q_p;
        let inner_frames = self.inner.fk(&inner_q)?;
        let mut out = [AAffine3::<F>::IDENTITY; M];

        out[0] = AAffine3::<F>::from_translation(offset);
        for i in 0..N {
            let mut f = inner_frames[i];
            // Apply translation by `offset` to inner frame: post-multiply with translate(offset).
            // Equivalent to setting f.translation += offset.
            f = AAffine3::<F>::from_mat3_translation(f.matrix3(), f.translation() + offset);
            out[i + 1] = f;
        }

        Ok(out)
    }

    fn fk_end(&self, q: &SRobotQ<M, F>) -> Result<AAffine3<F>, Self::Error> {
        let (q_p, inner_q) = self.split_q(q);
        let end = self.inner.fk_end(&inner_q)?;
        let offset = self.axis * q_p;
        Ok(AAffine3::<F>::from_mat3_translation(
            end.matrix3(),
            end.translation() + offset,
        ))
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<M, F>,
    ) -> Result<([AVec3<F>; M], [AVec3<F>; M], AVec3<F>), Self::Error> {
        let (q_p, inner_q) = self.split_q(q);
        let offset = self.axis * q_p;
        let (inner_axes, inner_pos, inner_p_ee) = self.inner.joint_axes_positions(&inner_q)?;

        let mut axes = [AVec3::<F>::ZERO; M];
        let mut positions = [AVec3::<F>::ZERO; M];

        axes[0] = self.axis;
        for i in 0..N {
            axes[i + 1] = inner_axes[i];
            positions[i + 1] = inner_pos[i] + offset;
        }

        Ok((axes, positions, inner_p_ee + offset))
    }

    fn jacobian(&self, q: &SRobotQ<M, F>) -> Result<[[F; M]; 6], Self::Error> {
        let (_q_p, inner_q) = self.split_q(q);
        let inner_j = self.inner.jacobian(&inner_q)?;
        let p_col = self.prismatic_col();
        let r_off = self.revolute_offset();

        let mut j = [[F::zero(); M]; 6];
        j[0][p_col] = self.axis.x();
        j[1][p_col] = self.axis.y();
        j[2][p_col] = self.axis.z();

        for row in 0..6 {
            for col in 0..N {
                j[row][col + r_off] = inner_j[row][col];
            }
        }

        Ok(j)
    }

    fn jacobian_dot(
        &self,
        q: &SRobotQ<M, F>,
        qdot: &SRobotQ<M, F>,
    ) -> Result<[[F; M]; 6], Self::Error> {
        let (_q_p, inner_q) = self.split_q(q);
        let (_qdot_p, inner_qdot) = self.split_q(qdot);
        let inner_jd = self.inner.jacobian_dot(&inner_q, &inner_qdot)?;
        let r_off = self.revolute_offset();

        let mut jd = [[F::zero(); M]; 6];
        for row in 0..6 {
            for col in 0..N {
                jd[row][col + r_off] = inner_jd[row][col];
            }
        }

        Ok(jd)
    }

    fn jacobian_ddot(
        &self,
        q: &SRobotQ<M, F>,
        qdot: &SRobotQ<M, F>,
        qddot: &SRobotQ<M, F>,
    ) -> Result<[[F; M]; 6], Self::Error> {
        let (_q_p, inner_q) = self.split_q(q);
        let (_qdot_p, inner_qdot) = self.split_q(qdot);
        let (_qddot_p, inner_qddot) = self.split_q(qddot);
        let inner_jdd = self
            .inner
            .jacobian_ddot(&inner_q, &inner_qdot, &inner_qddot)?;
        let r_off = self.revolute_offset();

        let mut jdd = [[F::zero(); M]; 6];
        for row in 0..6 {
            for col in 0..N {
                jdd[row][col + r_off] = inner_jdd[row][col];
            }
        }

        Ok(jdd)
    }
}

impl<const M: usize, const N: usize, FK32, FK64> From<PrismaticFK<M, N, f32, FK32>>
    for PrismaticFK<M, N, f64, FK64>
where
    FK32: FKChain<N, f32>,
    FK64: FKChain<N, f64> + From<FK32>,
{
    #[inline]
    fn from(p: PrismaticFK<M, N, f32, FK32>) -> Self {
        PrismaticFK {
            inner: FK64::from(p.inner),
            axis: p.axis.as_dvec3(),
            q_index_first: p.q_index_first,
        }
    }
}

impl<const M: usize, const N: usize, FK64, FK32> From<PrismaticFK<M, N, f64, FK64>>
    for PrismaticFK<M, N, f32, FK32>
where
    FK64: FKChain<N, f64>,
    FK32: FKChain<N, f32> + From<FK64>,
{
    #[inline]
    fn from(p: PrismaticFK<M, N, f64, FK64>) -> Self {
        PrismaticFK {
            inner: FK32::from(p.inner),
            axis: p.axis.as_vec3a(),
            q_index_first: p.q_index_first,
        }
    }
}
