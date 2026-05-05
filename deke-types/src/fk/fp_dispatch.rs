use crate::SRobotQ;

use super::{AAffine3, AVec3, FKChain};

/// FK chain wrapper that holds both an `f32` and an `f64` representation of
/// the same kinematic chain and dispatches to the correct precision when an
/// `FKChain<N, F>` method is invoked.
///
/// The two inner chains are expected to describe the same robot — typically
/// the `f64` is canonical and the `f32` is derived from it (see
/// [`FPDispatch::from_f64`]). Callers that already have both built can use
/// [`FPDispatch::new`].
///
/// Use this when the same robot needs to be consumed by stages with different
/// precision requirements: e.g. an RRT planner runs on `f32` for SIMD speed,
/// while a TOPP retimer runs on `f64` for solver stability. A single
/// `FPDispatch` can be passed to both — each picks up the precision-correct
/// `FKChain` impl via type inference.
///
/// ```
/// use deke_types::{DHChain, DHJoint, FKChain, FPDispatch, SRobotQ};
///
/// // Author the robot once in f64 (e.g. from a const URDF table).
/// const ARM: DHChain<2, f64> = DHChain::<2, f64>::new_f64([
///     DHJoint { a: 1.0, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
///     DHJoint { a: 1.0, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
/// ]);
///
/// // Derive the f32 chain via the cheap From cast and bundle them.
/// let dispatch: FPDispatch<2, DHChain<2, f32>, DHChain<2, f64>> =
///     FPDispatch::from_f64(ARM);
///
/// // f32 consumers get the SIMD path:
/// let q32 = SRobotQ::<2, f32>::from_array([0.5, -0.3]);
/// let _ = <FPDispatch<_, _, _> as FKChain<2, f32>>::fk_end(&dispatch, &q32).unwrap();
///
/// // f64 consumers get the precise path:
/// let q64 = SRobotQ::<2, f64>::from_array([0.5, -0.3]);
/// let _ = <FPDispatch<_, _, _> as FKChain<2, f64>>::fk_end(&dispatch, &q64).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FPDispatch<const N: usize, FK32, FK64>
where
    FK32: FKChain<N, f32>,
    FK64: FKChain<N, f64>,
{
    f32_chain: FK32,
    f64_chain: FK64,
}

impl<const N: usize, FK32, FK64> FPDispatch<N, FK32, FK64>
where
    FK32: FKChain<N, f32>,
    FK64: FKChain<N, f64>,
{
    /// Build from explicit f32 and f64 chains. The two are expected to encode
    /// the same robot; nothing checks that here.
    pub fn new(f32_chain: FK32, f64_chain: FK64) -> Self {
        Self { f32_chain, f64_chain }
    }

    pub fn f32_chain(&self) -> &FK32 {
        &self.f32_chain
    }

    pub fn f64_chain(&self) -> &FK64 {
        &self.f64_chain
    }
}

impl<const N: usize, FK32, FK64> FPDispatch<N, FK32, FK64>
where
    FK32: FKChain<N, f32> + From<FK64>,
    FK64: FKChain<N, f64> + Clone,
{
    /// Build from only the f64 chain by deriving the f32 chain via `From`.
    /// Available when the f32 chain implements `From<FK64>` (which it does
    /// for the leaf chain types `DHChain`, `HPChain`, and `URDFChain`).
    pub fn from_f64(f64_chain: FK64) -> Self {
        let f32_chain = FK32::from(f64_chain.clone());
        Self { f32_chain, f64_chain }
    }
}

impl<const N: usize, FK32, FK64> FKChain<N, f32> for FPDispatch<N, FK32, FK64>
where
    FK32: FKChain<N, f32>,
    FK64: FKChain<N, f64>,
{
    type Error = FK32::Error;

    fn base_tf(&self) -> AAffine3<f32> {
        self.f32_chain.base_tf()
    }

    fn max_reach(&self) -> Result<f32, Self::Error> {
        self.f32_chain.max_reach()
    }

    fn fk(&self, q: &SRobotQ<N, f32>) -> Result<[AAffine3<f32>; N], Self::Error> {
        self.f32_chain.fk(q)
    }

    fn fk_end(&self, q: &SRobotQ<N, f32>) -> Result<AAffine3<f32>, Self::Error> {
        self.f32_chain.fk_end(q)
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N, f32>,
    ) -> Result<([AVec3<f32>; N], [AVec3<f32>; N], AVec3<f32>), Self::Error> {
        self.f32_chain.joint_axes_positions(q)
    }

    fn jacobian(&self, q: &SRobotQ<N, f32>) -> Result<[[f32; N]; 6], Self::Error> {
        self.f32_chain.jacobian(q)
    }

    fn jacobian_dot(
        &self,
        q: &SRobotQ<N, f32>,
        qdot: &SRobotQ<N, f32>,
    ) -> Result<[[f32; N]; 6], Self::Error> {
        self.f32_chain.jacobian_dot(q, qdot)
    }

    fn jacobian_ddot(
        &self,
        q: &SRobotQ<N, f32>,
        qdot: &SRobotQ<N, f32>,
        qddot: &SRobotQ<N, f32>,
    ) -> Result<[[f32; N]; 6], Self::Error> {
        self.f32_chain.jacobian_ddot(q, qdot, qddot)
    }
}

impl<const N: usize, FK32, FK64> FKChain<N, f64> for FPDispatch<N, FK32, FK64>
where
    FK32: FKChain<N, f32>,
    FK64: FKChain<N, f64>,
{
    type Error = FK64::Error;

    fn base_tf(&self) -> AAffine3<f64> {
        self.f64_chain.base_tf()
    }

    fn max_reach(&self) -> Result<f64, Self::Error> {
        self.f64_chain.max_reach()
    }

    fn fk(&self, q: &SRobotQ<N, f64>) -> Result<[AAffine3<f64>; N], Self::Error> {
        self.f64_chain.fk(q)
    }

    fn fk_end(&self, q: &SRobotQ<N, f64>) -> Result<AAffine3<f64>, Self::Error> {
        self.f64_chain.fk_end(q)
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N, f64>,
    ) -> Result<([AVec3<f64>; N], [AVec3<f64>; N], AVec3<f64>), Self::Error> {
        self.f64_chain.joint_axes_positions(q)
    }

    fn jacobian(&self, q: &SRobotQ<N, f64>) -> Result<[[f64; N]; 6], Self::Error> {
        self.f64_chain.jacobian(q)
    }

    fn jacobian_dot(
        &self,
        q: &SRobotQ<N, f64>,
        qdot: &SRobotQ<N, f64>,
    ) -> Result<[[f64; N]; 6], Self::Error> {
        self.f64_chain.jacobian_dot(q, qdot)
    }

    fn jacobian_ddot(
        &self,
        q: &SRobotQ<N, f64>,
        qdot: &SRobotQ<N, f64>,
        qddot: &SRobotQ<N, f64>,
    ) -> Result<[[f64; N]; 6], Self::Error> {
        self.f64_chain.jacobian_ddot(q, qdot, qddot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DHChain, DHJoint};
    use glam_traits_ext::{TAffine3, TVec3};

    #[test]
    fn dispatch_routes_to_correct_precision() {
        const F32_CHAIN: DHChain<2, f32> = DHChain::<2, f32>::new([
            DHJoint { a: 1.0, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
            DHJoint { a: 1.0, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
        ]);
        const F64_CHAIN: DHChain<2, f64> = DHChain::<2, f64>::new_f64([
            DHJoint { a: 1.0, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
            DHJoint { a: 1.0, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
        ]);

        let dispatch = FPDispatch::new(F32_CHAIN, F64_CHAIN);

        let q32 = SRobotQ::<2, f32>::from_array([0.5, -0.3]);
        let q64 = SRobotQ::<2, f64>::from_array([0.5, -0.3]);

        let end32 = <FPDispatch<_, _, _> as FKChain<2, f32>>::fk_end(&dispatch, &q32)
            .unwrap()
            .translation();
        let end64 = <FPDispatch<_, _, _> as FKChain<2, f64>>::fk_end(&dispatch, &q64)
            .unwrap()
            .translation();

        assert!((end32.x() as f64 - end64.x()).abs() < 1e-5);
        assert!((end32.y() as f64 - end64.y()).abs() < 1e-5);
    }

    #[test]
    fn from_f64_derives_f32_chain() {
        const F64_CHAIN: DHChain<2, f64> = DHChain::<2, f64>::new_f64([
            DHJoint { a: 1.0, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
            DHJoint { a: 1.0, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
        ]);

        let dispatch: FPDispatch<2, DHChain<2, f32>, DHChain<2, f64>> =
            FPDispatch::from_f64(F64_CHAIN);

        let q32 = SRobotQ::<2, f32>::from_array([0.5, -0.3]);
        let q64 = SRobotQ::<2, f64>::from_array([0.5, -0.3]);
        let end32 = <FPDispatch<_, _, _> as FKChain<2, f32>>::fk_end(&dispatch, &q32)
            .unwrap()
            .translation();
        let end64 = <FPDispatch<_, _, _> as FKChain<2, f64>>::fk_end(&dispatch, &q64)
            .unwrap()
            .translation();
        assert!((end32.x() as f64 - end64.x()).abs() < 1e-4);
        assert!((end32.y() as f64 - end64.y()).abs() < 1e-4);
    }
}
