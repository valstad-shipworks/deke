use glam_traits_ext::{FloatVec, TAffine3};

use crate::SRobotQ;

use super::{AAffine3, AVec3, FKChain, FKScalar, URDFBuildError, URDFJoint, compose_fixed_joints};

/// Wraps an `FKChain` with an optional prefix (base) and/or suffix (tool) transform.
///
/// - `fk` applies only the prefix — intermediate frames stay in world coordinates
///   without the tool offset.
/// - `fk_end` and `joint_axes_positions` apply both — the end-effector includes
///   the tool tip.
#[derive(Debug, Clone)]
pub struct TransformedFK<const N: usize, F: FKScalar, FK: FKChain<N, F>> {
    inner: FK,
    prefix: Option<AAffine3<F>>,
    suffix: Option<AAffine3<F>>,
}

impl<const N: usize, F: FKScalar, FK: FKChain<N, F>> TransformedFK<N, F, FK> {
    pub const fn new(inner: FK) -> Self {
        Self {
            inner,
            prefix: None,
            suffix: None,
        }
    }

    pub fn with_prefix(mut self, prefix: AAffine3<F>) -> Self {
        self.prefix = Some(prefix);
        self
    }

    pub fn with_suffix(mut self, suffix: AAffine3<F>) -> Self {
        self.suffix = Some(suffix);
        self
    }

    /// Infallible setter for the prefix. `None` clears any previously set prefix.
    pub fn with_prefix_opt(mut self, prefix: Option<AAffine3<F>>) -> Self {
        self.prefix = prefix;
        self
    }

    /// Infallible setter for the suffix. `None` clears any previously set suffix.
    pub fn with_suffix_opt(mut self, suffix: Option<AAffine3<F>>) -> Self {
        self.suffix = suffix;
        self
    }

    pub fn set_prefix(&mut self, prefix: Option<AAffine3<F>>) {
        self.prefix = prefix;
    }

    pub fn set_suffix(&mut self, suffix: Option<AAffine3<F>>) {
        self.suffix = suffix;
    }

    pub fn prefix(&self) -> Option<&AAffine3<F>> {
        self.prefix.as_ref()
    }

    pub fn suffix(&self) -> Option<&AAffine3<F>> {
        self.suffix.as_ref()
    }

    pub fn inner(&self) -> &FK {
        &self.inner
    }
}

impl<const N: usize, FK: FKChain<N, f32>> TransformedFK<N, f32, FK> {
    /// Compose a slice of fixed URDF joints (parent→child order) and set the
    /// result as the base-side prefix transform, replacing any existing
    /// prefix. Every joint in `joints` must be `URDFJointType::Fixed`. An
    /// empty slice clears the prefix.
    pub fn with_prefix_joints(mut self, joints: &[URDFJoint]) -> Result<Self, URDFBuildError> {
        if joints.is_empty() {
            self.prefix = None;
            Ok(self)
        } else {
            self.prefix = Some(compose_fixed_joints(joints)?);
            Ok(self)
        }
    }

    /// Compose a slice of fixed URDF joints (parent→child order) and set the
    /// result as the tool-side suffix transform, replacing any existing
    /// suffix. Every joint in `joints` must be `URDFJointType::Fixed`. An
    /// empty slice clears the suffix.
    pub fn with_suffix_joints(mut self, joints: &[URDFJoint]) -> Result<Self, URDFBuildError> {
        if joints.is_empty() {
            self.suffix = None;
            Ok(self)
        } else {
            self.suffix = Some(compose_fixed_joints(joints)?);
            Ok(self)
        }
    }
}

impl<const N: usize, F: FKScalar, FK: FKChain<N, F>> FKChain<N, F> for TransformedFK<N, F, FK> {
    type Error = FK::Error;

    fn base_tf(&self) -> AAffine3<F> {
        match &self.prefix {
            Some(p) => *p * self.inner.base_tf(),
            None => self.inner.base_tf(),
        }
    }

    fn max_reach(&self) -> Result<F, Self::Error> {
        let mut reach = self.inner.max_reach()?;
        if let Some(suf) = &self.suffix {
            reach = reach + suf.translation().length();
        }
        Ok(reach)
    }

    fn fk(&self, q: &SRobotQ<N, F>) -> Result<[AAffine3<F>; N], Self::Error> {
        let mut frames = self.inner.fk(q)?;
        if let Some(pre) = &self.prefix {
            for f in &mut frames {
                *f = *pre * *f;
            }
        }
        Ok(frames)
    }

    fn fk_end(&self, q: &SRobotQ<N, F>) -> Result<AAffine3<F>, Self::Error> {
        let mut end = self.inner.fk_end(q)?;
        if let Some(pre) = &self.prefix {
            end = *pre * end;
        }
        if let Some(suf) = &self.suffix {
            end = end * *suf;
        }
        Ok(end)
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N, F>,
    ) -> Result<([AVec3<F>; N], [AVec3<F>; N], AVec3<F>), Self::Error> {
        let (mut axes, mut positions, inner_p_ee) = self.inner.joint_axes_positions(q)?;

        if let Some(pre) = &self.prefix {
            let rot = pre.matrix3();
            let t = pre.translation();
            for i in 0..N {
                axes[i] = rot * axes[i];
                positions[i] = rot * positions[i] + t;
            }
        }

        let p_ee = if self.prefix.is_some() || self.suffix.is_some() {
            self.fk_end(q)?.translation()
        } else {
            inner_p_ee
        };

        Ok((axes, positions, p_ee))
    }
}

impl<const N: usize, FK32, FK64> From<TransformedFK<N, f32, FK32>> for TransformedFK<N, f64, FK64>
where
    FK32: FKChain<N, f32>,
    FK64: FKChain<N, f64> + From<FK32>,
{
    #[inline]
    fn from(t: TransformedFK<N, f32, FK32>) -> Self {
        TransformedFK {
            inner: FK64::from(t.inner),
            prefix: t.prefix.map(|p| glam::DAffine3 {
                matrix3: p.matrix3.as_dmat3(),
                translation: p.translation.as_dvec3(),
            }),
            suffix: t.suffix.map(|p| glam::DAffine3 {
                matrix3: p.matrix3.as_dmat3(),
                translation: p.translation.as_dvec3(),
            }),
        }
    }
}

impl<const N: usize, FK64, FK32> From<TransformedFK<N, f64, FK64>> for TransformedFK<N, f32, FK32>
where
    FK64: FKChain<N, f64>,
    FK32: FKChain<N, f32> + From<FK64>,
{
    #[inline]
    fn from(t: TransformedFK<N, f64, FK64>) -> Self {
        TransformedFK {
            inner: FK32::from(t.inner),
            prefix: t.prefix.map(|p| glam::Affine3A {
                matrix3: glam::Mat3A::from(p.matrix3.as_mat3()),
                translation: p.translation.as_vec3a(),
            }),
            suffix: t.suffix.map(|p| glam::Affine3A {
                matrix3: glam::Mat3A::from(p.matrix3.as_mat3()),
                translation: p.translation.as_vec3a(),
            }),
        }
    }
}
