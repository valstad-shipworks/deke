use glam::{Affine3A, Vec3A};

use crate::SRobotQ;

use super::{FKChain, URDFBuildError, URDFJoint, compose_fixed_joints};

/// Wraps an `FKChain` with an optional prefix (base) and/or suffix (tool) transform.
///
/// - `fk` applies only the prefix — intermediate frames stay in world coordinates
///   without the tool offset.
/// - `fk_end` and `joint_axes_positions` apply both — the end-effector includes
///   the tool tip.
#[derive(Debug, Clone)]
pub struct TransformedFK<const N: usize, FK: FKChain<N>> {
    inner: FK,
    prefix: Option<Affine3A>,
    suffix: Option<Affine3A>,
}

impl<const N: usize, FK: FKChain<N>> TransformedFK<N, FK> {
    pub const fn new(inner: FK) -> Self {
        Self {
            inner,
            prefix: None,
            suffix: None,
        }
    }

    pub const fn with_prefix(mut self, prefix: Affine3A) -> Self {
        self.prefix = Some(prefix);
        self
    }

    pub const fn with_suffix(mut self, suffix: Affine3A) -> Self {
        self.suffix = Some(suffix);
        self
    }

    /// Infallible `const`-usable setter for the prefix. `None` clears any
    /// previously set prefix. Pair with [`compose_fixed_joints`] (const) to
    /// build the prefix from a slice of `Fixed` joints in a `const` context.
    pub const fn with_prefix_opt(mut self, prefix: Option<Affine3A>) -> Self {
        self.prefix = prefix;
        self
    }

    /// Infallible `const`-usable setter for the suffix. `None` clears any
    /// previously set suffix. Pair with [`compose_fixed_joints`] (const) to
    /// build the suffix from a slice of `Fixed` joints in a `const` context.
    pub const fn with_suffix_opt(mut self, suffix: Option<Affine3A>) -> Self {
        self.suffix = suffix;
        self
    }

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

    pub fn set_prefix(&mut self, prefix: Option<Affine3A>) {
        self.prefix = prefix;
    }

    pub fn set_suffix(&mut self, suffix: Option<Affine3A>) {
        self.suffix = suffix;
    }

    pub fn prefix(&self) -> Option<&Affine3A> {
        self.prefix.as_ref()
    }

    pub fn suffix(&self) -> Option<&Affine3A> {
        self.suffix.as_ref()
    }

    pub fn inner(&self) -> &FK {
        &self.inner
    }
}

impl<const N: usize, FK: FKChain<N>> FKChain<N> for TransformedFK<N, FK> {
    type Error = FK::Error;

    fn base_tf(&self) -> Affine3A {
        match &self.prefix {
            Some(p) => *p * self.inner.base_tf(),
            None => self.inner.base_tf(),
        }
    }

    fn max_reach(&self) -> Result<f32, Self::Error> {
        let mut reach = self.inner.max_reach()?;
        if let Some(suf) = &self.suffix {
            reach += Vec3A::from(suf.translation).length();
        }
        Ok(reach)
    }

    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], Self::Error> {
        let mut frames = self.inner.fk(q)?;
        if let Some(pre) = &self.prefix {
            for f in &mut frames {
                *f = *pre * *f;
            }
        }
        Ok(frames)
    }

    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, Self::Error> {
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
        q: &SRobotQ<N>,
    ) -> Result<([Vec3A; N], [Vec3A; N], Vec3A), Self::Error> {
        let (mut axes, mut positions, inner_p_ee) = self.inner.joint_axes_positions(q)?;

        if let Some(pre) = &self.prefix {
            let rot = pre.matrix3;
            let t = Vec3A::from(pre.translation);
            for i in 0..N {
                axes[i] = rot * axes[i];
                positions[i] = rot * positions[i] + t;
            }
        }

        let p_ee = if self.prefix.is_some() || self.suffix.is_some() {
            self.fk_end(q)?.translation
        } else {
            inner_p_ee
        };

        Ok((axes, positions, p_ee))
    }
}
