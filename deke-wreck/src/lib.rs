mod dynamic;
use std::sync::Arc;

pub use dynamic::DynamicWreckValidator;

use deke_types::{DekeError, DekeResult, FKChain, SRobotQ, Validator};
use glam::Affine3A;
use uuid::Uuid;
use wreck::{Collider, Transformable};

#[derive(Debug, Clone, Copy)]
pub struct CollisionFilter<const N: usize> {
    pub links: [bool; N],
    pub ee: bool,
    pub base: bool,
    pub obstacles: bool,
}

impl<const N: usize> CollisionFilter<N> {
    #[inline]
    pub fn allows(&self, idx: usize) -> bool {
        if idx < N {
            self.links[idx]
        } else if idx == N {
            self.ee
        } else {
            self.base
        }
    }
}

#[derive(Debug, Clone)]
pub struct Attachment<const N: usize> {
    pub collision: Collider,
    pub token: i16,
    pub uuid: Uuid,
    pub filter: CollisionFilter<N>,
}

pub struct CollisionBody<const N: usize> {
    pub base: Option<Collider>,
    pub filter: CollisionFilter<N>,
    pub attachments: Vec<Attachment<N>>,
    pub token: i16,
    last_transform: Option<Affine3A>,
}

impl<const N: usize> CollisionBody<N> {
    pub fn new(
        base: Option<Collider>,
        filter: CollisionFilter<N>,
        attachments: Vec<Attachment<N>>,
        token: i16,
    ) -> Self {
        Self {
            base,
            filter,
            attachments,
            token,
            last_transform: None,
        }
    }

    #[inline]
    pub fn sub_colliders(&self) -> impl Iterator<Item = (&Collider, &i16, &CollisionFilter<N>)> {
        self.base
            .iter()
            .map(|c| (c, &self.token, &self.filter))
            .chain(
                self.attachments
                    .iter()
                    .map(|att| (&att.collision, &att.token, &att.filter)),
            )
    }

    pub fn bring_to_current(&mut self, collider: &mut Collider) {
        if let Some(tf) = self.last_transform {
            collider.transform(tf);
        }
    }

    #[inline]
    pub fn compute_delta(&self, new_tf: Affine3A) -> Affine3A {
        match self.last_transform {
            Some(prev) => rigid_delta(new_tf, prev),
            None => new_tf,
        }
    }

    #[inline]
    pub fn apply_transform(&mut self, new_tf: Affine3A) -> Affine3A {
        let tf = self.compute_delta(new_tf);
        self.apply_precomputed(tf, new_tf);
        tf
    }

    pub fn apply_precomputed(&mut self, tf: Affine3A, new_tf: Affine3A) {
        if let Some(base) = &mut self.base {
            base.transform(tf);
        }
        for att in &mut self.attachments {
            att.collision.transform(tf);
        }
        self.last_transform = Some(new_tf);
    }
}

impl<const N: usize> std::fmt::Debug for CollisionBody<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollisionBody")
            .field("token", &self.token)
            .field("filter", &self.filter)
            .finish()
    }
}

impl<const N: usize> Clone for CollisionBody<N> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            filter: self.filter,
            attachments: self.attachments.clone(),
            token: self.token,
            last_transform: self.last_transform,
        }
    }
}

pub struct WreckValidator<const N: usize, FK: FKChain<N>> {
    base: Option<CollisionBody<N>>,
    links: [CollisionBody<N>; N],
    ee: CollisionBody<N>,
    environment: Arc<Collider>,
    fk: FK,
    self_collisions: bool,
}

impl<const N: usize, FK: FKChain<N>> std::fmt::Debug for WreckValidator<N, FK> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WreckValidator")
            .field("base", &self.base)
            .field("links", &self.links)
            .field("ee", &self.ee)
            .finish()
    }
}

impl<const N: usize, FK: FKChain<N>> Clone for WreckValidator<N, FK> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            links: self.links.clone(),
            ee: self.ee.clone(),
            environment: self.environment.clone(),
            fk: self.fk.clone(),
            self_collisions: self.self_collisions,
        }
    }
}

/// Computes the rigid-body delta transform `new_tf * prev_tf.inverse()` as
/// `Affine3`, exploiting the fact that FK produces pure rotation+translation
/// (orthogonal matrix → inverse is transpose).
#[inline]
fn rigid_delta(new_tf: Affine3A, prev_tf: Affine3A) -> Affine3A {
    let prev_rt = prev_tf.matrix3.transpose();
    let delta_m = new_tf.matrix3 * prev_rt;
    let delta_t = new_tf.translation - delta_m * prev_tf.translation;
    Affine3A {
        matrix3: delta_m,
        translation: delta_t,
    }
}

impl<const N: usize, FK: FKChain<N>> WreckValidator<N, FK> {
    pub fn new(
        links: [CollisionBody<N>; N],
        ee: CollisionBody<N>,
        base: Option<CollisionBody<N>>,
        environment: Arc<Collider>,
        fk: FK,
    ) -> Self {
        Self {
            base,
            links,
            ee,
            environment,
            fk,
            self_collisions: true,
        }
    }

    pub fn set_self_collisions(&mut self, enabled: bool) {
        self.self_collisions = enabled;
    }

    pub fn self_collisions(&self) -> bool {
        self.self_collisions
    }

    pub fn links_ref(&self) -> &[CollisionBody<N>; N] {
        &self.links
    }

    pub fn base_ref(&self) -> &Option<CollisionBody<N>> {
        &self.base
    }

    pub fn into_parts(
        self,
    ) -> (
        [CollisionBody<N>; N],
        CollisionBody<N>,
        Option<CollisionBody<N>>,
        Arc<Collider>,
        FK,
    ) {
        (self.links, self.ee, self.base, self.environment, self.fk)
    }

    pub fn with_environment(&mut self, environment: Arc<Collider>) {
        self.environment = environment;
    }

    pub fn environment(&self) -> &Arc<Collider> {
        &self.environment
    }

    /// Adds an attachment to a link (0..N-1) or the end effector (N).
    pub fn add_attachment(&mut self, index: usize, mut attachment: Attachment<N>) {
        let body = if index < N {
            &mut self.links[index]
        } else {
            &mut self.ee
        };
        body.bring_to_current(&mut attachment.collision);
        body.attachments.push(attachment);
    }

    /// Removes an attachment by uuid from a link (0..N-1) or the end effector (N).
    pub fn remove_attachment(&mut self, index: usize, uuid: &Uuid) {
        let body = if index < N {
            &mut self.links[index]
        } else {
            &mut self.ee
        };
        if let Some(pos) = body.attachments.iter().position(|a| &a.uuid == uuid) {
            body.attachments.remove(pos);
        }
    }

    fn check_collisions(&mut self, q: &SRobotQ<N>) -> DekeResult<()> {
        let transforms = self.fk.fk(q).map_err(Into::into)?;

        for i in 0..N - 1 {
            self.links[i].apply_transform(transforms[i]);
            self.check_link(i)?;
        }

        let last_tf = transforms[N - 1];
        let delta = self.links[N - 1].compute_delta(last_tf);
        self.links[N - 1].apply_precomputed(delta, last_tf);
        self.check_link(N - 1)?;
        self.ee.apply_precomputed(delta, last_tf);
        self.check_ee()?;

        Ok(())
    }

    #[inline]
    fn check_body_env(&self, body: &CollisionBody<N>) -> DekeResult<()> {
        for (collider, token, filter) in body.sub_colliders() {
            if filter.obstacles && collider.collides_other(&self.environment) {
                return Err(DekeError::EnvironmentCollision(*token, *token));
            }
        }
        Ok(())
    }

    fn check_link(&self, i: usize) -> DekeResult<()> {
        self.check_body_env(&self.links[i])?;
        if self.self_collisions {
            for j in 0..i {
                self.check_body_pair(&self.links[j], &self.links[i], j, i)?;
            }
            if let Some(base) = &self.base {
                self.check_body_pair(&self.links[i], base, i, N + 1)?;
            }
        }
        Ok(())
    }

    fn check_ee(&self) -> DekeResult<()> {
        self.check_body_env(&self.ee)?;
        if self.self_collisions {
            for j in 0..N {
                self.check_body_pair(&self.links[j], &self.ee, j, N)?;
            }
            if let Some(base) = &self.base {
                self.check_body_pair(&self.ee, base, N, N + 1)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn check_body_pair(
        &self,
        body_a: &CollisionBody<N>,
        body_b: &CollisionBody<N>,
        a_idx: usize,
        b_idx: usize,
    ) -> DekeResult<()> {
        for (ca, ta, fa) in body_a.sub_colliders() {
            if !fa.allows(b_idx) {
                continue;
            }
            for (cb, tb, fb) in body_b.sub_colliders() {
                if fb.allows(a_idx) && ca.collides_other(cb) {
                    return Err(DekeError::SelfCollision(*ta, *tb));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SelfCollisionDetail {
    pub body_a: String,
    pub body_b: String,
    pub sphere_a_idx: usize,
    pub sphere_b_idx: usize,
    pub center_a: [f32; 3],
    pub center_b: [f32; 3],
    pub radius_a: f32,
    pub radius_b: f32,
    pub distance: f32,
    pub overlap: f32,
}

impl std::fmt::Display for SelfCollisionDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}[{}] vs {}[{}]: dist={:.6} r_sum={:.6} overlap={:.6}\n  a=({:.6}, {:.6}, {:.6}) r={:.6}\n  b=({:.6}, {:.6}, {:.6}) r={:.6}",
            self.body_a,
            self.sphere_a_idx,
            self.body_b,
            self.sphere_b_idx,
            self.distance,
            self.radius_a + self.radius_b,
            self.overlap,
            self.center_a[0],
            self.center_a[1],
            self.center_a[2],
            self.radius_a,
            self.center_b[0],
            self.center_b[1],
            self.center_b[2],
            self.radius_b,
        )
    }
}

impl<const N: usize, FK: FKChain<N>> WreckValidator<N, FK> {
    pub fn debug_self_collisions(
        &mut self,
        q: &SRobotQ<N>,
    ) -> DekeResult<Vec<SelfCollisionDetail>> {
        let transforms = self.fk.fk(q).map_err(Into::into)?;
        for i in 0..N - 1 {
            self.links[i].apply_transform(transforms[i]);
        }
        let last_tf = transforms[N - 1];
        let delta = self.links[N - 1].compute_delta(last_tf);
        self.links[N - 1].apply_precomputed(delta, last_tf);
        self.ee.apply_precomputed(delta, last_tf);

        let mut details = Vec::new();

        let body_name = |idx: usize| -> String {
            if idx < N {
                format!("link_{}", idx)
            } else if idx == N {
                "ee".to_string()
            } else {
                "base".to_string()
            }
        };

        let check_pair = |a: &CollisionBody<N>,
                          b: &CollisionBody<N>,
                          a_idx: usize,
                          b_idx: usize,
                          details: &mut Vec<SelfCollisionDetail>| {
            for (ca, _ta, fa) in a.sub_colliders() {
                if !fa.allows(b_idx) {
                    continue;
                }
                for (cb, _tb, fb) in b.sub_colliders() {
                    if !fb.allows(a_idx) {
                        continue;
                    }
                    for (si, sa) in ca.spheres().iter().enumerate() {
                        for (sj, sb) in cb.spheres().iter().enumerate() {
                            let d = sa.center.distance(sb.center);
                            let r_sum = sa.radius + sb.radius;
                            if d < r_sum {
                                details.push(SelfCollisionDetail {
                                    body_a: body_name(a_idx),
                                    body_b: body_name(b_idx),
                                    sphere_a_idx: si,
                                    sphere_b_idx: sj,
                                    center_a: sa.center.into(),
                                    center_b: sb.center.into(),
                                    radius_a: sa.radius,
                                    radius_b: sb.radius,
                                    distance: d,
                                    overlap: r_sum - d,
                                });
                            }
                        }
                    }
                }
            }
        };

        for i in 0..N {
            for j in 0..i {
                check_pair(&self.links[j], &self.links[i], j, i, &mut details);
            }
            if let Some(base) = &self.base {
                check_pair(&self.links[i], base, i, N + 1, &mut details);
            }
        }

        for j in 0..N {
            check_pair(&self.links[j], &self.ee, j, N, &mut details);
        }
        if let Some(base) = &self.base {
            check_pair(&self.ee, base, N, N + 1, &mut details);
        }

        Ok(details)
    }
}

impl<const N: usize, FK: FKChain<N> + 'static> Validator<N> for WreckValidator<N, FK> {
    fn validate<E: Into<DekeError>, A: TryInto<SRobotQ<N>, Error = E>>(
        &mut self,
        q: A,
    ) -> DekeResult<()> {
        let q = q.try_into().map_err(|e| e.into())?;
        self.check_collisions(&q)
    }

    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> DekeResult<()> {
        for q in qs {
            self.check_collisions(q)?;
        }
        Ok(())
    }
}
