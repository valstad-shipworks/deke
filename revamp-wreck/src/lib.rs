use std::marker::PhantomData;


use glam::Affine3A;
use revamp_types::{
    FKChain, RevampError, RevampResult, SRobotQ, Token, Validator,
};
use uuid::Uuid;
use wreck::{Collider, ColliderComponent, Transformable};

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
        if idx < N { self.links[idx] }
        else if idx == N { self.ee }
        else { self.base }
    }
}

#[derive(Debug, Clone)]
pub struct Attachment<const N: usize, TKN: Token> {
    pub collision: Collider,
    pub token: TKN,
    pub uuid: Uuid,
    pub filter: CollisionFilter<N>,
}

pub struct CollisionBody<const N: usize, TKN: Token> {
    pub base: Option<Collider>,
    pub filter: CollisionFilter<N>,
    pub attachments: Vec<Attachment<N, TKN>>,
    pub token: TKN,
    last_transform: Option<Affine3A>,
}

impl<const N: usize, TKN: Token> CollisionBody<N, TKN> {
    pub fn new(
        base: Option<Collider>,
        filter: CollisionFilter<N>,
        attachments: Vec<Attachment<N, TKN>>,
        token: TKN,
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
    fn sub_colliders(&self) -> impl Iterator<Item = (&Collider, &TKN, &CollisionFilter<N>)> {
        self.base
            .iter()
            .map(|c| (c, &self.token, &self.filter))
            .chain(
                self.attachments
                    .iter()
                    .map(|att| (&att.collision, &att.token, &att.filter)),
            )
    }

    fn bring_to_current(&mut self, collider: &mut Collider) {
        if let Some(tf) = self.last_transform {
            collider.transform(tf);
        }
    }

    #[inline]
    fn compute_delta(&self, new_tf: Affine3A) -> Affine3A {
        match self.last_transform {
            Some(prev) => rigid_delta(new_tf, prev),
            None => new_tf,
        }
    }

    #[inline]
    fn apply_transform(&mut self, new_tf: Affine3A) -> Affine3A {
        let tf = self.compute_delta(new_tf);
        self.apply_precomputed(tf, new_tf);
        tf
    }

    #[inline]
    fn apply_precomputed(&mut self, tf: Affine3A, new_tf: Affine3A) {
        if let Some(base) = &mut self.base {
            base.transform(tf);
        }
        for att in &mut self.attachments {
            att.collision.transform(tf);
        }
        self.last_transform = Some(new_tf);
    }
}

impl<const N: usize, TKN: Token> std::fmt::Debug for CollisionBody<N, TKN> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollisionBody")
            .field("token", &self.token)
            .field("filter", &self.filter)
            .finish()
    }
}

impl<const N: usize, TKN: Token> Clone for CollisionBody<N, TKN> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            filter: self.filter,
            attachments: self.attachments.clone(),
            token: self.token.clone(),
            last_transform: self.last_transform,
        }
    }
}

pub struct WreckValidator<const N: usize, TKN: Token, FK: FKChain<N>> {
    base: Option<CollisionBody<N, TKN>>,
    links: [CollisionBody<N, TKN>; N],
    ee: CollisionBody<N, TKN>,
    environment: Collider,
    fk: FK,
    _token: PhantomData<TKN>,
}

impl<const N: usize, TKN: Token, FK: FKChain<N>> std::fmt::Debug
    for WreckValidator<N, TKN, FK>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WreckValidator")
            .field("base", &self.base)
            .field("links", &self.links)
            .field("ee", &self.ee)
            .finish()
    }
}

impl<const N: usize, TKN: Token, FK: FKChain<N>> Clone
    for WreckValidator<N, TKN, FK>
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            links: self.links.clone(),
            ee: self.ee.clone(),
            environment: self.environment.clone(),
            fk: self.fk.clone(),
            _token: PhantomData,
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

impl<const N: usize, TKN: Token, FK: FKChain<N>> WreckValidator<N, TKN, FK> {
    pub fn new(
        links: [CollisionBody<N, TKN>; N],
        ee: CollisionBody<N, TKN>,
        base: Option<CollisionBody<N, TKN>>,
        obstacles: Collider,
        fk: FK,
    ) -> Self {
        Self {
            base,
            links,
            ee,
            environment: obstacles,
            fk,
            _token: PhantomData,
        }
    }

    pub fn links_ref(&self) -> &[CollisionBody<N, TKN>; N] {
        &self.links
    }

    pub fn base_ref(&self) -> &Option<CollisionBody<N, TKN>> {
        &self.base
    }

    pub fn into_parts(
        self,
    ) -> (
        [CollisionBody<N, TKN>; N],
        CollisionBody<N, TKN>,
        Option<CollisionBody<N, TKN>>,
        Collider,
        FK
    ) {
        (
            self.links,
            self.ee,
            self.base,
            self.environment,
            self.fk
        )
    }

    pub fn with_environment(&mut self, environment: Collider) {
        self.environment = environment;
    }

    pub fn with_environment_component(mut self, component: impl ColliderComponent) -> Self {
        self.environment.add(component);
        self
    }

    pub fn with_environment_collider(mut self, collider: Collider) -> Self {
        self.environment.include(collider);
        self
    }

    /// Adds an attachment to a link (0..N-1) or the end effector (N).
    pub fn with_attachment(mut self, index: usize, mut attachment: Attachment<N, TKN>) -> Self {
        let body = if index < N {
            &mut self.links[index]
        } else {
            &mut self.ee
        };
        body.bring_to_current(&mut attachment.collision);
        body.attachments.push(attachment);
        self
    }

    /// Removes an attachment by uuid from a link (0..N-1) or the end effector (N).
    pub fn without_attachment(mut self, index: usize, uuid: &Uuid) -> Self {
        let body = if index < N {
            &mut self.links[index]
        } else {
            &mut self.ee
        };
        if let Some(pos) = body.attachments.iter().position(|a| &a.uuid == uuid) {
            body.attachments.remove(pos);
        }
        self
    }

    #[inline]
    fn check_collisions(&mut self, q: &SRobotQ<N>) -> RevampResult<(), TKN> {
        let transforms = self.fk.fk(q);

        for i in 0..N - 1 {
            self.links[i].apply_transform(transforms[i]);
            self.check_link(i)?;
        }

        let last_tf = transforms[N - 1];
        let delta = self.links[N - 1].compute_delta(last_tf);
        self.links[N - 1].apply_precomputed(delta, last_tf);
        self.ee.apply_precomputed(delta, last_tf);
        self.check_link(N - 1)?;
        self.check_ee()?;

        Ok(())
    }

    #[inline]
    fn check_body_env(&self, body: &CollisionBody<N, TKN>) -> RevampResult<(), TKN> {
        for (collider, token, filter) in body.sub_colliders() {
            if filter.obstacles && collider.collides_other(&self.environment) {
                return Err(RevampError::EnvironmentCollision(
                    token.clone(),
                    token.clone(),
                ));
            }
        }
        Ok(())
    }

    #[inline]
    fn check_link(&self, i: usize) -> RevampResult<(), TKN> {
        self.check_body_env(&self.links[i])?;
        for j in 0..i {
            self.check_body_pair(&self.links[j], &self.links[i], j, i)?;
        }
        if let Some(base) = &self.base {
            self.check_body_pair(&self.links[i], base, i, N + 1)?;
        }
        Ok(())
    }

    #[inline]
    fn check_ee(&self) -> RevampResult<(), TKN> {
        self.check_body_env(&self.ee)?;
        for j in 0..N {
            self.check_body_pair(&self.links[j], &self.ee, j, N)?;
        }
        if let Some(base) = &self.base {
            self.check_body_pair(&self.ee, base, N, N + 1)?;
        }
        Ok(())
    }

    #[inline]
    fn check_body_pair(
        &self,
        body_a: &CollisionBody<N, TKN>,
        body_b: &CollisionBody<N, TKN>,
        a_idx: usize,
        b_idx: usize,
    ) -> RevampResult<(), TKN> {
        for (ca, ta, fa) in body_a.sub_colliders() {
            if !fa.allows(b_idx) {
                continue;
            }
            for (cb, tb, fb) in body_b.sub_colliders() {
                if fb.allows(a_idx) && ca.collides_other(cb) {
                    return Err(RevampError::SelfCollison(ta.clone(), tb.clone()));
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
            self.body_a, self.sphere_a_idx,
            self.body_b, self.sphere_b_idx,
            self.distance, self.radius_a + self.radius_b, self.overlap,
            self.center_a[0], self.center_a[1], self.center_a[2], self.radius_a,
            self.center_b[0], self.center_b[1], self.center_b[2], self.radius_b,
        )
    }
}

impl<const N: usize, TKN: Token, FK: FKChain<N>> WreckValidator<N, TKN, FK> {
    pub fn debug_self_collisions(&mut self, q: &SRobotQ<N>) -> Vec<SelfCollisionDetail> {
        let transforms = self.fk.fk(q);
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

        let check_pair = |a: &CollisionBody<N, TKN>,
                           b: &CollisionBody<N, TKN>,
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

        details
    }
}

impl<const N: usize, TKN: Token + 'static, FK: FKChain<N> + 'static>
    Validator<N, TKN>
    for WreckValidator<N, TKN, FK>
{
    fn validate<E: Into<RevampError<TKN>>, A: TryInto<SRobotQ<N>, Error = E>>(
        &mut self,
        q: A,
    ) -> RevampResult<(), TKN> {
        let q = q.try_into().map_err(|e| e.into())?;
        self.check_collisions(&q)
    }

    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> RevampResult<(), TKN> {
        for q in qs {
            self.check_collisions(q)?;
        }
        Ok(())
    }
}
