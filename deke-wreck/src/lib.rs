mod dynamic;

use std::cell::RefCell;

pub use dynamic::DynamicWreckValidator;

use deke_types::{
    DekeError, DekeResult, FKChain, SRobotQ, SRobotQLike, Validator, ValidatorContext,
};
use glam::Affine3A;
use uuid::Uuid;
use wreck::{Collider, Transformable};

#[derive(Default)]
struct BodyScratch {
    base: Option<Collider>,
    attachments: Vec<Collider>,
}

impl BodyScratch {
    /// Refreshes `self` from `body`'s pristine colliders, baking in the
    /// absolute world-frame transform `tf`. Existing allocations are reused
    /// via `Collider::clone_from`.
    fn refresh<const N: usize>(&mut self, body: &CollisionBody<N>, tf: Affine3A) {
        match (&mut self.base, &body.base) {
            (Some(dst), Some(src)) => {
                dst.clone_from(src);
                dst.transform(tf);
            }
            (slot @ None, Some(src)) => {
                let mut c = Collider::default();
                c.clone_from(src);
                c.transform(tf);
                *slot = Some(c);
            }
            (slot @ Some(_), None) => *slot = None,
            (None, None) => {}
        }
        self.attachments
            .resize_with(body.attachments.len(), Collider::default);
        for (dst, src) in self.attachments.iter_mut().zip(body.attachments.iter()) {
            dst.clone_from(&src.collision);
            dst.transform(tf);
        }
    }
}

thread_local! {
    static SCRATCH_BODIES: RefCell<Vec<BodyScratch>> = const { RefCell::new(Vec::new()) };
}

/// Iterator over a body's sub-colliders, yielding either the scratch (moved)
/// copy or the pristine body-frame collider depending on whether a scratch is
/// provided.
struct SubColliderIter<'a, const N: usize> {
    body: &'a CollisionBody<N>,
    scratch: Option<&'a BodyScratch>,
    idx: usize,
}

impl<'a, const N: usize> Iterator for SubColliderIter<'a, N> {
    type Item = (&'a Collider, i16, &'a CollisionFilter<N>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == 0 {
            self.idx = 1;
            let collider = match self.scratch {
                Some(s) => s.base.as_ref(),
                None => self.body.base.as_ref(),
            };
            if let Some(c) = collider {
                return Some((c, self.body.token, &self.body.filter));
            }
        }
        let att_idx = self.idx - 1;
        if att_idx >= self.body.attachments.len() {
            return None;
        }
        self.idx += 1;
        let att = &self.body.attachments[att_idx];
        let collider = match self.scratch {
            Some(s) => &s.attachments[att_idx],
            None => &att.collision,
        };
        Some((collider, att.token, &att.filter))
    }
}

fn sub_colliders<'a, const N: usize>(
    body: &'a CollisionBody<N>,
    scratch: Option<&'a BodyScratch>,
) -> SubColliderIter<'a, N> {
    SubColliderIter {
        body,
        scratch,
        idx: 0,
    }
}

pub struct WreckValidatorContext<'a, const N: usize> {
    pub extra_attachments: &'a [&'a Attachment<N>],
    pub self_collisions: bool,
    pub environment: &'a Collider,
}

impl<'a, const N: usize> WreckValidatorContext<'a, N> {
    pub fn new(environment: &'a Collider) -> Self {
        Self {
            extra_attachments: &[],
            self_collisions: true,
            environment,
        }
    }

    pub fn with_extras(mut self, extras: &'a [&'a Attachment<N>]) -> Self {
        self.extra_attachments = extras;
        self
    }

    pub fn with_self_collisions(mut self, enabled: bool) -> Self {
        self.self_collisions = enabled;
        self
    }
}

impl<'a, const N: usize> ValidatorContext for WreckValidatorContext<'a, N> {}

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
        }
    }

    /// Iterates sub-colliders in the body's pristine (body-local) frame.
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
        }
    }
}

pub struct WreckValidator<const N: usize, FK: FKChain<N>> {
    base: Option<CollisionBody<N>>,
    links: [CollisionBody<N>; N],
    ee: CollisionBody<N>,
    fk: FK,
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
            fk: self.fk.clone(),
        }
    }
}

impl<const N: usize, FK: FKChain<N>> WreckValidator<N, FK> {
    pub fn new(
        links: [CollisionBody<N>; N],
        ee: CollisionBody<N>,
        base: Option<CollisionBody<N>>,
        fk: FK,
    ) -> Self {
        Self {
            base,
            links,
            ee,
            fk,
        }
    }

    pub fn links_ref(&self) -> &[CollisionBody<N>; N] {
        &self.links
    }

    pub fn base_ref(&self) -> &Option<CollisionBody<N>> {
        &self.base
    }

    pub fn ee_ref(&self) -> &CollisionBody<N> {
        &self.ee
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        [CollisionBody<N>; N],
        CollisionBody<N>,
        Option<CollisionBody<N>>,
        FK,
    ) {
        (self.links, self.ee, self.base, self.fk)
    }

    fn check_collisions(
        &self,
        q: &SRobotQ<N>,
        ctx: &WreckValidatorContext<'_, N>,
    ) -> DekeResult<()> {
        let transforms = self.fk.fk(q).map_err(Into::into)?;

        SCRATCH_BODIES.with_borrow_mut(|scratch| {
            if scratch.len() < N + 1 {
                scratch.resize_with(N + 1, BodyScratch::default);
            }

            for i in 0..N {
                scratch[i].refresh(&self.links[i], transforms[i]);
            }
            scratch[N].refresh(&self.ee, transforms[N - 1]);

            for i in 0..N {
                check_body_env(&self.links[i], Some(&scratch[i]), ctx)?;
            }
            check_body_env(&self.ee, Some(&scratch[N]), ctx)?;

            if ctx.self_collisions {
                for i in 0..N {
                    for j in 0..i {
                        check_body_pair(
                            &self.links[j],
                            Some(&scratch[j]),
                            &self.links[i],
                            Some(&scratch[i]),
                            j,
                            i,
                        )?;
                    }
                    if let Some(base) = &self.base {
                        check_body_pair(
                            &self.links[i],
                            Some(&scratch[i]),
                            base,
                            None,
                            i,
                            N + 1,
                        )?;
                    }
                }
                for j in 0..N {
                    check_body_pair(
                        &self.links[j],
                        Some(&scratch[j]),
                        &self.ee,
                        Some(&scratch[N]),
                        j,
                        N,
                    )?;
                }
                if let Some(base) = &self.base {
                    check_body_pair(&self.ee, Some(&scratch[N]), base, None, N, N + 1)?;
                }
            }

            Ok(())
        })
    }
}

#[inline]
fn check_body_env<const N: usize>(
    body: &CollisionBody<N>,
    scratch: Option<&BodyScratch>,
    ctx: &WreckValidatorContext<'_, N>,
) -> DekeResult<()> {
    for (collider, token, filter) in sub_colliders(body, scratch) {
        if filter.obstacles {
            if collider.collides_other(ctx.environment) {
                return Err(DekeError::EnvironmentCollision(token, token));
            }
            for extra in ctx.extra_attachments {
                if collider.collides_other(&extra.collision) {
                    return Err(DekeError::EnvironmentCollision(token, extra.token));
                }
            }
        }
    }
    Ok(())
}

#[inline]
fn check_body_pair<const N: usize>(
    body_a: &CollisionBody<N>,
    scratch_a: Option<&BodyScratch>,
    body_b: &CollisionBody<N>,
    scratch_b: Option<&BodyScratch>,
    a_idx: usize,
    b_idx: usize,
) -> DekeResult<()> {
    for (ca, ta, fa) in sub_colliders(body_a, scratch_a) {
        if !fa.allows(b_idx) {
            continue;
        }
        for (cb, tb, fb) in sub_colliders(body_b, scratch_b) {
            if fb.allows(a_idx) && ca.collides_other(cb) {
                return Err(DekeError::SelfCollision(ta, tb));
            }
        }
    }
    Ok(())
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
        &self,
        q: &SRobotQ<N>,
    ) -> DekeResult<Vec<SelfCollisionDetail>> {
        let transforms = self.fk.fk(q).map_err(Into::into)?;

        SCRATCH_BODIES.with_borrow_mut(|scratch| {
            if scratch.len() < N + 1 {
                scratch.resize_with(N + 1, BodyScratch::default);
            }
            for i in 0..N {
                scratch[i].refresh(&self.links[i], transforms[i]);
            }
            scratch[N].refresh(&self.ee, transforms[N - 1]);

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
                                  sa_scratch: Option<&BodyScratch>,
                                  b: &CollisionBody<N>,
                                  sb_scratch: Option<&BodyScratch>,
                                  a_idx: usize,
                                  b_idx: usize,
                                  details: &mut Vec<SelfCollisionDetail>| {
                for (ca, _ta, fa) in sub_colliders(a, sa_scratch) {
                    if !fa.allows(b_idx) {
                        continue;
                    }
                    for (cb, _tb, fb) in sub_colliders(b, sb_scratch) {
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
                    check_pair(
                        &self.links[j],
                        Some(&scratch[j]),
                        &self.links[i],
                        Some(&scratch[i]),
                        j,
                        i,
                        &mut details,
                    );
                }
                if let Some(base) = &self.base {
                    check_pair(
                        &self.links[i],
                        Some(&scratch[i]),
                        base,
                        None,
                        i,
                        N + 1,
                        &mut details,
                    );
                }
            }

            for j in 0..N {
                check_pair(
                    &self.links[j],
                    Some(&scratch[j]),
                    &self.ee,
                    Some(&scratch[N]),
                    j,
                    N,
                    &mut details,
                );
            }
            if let Some(base) = &self.base {
                check_pair(
                    &self.ee,
                    Some(&scratch[N]),
                    base,
                    None,
                    N,
                    N + 1,
                    &mut details,
                );
            }

            Ok(details)
        })
    }

    /// Evaluates FK at `q` and returns a single `Collider` containing every
    /// sub-collider of the validator (base, each link, the end effector, and
    /// all of their attachments) transformed into world frame.
    pub fn debug_bodies(&self, q: &SRobotQ<N>) -> DekeResult<Collider> {
        let transforms = self.fk.fk(q).map_err(Into::into)?;
        let mut out = Collider::default();

        let mut push_body = |body: &CollisionBody<N>, tf: Option<Affine3A>| {
            if let Some(base) = &body.base {
                let mut c = base.clone();
                if let Some(tf) = tf {
                    c.transform(tf);
                }
                out.include(c);
            }
            for att in &body.attachments {
                let mut c = att.collision.clone();
                if let Some(tf) = tf {
                    c.transform(tf);
                }
                out.include(c);
            }
        };

        if let Some(base) = &self.base {
            push_body(base, None);
        }
        for i in 0..N {
            push_body(&self.links[i], Some(transforms[i]));
        }
        push_body(&self.ee, Some(transforms[N - 1]));

        Ok(out)
    }
}

impl<const N: usize, FK: FKChain<N> + 'static> Validator<N> for WreckValidator<N, FK> {
    type Context<'ctx> = WreckValidatorContext<'ctx, N>;

    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<N, E>>(
        &self,
        q: A,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        let q = q.to_srobotq().map_err(Into::into)?;
        self.check_collisions(&q, ctx)
    }

    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        for q in qs {
            self.check_collisions(q, ctx)?;
        }
        Ok(())
    }
}
