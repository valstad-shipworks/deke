mod dynamic;

#[cfg(feature = "valuable")]
mod valuable_impls;

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
    static SCRATCH_EXTRAS: RefCell<Vec<Collider>> = const { RefCell::new(Vec::new()) };
}

/// World-frame transform of the body identified by `mounted_on` using the
/// same index convention as [`CollisionFilter::allows`]: `0..N` link, `N`
/// EE, `N + 1` base. Returns `None` for any other index (e.g. the static
/// world body), in which case the collider is taken as already in the
/// world frame.
///
/// The EE uses `ee_tf` (from [`FKChain::fk_end`]) rather than
/// `transforms[N - 1]`: chains with a tool/suffix offset (e.g. trailing
/// fixed joints in a URDF, [`deke_types::TransformedFK`] with a suffix)
/// place the EE at a frame distinct from the last joint.
#[inline]
fn mounted_tf<const N: usize>(
    mounted_on: usize,
    transforms: &[Affine3A; N],
    ee_tf: Affine3A,
    base_tf: Affine3A,
) -> Option<Affine3A> {
    if mounted_on < N {
        Some(transforms[mounted_on])
    } else if mounted_on == N {
        Some(ee_tf)
    } else if mounted_on == N + 1 {
        Some(base_tf)
    } else {
        None
    }
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

/// Body-index convention used by [`CollisionFilter::allows`] and
/// [`Attachment::mounted_on`]:
///
/// - `0..N` → link `i`
/// - `N` → end effector
/// - `N + 1` → base
///
/// Helper constants for readability.
pub const fn ee_idx<const N: usize>() -> usize {
    N
}
pub const fn base_idx<const N: usize>() -> usize {
    N + 1
}

#[derive(Debug, Clone)]
pub struct Attachment<const N: usize> {
    pub collision: Collider,
    pub token: i16,
    pub uuid: Uuid,
    pub filter: CollisionFilter<N>,
    /// Optional owning body index. When this `Attachment` is used as an
    /// *extra* (`WreckValidatorContext::extra_attachments`), the validator
    /// (1) skips the body whose index equals `mounted_on` so the extra
    /// can't collide with the body it's physically bolted to (e.g. a
    /// payload mounted on the EE should not report collision against the
    /// EE or its own gripper), and (2) interprets [`Attachment::collision`]
    /// as living in that body's local frame and applies the body's
    /// world-frame FK transform before checking collisions. When
    /// `mounted_on` is `None`, the collider is taken as already in the
    /// world frame. Has no effect for attachments installed directly on a
    /// [`CollisionBody`] (that body already skips its own sub-colliders by
    /// construction and transforms its attachments via FK).
    ///
    /// Index convention matches [`CollisionFilter::allows`]: `0..N` links,
    /// `N` EE, `N + 1` base. Use [`ee_idx`] / [`base_idx`] for clarity.
    pub mounted_on: Option<usize>,
}

impl<const N: usize> Attachment<N> {
    pub fn new(collision: Collider, token: i16, uuid: Uuid, filter: CollisionFilter<N>) -> Self {
        Self {
            collision,
            token,
            uuid,
            filter,
            mounted_on: None,
        }
    }

    pub fn with_mounted_on(mut self, body_idx: usize) -> Self {
        self.mounted_on = Some(body_idx);
        self
    }
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
    /// Static collision geometry that is *never* transformed — always in the
    /// world frame (e.g. fixtures, tool stands, safety fences that aren't
    /// part of the environment `Collider`). Participates only in
    /// self-collision pair checks with the moving parts.
    world: Option<CollisionBody<N>>,
    links: [CollisionBody<N>; N],
    ee: CollisionBody<N>,
    fk: FK,
}

impl<const N: usize, FK: FKChain<N>> std::fmt::Debug for WreckValidator<N, FK> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WreckValidator")
            .field("base", &self.base)
            .field("world", &self.world)
            .field("links", &self.links)
            .field("ee", &self.ee)
            .finish()
    }
}

impl<const N: usize, FK: FKChain<N>> Clone for WreckValidator<N, FK> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            world: self.world.clone(),
            links: self.links.clone(),
            ee: self.ee.clone(),
            fk: self.fk.clone(),
        }
    }
}

impl<const N: usize, FK: FKChain<N>> WreckValidator<N, FK> {
    /// Build a validator.
    ///
    /// `world` is a static body in world coordinates that is never
    /// transformed (e.g. fences, stands, fixtures). `base` is placed at
    /// [`FKChain::base_tf`].
    pub fn new(
        links: [CollisionBody<N>; N],
        ee: CollisionBody<N>,
        base: Option<CollisionBody<N>>,
        world: Option<CollisionBody<N>>,
        fk: FK,
    ) -> Self {
        Self {
            base,
            world,
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

    pub fn world_ref(&self) -> &Option<CollisionBody<N>> {
        &self.world
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
        Option<CollisionBody<N>>,
        FK,
    ) {
        (self.links, self.ee, self.base, self.world, self.fk)
    }

    fn check_collisions(
        &self,
        q: &SRobotQ<N>,
        ctx: &WreckValidatorContext<'_, N>,
    ) -> DekeResult<()> {
        let (base_tf, transforms, ee_tf) = self.fk.all_fk(q).map_err(Into::into)?;

        SCRATCH_BODIES.with_borrow_mut(|scratch| {
            SCRATCH_EXTRAS.with_borrow_mut(|extras_scratch| {
            if scratch.len() < N + 2 {
                scratch.resize_with(N + 2, BodyScratch::default);
            }

            for i in 0..N {
                scratch[i].refresh(&self.links[i], transforms[i]);
            }
            scratch[N].refresh(&self.ee, ee_tf);
            if let Some(base) = &self.base {
                scratch[N + 1].refresh(base, base_tf);
            }

            extras_scratch.resize_with(ctx.extra_attachments.len(), Collider::default);
            for (dst, src) in extras_scratch.iter_mut().zip(ctx.extra_attachments.iter()) {
                dst.clone_from(&src.collision);
                if let Some(idx) = src.mounted_on
                    && let Some(tf) = mounted_tf::<N>(idx, &transforms, ee_tf, base_tf) {
                        dst.transform(tf);
                    }
            }

            #[allow(clippy::needless_range_loop)]
            for i in 0..N {
                check_body_env(&self.links[i], Some(&scratch[i]), i, ctx, extras_scratch)?;
            }
            check_body_env(&self.ee, Some(&scratch[N]), N, ctx, extras_scratch)?;

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
                            Some(&scratch[N + 1]),
                            i,
                            N + 1,
                        )?;
                    }
                    if let Some(world) = &self.world {
                        check_body_pair(
                            &self.links[i],
                            Some(&scratch[i]),
                            world,
                            None,
                            i,
                            N + 2,
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
                    check_body_pair(
                        &self.ee,
                        Some(&scratch[N]),
                        base,
                        Some(&scratch[N + 1]),
                        N,
                        N + 1,
                    )?;
                }
                if let Some(world) = &self.world {
                    check_body_pair(&self.ee, Some(&scratch[N]), world, None, N, N + 2)?;
                }
                if let (Some(base), Some(world)) = (&self.base, &self.world) {
                    check_body_pair(
                        base,
                        Some(&scratch[N + 1]),
                        world,
                        None,
                        N + 1,
                        N + 2,
                    )?;
                }
            }

            Ok(())
            })
        })
    }
}

#[inline]
fn check_body_env<const N: usize>(
    body: &CollisionBody<N>,
    scratch: Option<&BodyScratch>,
    body_idx: usize,
    ctx: &WreckValidatorContext<'_, N>,
    extras_world: &[Collider],
) -> DekeResult<()> {
    for (collider, token, filter) in sub_colliders(body, scratch) {
        if !filter.obstacles {
            continue;
        }
        if collider.collides_other(ctx.environment) {
            return Err(DekeError::EnvironmentCollision(token, token));
        }
        for (extra, extra_world) in ctx.extra_attachments.iter().zip(extras_world.iter()) {
            if matches!(extra.mounted_on, Some(idx) if idx == body_idx) {
                continue;
            }
            if !extra.filter.allows(body_idx) {
                continue;
            }
            if collider.collides_other(extra_world) {
                return Err(DekeError::EnvironmentCollision(token, extra.token));
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
        let (base_tf, transforms, ee_tf) = self.fk.all_fk(q).map_err(Into::into)?;

        SCRATCH_BODIES.with_borrow_mut(|scratch| {
            if scratch.len() < N + 2 {
                scratch.resize_with(N + 2, BodyScratch::default);
            }
            for i in 0..N {
                scratch[i].refresh(&self.links[i], transforms[i]);
            }
            scratch[N].refresh(&self.ee, ee_tf);
            if let Some(base) = &self.base {
                scratch[N + 1].refresh(base, base_tf);
            }

            let mut details = Vec::new();

            let body_name = |idx: usize| -> String {
                if idx < N {
                    format!("link_{}", idx)
                } else if idx == N {
                    "ee".to_string()
                } else if idx == N + 1 {
                    "base".to_string()
                } else {
                    "world".to_string()
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
                        Some(&scratch[N + 1]),
                        i,
                        N + 1,
                        &mut details,
                    );
                }
                if let Some(world) = &self.world {
                    check_pair(
                        &self.links[i],
                        Some(&scratch[i]),
                        world,
                        None,
                        i,
                        N + 2,
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
                    Some(&scratch[N + 1]),
                    N,
                    N + 1,
                    &mut details,
                );
            }
            if let Some(world) = &self.world {
                check_pair(
                    &self.ee,
                    Some(&scratch[N]),
                    world,
                    None,
                    N,
                    N + 2,
                    &mut details,
                );
            }
            if let (Some(base), Some(world)) = (&self.base, &self.world) {
                check_pair(
                    base,
                    Some(&scratch[N + 1]),
                    world,
                    None,
                    N + 1,
                    N + 2,
                    &mut details,
                );
            }

            Ok(details)
        })
    }

    /// Evaluates FK at `q` and returns a single `Collider` containing every
    /// sub-collider of the validator (base, each link, the end effector,
    /// the static world body, and all of their attachments) expressed in
    /// the world frame. Base is placed at [`FKChain::base_tf`]; the world
    /// body is passed through without transformation.
    pub fn debug_bodies(&self, q: &SRobotQ<N>) -> DekeResult<Collider> {
        let (base_tf, transforms, ee_tf) = self.fk.all_fk(q).map_err(Into::into)?;
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
            push_body(base, Some(base_tf));
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..N {
            push_body(&self.links[i], Some(transforms[i]));
        }
        push_body(&self.ee, Some(ee_tf));
        if let Some(world) = &self.world {
            push_body(world, None);
        }

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

/// f64 entry point — downcasts to f32 and dispatches to the f32 impl.
/// `WreckValidator` is f32-only internally for SIMD performance; this lets
/// it plug into f64 retimers (e.g. `Topp3Tcp6`) at the cost of an inexpensive
/// f64 → f32 narrowing per query.
impl<const N: usize, FK: FKChain<N> + 'static> Validator<N, (), f64> for WreckValidator<N, FK> {
    type Context<'ctx> = WreckValidatorContext<'ctx, N>;

    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<N, E, f64>>(
        &self,
        q: A,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        let q64 = q.to_srobotq().map_err(Into::into)?;
        let q32: SRobotQ<N, f32> = q64.into();
        self.check_collisions(&q32, ctx)
    }

    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N, f64>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        for q in qs {
            let q32: SRobotQ<N, f32> = (*q).into();
            self.check_collisions(&q32, ctx)?;
        }
        Ok(())
    }
}
