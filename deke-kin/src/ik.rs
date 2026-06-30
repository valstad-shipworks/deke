//! Inverse-kinematics strategy resolution for [`Kinematics`].
//!
//! [`Kinematics`] is the single API surface: every constructor eagerly resolves
//! *how* the chain can be inverted and caches the verdict in an
//! [`IkSolverDiagnostic`]. The resolution order is:
//!
//! 1. **Analytic** — read the chain's [`KinSpec`] at the home configuration into
//!    EAIK space-frame screws and check for a known subproblem decomposition
//!    (spherical wrist, intersecting/parallel-axis families, …). Fastest and
//!    exact; covers 1–6R supported classes.
//! 2. **Generic 6R** — for an all-revolute 6-DOF chain with no known
//!    decomposition, fall back to the Raghavan–Roth/Manocha–Canny eigenvalue
//!    solver ([`crate::rr_ik`]). Complete but slower.
//! 3. **Not viable** — a chain that cannot be inverted at all (a non-revolute
//!    joint, or no strategy applies). Every [`IkSolver::ik`] call then returns
//!    [`DekeError::IkNotViable`].
//!
//! [`Kinematics`]: crate::Kinematics
//! [`KinSpec`]: deke_types::KinSpec
//! [`IkSolver::ik`]: deke_types::IkSolver::ik

use std::sync::Arc;

use crate::AAffine3;
use deke_types::{DekeError, SRobotQ};
use deke_types::{IkOutcome, IkSolutions, IkSolver, JointSpec, KinScalar, KinSpec};
use glam::{DAffine3, DMat3, DMat4, DVec3};
use glam_traits_ext::{FloatMat, TAffine3, TVec3};

use crate::Kinematics;
use crate::kinematics::scalar_from_f64;

/// A constraint applied to a joint so that an over-actuated chain (more than 6
/// DOF, or with linear/prismatic axes) can still be inverted. Rules are supplied
/// at construction; they reduce the chain to a ≤6-DOF revolute problem the
/// analytic / generic solver can handle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IkRules<F: KinScalar> {
    /// Hold joint `idx` at the fixed value `pos`. Folded into the surrounding
    /// link geometry once at construction, so it costs nothing per IK call.
    FixedAxis { idx: usize, pos: F },
    /// Sweep joint `idx` across its joint limits in increments of `step_size`,
    /// running a reduced IK solve at each sample. Produces one (or more)
    /// solution per reachable sample.
    DiscreteAxis { idx: usize, step_size: F },
    /// After solving, also emit any solution with joint `idx` wrapped by ±2π
    /// when the wrapped value still lies within the joint limits.
    IncludeWrapped { idx: usize },
}

impl<F: KinScalar> IkRules<F> {
    fn idx(&self) -> usize {
        match self {
            IkRules::FixedAxis { idx, .. }
            | IkRules::DiscreteAxis { idx, .. }
            | IkRules::IncludeWrapped { idx } => *idx,
        }
    }
}

/// Which inverse-kinematics strategy a [`Kinematics`](crate::Kinematics) chain
/// resolved to at construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IkStrategy {
    /// A closed-form subproblem decomposition is available. `family` names the
    /// recognised kinematic class (e.g. `"6R spherical wrist"`).
    Analytic { family: String },
    /// No closed form. The general Raghavan–Roth/Manocha–Canny eigenvalue solver
    /// applies: directly for a 6R chain, and for a 5R chain by lifting it to a
    /// generic 6R with a virtual joint that is required to return to zero.
    Generic6R,
    /// [`IkRules`] reduce an over-actuated chain (>6 DOF, or with linear axes)
    /// to a solvable ≤6R sub-problem. `free_dof` is the number of unconstrained
    /// joints actually solved; `discrete` is the number of swept axes.
    Ruled { free_dof: usize, discrete: usize },
    /// The chain cannot be inverted by any available strategy.
    None,
}

/// The cached verdict of IK-strategy resolution, retrievable via
/// [`Kinematics::ik_diagnostic`](crate::Kinematics::ik_diagnostic).
#[derive(Debug, Clone)]
pub struct IkSolverDiagnostic {
    /// `true` iff [`IkSolver::ik`] can produce solutions for some pose.
    pub viable: bool,
    /// The strategy chosen (or [`IkStrategy::None`]).
    pub strategy: IkStrategy,
    /// Effective actuated DOF the solver operates on.
    pub effective_dof: usize,
    /// Human-readable explanation — why a chain is not viable, or extra detail
    /// about the chosen strategy.
    pub reason: String,
}

impl IkSolverDiagnostic {
    /// `true` iff the chain resolved to an analytic decomposition.
    pub fn is_analytic(&self) -> bool {
        matches!(self.strategy, IkStrategy::Analytic { .. })
    }

    /// The recognised analytic kinematic family, if any.
    pub fn family(&self) -> Option<&str> {
        match &self.strategy {
            IkStrategy::Analytic { family } => Some(family),
            _ => None,
        }
    }
}

/// Per-joint limits in the original chain order, as `f64`.
#[derive(Debug, Clone)]
pub(crate) struct Limits<const N: usize> {
    pub lower: [f64; N],
    pub upper: [f64; N],
}

/// The role a joint plays during IK, derived from the [`IkRules`].
#[derive(Debug, Clone, Copy)]
enum Role {
    /// Solved for (must be revolute; counts toward the ≤6 budget).
    Free,
    /// Held at a fixed value, folded into geometry.
    Fixed(f64),
    /// Swept across `[lower, upper]` in `step` increments.
    Discrete { step: f64 },
}

/// Internal: the resolved engine plus its diagnostic. Shared across clones via
/// [`Arc`] so [`Kinematics`](crate::Kinematics) stays cheap to clone.
pub(crate) struct IkResolved<const N: usize> {
    diagnostic: IkSolverDiagnostic,
    /// Home-configuration KinSpec as f64, used to refold per discrete sample.
    spec: KinSpec<f64, N>,
    limits: Limits<N>,
    roles: [Role; N],
    /// Joints to additionally emit wrapped by ±2π (within limits).
    wrap: Vec<usize>,
    viable: bool,
    /// Present only for a chain with no swept (discrete) axis, where the fold is
    /// independent of the target pose. `None` for the discrete-sweep path.
    cache: Option<CachedReduced>,
}

/// Pose-independent solve artifacts for a chain with no swept (discrete) axis.
/// With no discrete sweep the fold and its reduced screws are the same for every
/// target, so they are resolved once and the analytical [`RuntimeRobot`] is held
/// ready to solve.
#[allow(dead_code)]
pub(crate) struct CachedReduced {
    joints: Vec<(DAffine3, JointSpec<f64>)>,
    free_idx: Vec<usize>,
    base: DAffine3,
    end_to_ee: DAffine3,
    h: Vec<DVec3>,
    p: Vec<DVec3>,
    r6t: DMat3,
    robot: crate::RuntimeRobot,
}

impl<const N: usize> std::fmt::Debug for IkResolved<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IkResolved")
            .field("diagnostic", &self.diagnostic)
            .finish()
    }
}

impl<const N: usize> IkResolved<N> {
    pub(crate) fn diagnostic(&self) -> &IkSolverDiagnostic {
        &self.diagnostic
    }

    /// The resolved per-joint limits as `f64` arrays `(lower, upper)`.
    pub(crate) fn limits_f64(&self) -> ([f64; N], [f64; N]) {
        (self.limits.lower, self.limits.upper)
    }
}

/// Eagerly resolve the IK strategy for a chain described by `spec`, given the
/// joint `limits` and the user `rules`.
pub(crate) fn resolve_ik<const N: usize, F: KinScalar>(
    spec_f: &KinSpec<F, N>,
    limits: Limits<N>,
    rules: &[IkRules<f64>],
) -> Arc<IkResolved<N>> {
    let spec = f64_kinspec(spec_f);

    // Assign each joint a role from the rules. Conflicting/out-of-range rules
    // make the chain not viable rather than panicking.
    let mut roles = [Role::Free; N];
    let mut wrap = Vec::new();
    let mut rule_error: Option<String> = None;
    for r in rules {
        let idx = r.idx();
        if idx >= N {
            rule_error = Some(format!(
                "rule references joint {idx} but chain has {N} joints"
            ));
            break;
        }
        match *r {
            IkRules::FixedAxis { pos, .. } => roles[idx] = Role::Fixed(pos),
            IkRules::DiscreteAxis { step_size, .. } => {
                if step_size <= 0.0 {
                    rule_error = Some(format!("DiscreteAxis on joint {idx} needs step_size > 0"));
                    break;
                }
                roles[idx] = Role::Discrete { step: step_size };
            }
            IkRules::IncludeWrapped { idx } => wrap.push(idx),
        }
    }

    let not_viable = |reason: String| {
        Arc::new(IkResolved {
            diagnostic: IkSolverDiagnostic {
                viable: false,
                strategy: IkStrategy::None,
                effective_dof: N,
                reason,
            },
            spec,
            limits: limits.clone(),
            roles,
            wrap: wrap.clone(),
            viable: false,
            cache: None,
        })
    };

    if let Some(e) = rule_error {
        return not_viable(e);
    }

    // Free joints must be revolute and number ≤ 6.
    let mut free_dof = 0usize;
    let mut discrete = 0usize;
    #[allow(clippy::needless_range_loop)]
    for i in 0..N {
        match roles[i] {
            Role::Free => {
                if matches!(spec.joints[i].1, JointSpec::Prismatic { .. }) {
                    return not_viable(format!(
                        "joint {i} is prismatic and unconstrained; add a FixedAxis or \
                         DiscreteAxis rule so the chain reduces to a revolute sub-problem"
                    ));
                }
                free_dof += 1;
            }
            Role::Discrete { .. } => discrete += 1,
            Role::Fixed(_) => {}
        }
    }

    if free_dof > 6 {
        return not_viable(format!(
            "{free_dof} unconstrained revolute joints after applying rules — at most 6 \
             can be solved; constrain {} more with FixedAxis/DiscreteAxis rules",
            free_dof - 6
        ));
    }
    if free_dof == 0 {
        return not_viable("no free joints to solve after applying rules".to_string());
    }

    // Probe a representative reduced chain (all discrete axes at their lower
    // limit) to name the kinematic strategy for the diagnostic.
    let probe: [f64; N] = std::array::from_fn(|i| match roles[i] {
        Role::Discrete { .. } => limits.lower[i],
        _ => 0.0,
    });
    let strategy_name = match reduce_and_classify(&spec, &roles, &probe) {
        Some(family) => format!("analytic ({family})"),
        None if (5..=6).contains(&free_dof) => "generic eigenvalue solver".to_string(),
        None => "reduced revolute sub-problem".to_string(),
    };

    let has_rules = rules
        .iter()
        .any(|r| !matches!(r, IkRules::IncludeWrapped { .. }));
    let strategy = if !has_rules {
        // No reducing rules: behave like the plain chain.
        match reduce_and_classify(&spec, &roles, &probe) {
            Some(family) => IkStrategy::Analytic { family },
            None if (5..=6).contains(&free_dof) => IkStrategy::Generic6R,
            None => {
                return not_viable(format!(
                    "no analytical decomposition is known for this {N}-DOF chain; the \
                     general eigenvalue solver supports 5R and 6R"
                ));
            }
        }
    } else {
        IkStrategy::Ruled { free_dof, discrete }
    };

    let cache = if discrete == 0 {
        build_cache(&spec, &roles)
    } else {
        None
    };

    Arc::new(IkResolved {
        diagnostic: IkSolverDiagnostic {
            viable: true,
            strategy,
            effective_dof: free_dof,
            reason: format!("{free_dof} free DOF, {discrete} discrete axis(es); {strategy_name}"),
        },
        spec,
        limits,
        roles,
        wrap,
        viable: true,
        cache,
    })
}

/// Fold the static (free + fixed) geometry once and build the reduced analytical
/// solver, for a chain with no swept axis. Returns `None` when the reduced chain
/// is out of the solvable `1..=6` range or has no closed-form decomposition (the
/// per-call path then handles the generic fallback).
fn build_cache<const N: usize>(spec: &KinSpec<f64, N>, roles: &[Role; N]) -> Option<CachedReduced> {
    let frozen: [Option<f64>; N] = std::array::from_fn(|i| match roles[i] {
        Role::Fixed(v) => Some(v),
        _ => None,
    });
    let (joints, free_idx, base, end_to_ee) = fold(spec, &frozen);
    if joints.is_empty() || joints.len() > 6 {
        return None;
    }
    let (h, p, r6t) = reduced_screws(base, &joints, end_to_ee);
    let robot = crate::solver_robot(&h, &p, r6t)?;
    if !robot.has_known_decomposition() {
        return None;
    }
    Some(CachedReduced {
        joints,
        free_idx,
        base,
        end_to_ee,
        h,
        p,
        r6t,
        robot,
    })
}

impl<const N: usize, F: KinScalar> IkSolver<N, F> for Kinematics<N, F> {
    type IkConfig = ();

    fn ik_with_config(
        &self,
        target: AAffine3<F>,
        _config: &(),
    ) -> Result<IkOutcome<N, F>, DekeError> {
        let resolved = self.ik_resolved();
        if !resolved.viable {
            return Err(DekeError::IkNotViable(resolved.diagnostic.reason.clone()));
        }
        let target = to_dmat4::<F>(&target);
        let sols = solve_ruled(resolved, &target);
        Ok(outcome_from_f64::<N, F>(sols))
    }
}

/// Run the full rule-driven solve: enumerate discrete samples, solve each
/// reduced chain, reassemble to N joints, FK-verify, apply IncludeWrapped, and
/// filter by joint limits.
fn solve_ruled<const N: usize>(r: &IkResolved<N>, target: &DMat4) -> Vec<[f64; N]> {
    // Indices of the discrete (swept) axes and their sample grids.
    let discrete_axes: Vec<usize> = (0..N)
        .filter(|&i| matches!(r.roles[i], Role::Discrete { .. }))
        .collect();

    if discrete_axes.is_empty()
        && let Some(cache) = &r.cache
    {
        return solve_cached(r, cache, target);
    }

    let mut out: Vec<[f64; N]> = Vec::new();

    // Optional rail fast-path: a single discrete *prismatic* axis at joint 0,
    // prune samples whose wrist-centre lies outside the reduced arm's reach.
    let reach = reduced_max_reach(r);

    // Cartesian product of all discrete-axis samples.
    let grids: Vec<Vec<f64>> = discrete_axes
        .iter()
        .map(|&i| {
            let step = match r.roles[i] {
                Role::Discrete { step } => step,
                _ => unreachable!(),
            };
            sample_grid(r.limits.lower[i], r.limits.upper[i], step)
        })
        .collect();

    let mut sample = vec![0.0f64; discrete_axes.len()];
    enumerate_samples(&grids, 0, &mut sample, &mut |sample| {
        // Freeze discrete axes at this sample, fold, solve the reduced chain.
        let mut frozen: [Option<f64>; N] = std::array::from_fn(|i| match r.roles[i] {
            Role::Fixed(v) => Some(v),
            Role::Free => None,
            Role::Discrete { .. } => None,
        });
        for (k, &i) in discrete_axes.iter().enumerate() {
            frozen[i] = Some(sample[k]);
        }

        // Cheap reach prune for a prismatic discrete first axis.
        if let Some(reach) = reach
            && discrete_axes.first() == Some(&0)
            && matches!(r.spec.joints[0].1, JointSpec::Prismatic { .. })
            && !within_reach(r, &frozen, target, reach)
        {
            return;
        }

        for full in solve_reduced(r, &frozen, target) {
            push_filtered(r, full, &mut out);
        }
    });

    out
}

/// Build an [`IkOutcome`] from f64 solutions. An empty set maps to `Failed`.
fn outcome_from_f64<const N: usize, F: KinScalar>(sols: Vec<[f64; N]>) -> IkOutcome<N, F> {
    if sols.is_empty() {
        return IkOutcome::Failed {
            partial: None,
            residual: <F as num_traits::Float>::infinity(),
        };
    }
    let out: IkSolutions<N, F> = sols
        .into_iter()
        .map(|q| SRobotQ::from_array(std::array::from_fn(|i| scalar_from_f64::<F>(q[i]))))
        .collect();
    IkOutcome::Solved(out)
}

/// Convert a chain's [`KinSpec<F, N>`] into an `f64` [`KinSpec`] of the same DOF.
pub(crate) fn kinspec_to_f64<const N: usize, F: KinScalar>(
    spec: &KinSpec<F, N>,
) -> KinSpec<f64, N> {
    f64_kinspec(spec)
}

/// Convert a chain's [`KinSpec<F, N>`] into an `f64` [`KinSpec`] of the same DOF.
fn f64_kinspec<const N: usize, F: KinScalar>(spec: &KinSpec<F, N>) -> KinSpec<f64, N> {
    let joints = std::array::from_fn(|i| {
        let (aff, js) = spec.joints[i];
        let js64 = match js {
            JointSpec::Revolute { axis_local } => JointSpec::Revolute {
                axis_local: dvec3::<F>(axis_local),
            },
            JointSpec::Prismatic { axis_local } => JointSpec::Prismatic {
                axis_local: dvec3::<F>(axis_local),
            },
        };
        (f64_affine::<F>(&aff), js64)
    });
    KinSpec::new(
        f64_affine::<F>(&spec.base_to_first),
        joints,
        f64_affine::<F>(&spec.end_to_ee),
    )
}

/// One joint transform of an f64 [`KinSpec`] at value `q`:
/// `parent_to_joint · joint_motion(q)`.
fn joint_tf(j: &(DAffine3, JointSpec<f64>), q: f64) -> DAffine3 {
    let motion = match j.1 {
        JointSpec::Revolute { axis_local } => DAffine3::from_axis_angle(axis_local.normalize(), q),
        JointSpec::Prismatic { axis_local } => {
            DAffine3::from_translation(axis_local.normalize() * q)
        }
    };
    j.0 * motion
}

/// Full forward kinematics of an f64 [`KinSpec`] at configuration `q`.
fn kinspec_fk<const N: usize>(spec: &KinSpec<f64, N>, q: &[f64; N]) -> DMat4 {
    let mut t = spec.base_to_first;
    for (joint, &qi) in spec.joints.iter().zip(q.iter()) {
        t *= joint_tf(joint, qi);
    }
    t *= spec.end_to_ee;
    DMat4::from(t)
}

/// Reduce the chain by folding every joint with a concrete value in `frozen`
/// EAIK space-frame screws (H, P, R6T) for a reduced free-joint chain, given the
/// folded joints and the composed `end_to_ee`. Mirrors `Robot`'s screw
/// extraction but over a runtime-sized chain.
fn reduced_screws(
    base: DAffine3,
    joints: &[(DAffine3, JointSpec<f64>)],
    end_to_ee: DAffine3,
) -> (Vec<DVec3>, Vec<DVec3>, DMat3) {
    let n = joints.len();
    let mut h = Vec::with_capacity(n);
    let mut origins = Vec::with_capacity(n);
    let mut current = base;
    for j in joints {
        current *= j.0;
        origins.push(current.translation);
        let axis = match j.1 {
            JointSpec::Revolute { axis_local } => axis_local.normalize(),
            // A reduced chain is all-revolute by construction (free prismatic
            // joints make the chain non-viable); treat defensively as Z.
            JointSpec::Prismatic { .. } => DVec3::Z,
        };
        h.push((current.matrix3 * axis).normalize());
    }
    let ee = current * end_to_ee;
    let r6t = ee.matrix3;
    let mut p = Vec::with_capacity(n + 1);
    if n > 0 {
        p.push(origins[0]);
        for i in 1..n {
            p.push(origins[i] - origins[i - 1]);
        }
        p.push(ee.translation - origins[n - 1]);
    }
    (h, p, r6t)
}

/// Fold `frozen` joints into the surrounding geometry and return the reduced
/// free joints, their original indices, and the composed `(base, end_to_ee)`.
/// A frozen revolute folds in as a fixed rotation and a frozen prismatic as a
/// fixed translation — the "fold a fixed axis into the next link as a static
/// offset" optimization, generalized to linear axes.
type FoldedChain = (
    Vec<(DAffine3, JointSpec<f64>)>,
    Vec<usize>,
    DAffine3,
    DAffine3,
);

fn fold<const N: usize>(spec: &KinSpec<f64, N>, frozen: &[Option<f64>; N]) -> FoldedChain {
    let mut joints: Vec<(DAffine3, JointSpec<f64>)> = Vec::new();
    let mut free_idx: Vec<usize> = Vec::new();
    let mut pending = spec.base_to_first;
    let mut base = spec.base_to_first;
    let mut prefix_set = false;
    #[allow(clippy::needless_range_loop)]
    for i in 0..N {
        match frozen[i] {
            Some(v) => pending *= joint_tf(&spec.joints[i], v),
            None => {
                let origin = if prefix_set {
                    pending * spec.joints[i].0
                } else {
                    base = pending;
                    prefix_set = true;
                    spec.joints[i].0
                };
                joints.push((origin, spec.joints[i].1));
                free_idx.push(i);
                pending = DAffine3::IDENTITY;
            }
        }
    }
    if !prefix_set {
        base = pending;
    }
    let end_to_ee = pending * spec.end_to_ee;
    (joints, free_idx, base, end_to_ee)
}

/// Classify the reduced chain (at the given full config) into an analytic
/// kinematic family name, or `None` if no closed form is recognised.
fn reduce_and_classify<const N: usize>(
    spec: &KinSpec<f64, N>,
    roles: &[Role; N],
    config: &[f64; N],
) -> Option<String> {
    let frozen: [Option<f64>; N] = std::array::from_fn(|i| match roles[i] {
        Role::Free => None,
        Role::Fixed(v) => Some(v),
        Role::Discrete { .. } => Some(config[i]),
    });
    let (joints, _free, base, end_to_ee) = fold(spec, &frozen);
    if joints.is_empty() || joints.len() > 6 {
        return None;
    }
    let (h, p, r6t) = reduced_screws(base, &joints, end_to_ee);
    let robot = crate::solver_robot(&h, &p, r6t)?;
    if robot.has_known_decomposition() {
        Some(robot.kinematic_family())
    } else {
        None
    }
}

/// Solve the reduced chain for `frozen` (all non-free joints have a value) at
/// `target`, reassemble each solution back to a full N-vector, FK-verify, and
/// emit IncludeWrapped variants. Returns full N-joint solutions (unfiltered by
/// limits — the caller filters).
fn solve_reduced<const N: usize>(
    r: &IkResolved<N>,
    frozen: &[Option<f64>; N],
    target: &DMat4,
) -> Vec<[f64; N]> {
    let (joints, free_idx, base, end_to_ee) = fold(&r.spec, frozen);
    if joints.is_empty() || joints.len() > 6 {
        return Vec::new();
    }
    let (h, p, r6t) = reduced_screws(base, &joints, end_to_ee);

    // Reduced-chain analytic solve (mirrors Robot::ik) over the runtime chain;
    // falls back to the generic 6R solver when the reduced chain is exactly 6R
    // with no closed form.
    let reduced_sols = crate::solve_reduced_chain(&h, &p, r6t, &joints, base, end_to_ee, target);

    let mut out = Vec::new();
    for reduced in reduced_sols {
        // Reassemble: free joints take solved values, others their frozen value.
        let mut full = [0.0f64; N];
        for i in 0..N {
            full[i] = frozen[i].unwrap_or(0.0);
        }
        for (k, &orig) in free_idx.iter().enumerate() {
            full[orig] = reduced.as_slice()[k];
        }
        // FK-verify against the true full chain — a fold or screw error cannot
        // ship a wrong solution.
        if pose_close(&kinspec_fk(&r.spec, &full), target) {
            out.push(full);
            // IncludeWrapped variants.
            for &w in &r.wrap {
                for delta in [-std::f64::consts::TAU, std::f64::consts::TAU] {
                    let wrapped = full[w] + delta;
                    if wrapped >= r.limits.lower[w] && wrapped <= r.limits.upper[w] {
                        let mut v = full;
                        v[w] = wrapped;
                        out.push(v);
                    }
                }
            }
        }
    }
    out
}

/// Solve via the precomputed [`CachedReduced`] for a chain with no swept axis:
/// the cached analytical solver yields free-joint values directly, which are
/// reassembled, FK-verified, and limit-filtered exactly as [`solve_reduced`]
/// does per call.
fn solve_cached<const N: usize>(
    r: &IkResolved<N>,
    cache: &CachedReduced,
    target: &DMat4,
) -> Vec<[f64; N]> {
    let mut out: Vec<[f64; N]> = Vec::new();
    for reduced in cache.robot.solve(*target) {
        let mut full = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for i in 0..N {
            if let Role::Fixed(v) = r.roles[i] {
                full[i] = v;
            }
        }
        for (k, &orig) in cache.free_idx.iter().enumerate() {
            full[orig] = reduced.as_slice()[k];
        }
        if pose_close(&kinspec_fk(&r.spec, &full), target) {
            push_filtered(r, full, &mut out);
            for &w in &r.wrap {
                for delta in [-std::f64::consts::TAU, std::f64::consts::TAU] {
                    let wrapped = full[w] + delta;
                    if wrapped >= r.limits.lower[w] && wrapped <= r.limits.upper[w] {
                        let mut v = full;
                        v[w] = wrapped;
                        push_filtered(r, v, &mut out);
                    }
                }
            }
        }
    }
    out
}

/// Append `full` to `out` iff it lies within joint limits and is not a duplicate.
fn push_filtered<const N: usize>(r: &IkResolved<N>, full: [f64; N], out: &mut Vec<[f64; N]>) {
    #[allow(clippy::needless_range_loop)]
    for i in 0..N {
        if full[i] < r.limits.lower[i] - 1e-9 || full[i] > r.limits.upper[i] + 1e-9 {
            return;
        }
    }
    if out
        .iter()
        .any(|e| (0..N).all(|i| (e[i] - full[i]).abs() < 1e-6))
    {
        return;
    }
    out.push(full);
}

/// Inclusive sample grid `[lo, hi]` stepped by `step` (always includes `hi`).
fn sample_grid(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let mut v = Vec::new();
    if hi < lo || step <= 0.0 {
        return v;
    }
    let n = ((hi - lo) / step).floor() as usize;
    for k in 0..=n {
        v.push(lo + step * k as f64);
    }
    if (v.last().copied().unwrap_or(lo) - hi).abs() > 1e-9 {
        v.push(hi);
    }
    v
}

/// Recurse over the Cartesian product of per-axis sample grids.
fn enumerate_samples(
    grids: &[Vec<f64>],
    depth: usize,
    sample: &mut [f64],
    f: &mut impl FnMut(&[f64]),
) {
    if depth == grids.len() {
        f(sample);
        return;
    }
    for &v in &grids[depth] {
        sample[depth] = v;
        enumerate_samples(grids, depth + 1, sample, f);
    }
}

/// Maximum reach of the reduced arm (all free joints), used to prune rail
/// samples. `None` when there is no prismatic discrete first axis to optimize.
fn reduced_max_reach<const N: usize>(r: &IkResolved<N>) -> Option<f64> {
    if !matches!(
        r.spec.joints.first().map(|j| j.1),
        Some(JointSpec::Prismatic { .. })
    ) {
        return None;
    }
    if !matches!(r.roles[0], Role::Discrete { .. }) {
        return None;
    }
    // Reach = sum of free-joint link lengths with the rail removed.
    let frozen: [Option<f64>; N] = std::array::from_fn(|i| match r.roles[i] {
        Role::Free => None,
        Role::Fixed(v) => Some(v),
        Role::Discrete { .. } => Some(r.limits.lower[i]),
    });
    let (joints, _free, _base, end_to_ee) = fold(&r.spec, &frozen);
    let mut total = 0.0;
    let mut prev = DVec3::ZERO;
    let mut cur = DAffine3::IDENTITY;
    for (k, j) in joints.iter().enumerate() {
        cur *= j.0;
        if k > 0 {
            total += (cur.translation - prev).length();
        }
        prev = cur.translation;
    }
    total += (end_to_ee.translation).length();
    Some(total)
}

/// For a rail fast-path: does the wrist centre (target origin minus the rail
/// offset at this sample) lie within the reduced arm's reach?
fn within_reach<const N: usize>(
    r: &IkResolved<N>,
    frozen: &[Option<f64>; N],
    target: &DMat4,
    reach: f64,
) -> bool {
    // Rail base position at the frozen sample.
    let rail = joint_tf(&r.spec.joints[0], frozen[0].unwrap_or(0.0));
    let base = (r.spec.base_to_first * rail).translation;
    let tip = target.w_axis.truncate();
    (tip - base).length() <= reach + 1e-6
}

fn pose_close(a: &DMat4, b: &DMat4) -> bool {
    let aa = a.to_cols_array();
    let bb = b.to_cols_array();
    aa.iter().zip(bb.iter()).all(|(x, y)| (x - y).abs() < 1e-6)
}

#[inline]
fn to_dmat4<F: KinScalar>(a: &AAffine3<F>) -> DMat4 {
    let m = a.matrix3();
    let t = a.translation();
    DMat4::from_cols(
        dvec3::<F>(m.col(0)).extend(0.0),
        dvec3::<F>(m.col(1)).extend(0.0),
        dvec3::<F>(m.col(2)).extend(0.0),
        dvec3::<F>(t).extend(1.0),
    )
}

#[inline]
fn f64_affine<F: KinScalar>(a: &AAffine3<F>) -> DAffine3 {
    let m = a.matrix3();
    DAffine3::from_mat3_translation(
        DMat3::from_cols(
            dvec3::<F>(m.col(0)),
            dvec3::<F>(m.col(1)),
            dvec3::<F>(m.col(2)),
        ),
        dvec3::<F>(a.translation()),
    )
}

#[inline]
fn dvec3<F: KinScalar>(v: <F as KinScalar>::AVec3) -> DVec3 {
    DVec3::new(to_f64(v.x()), to_f64(v.y()), to_f64(v.z()))
}

#[inline]
fn to_f64<F: KinScalar>(x: F) -> f64 {
    num_traits::ToPrimitive::to_f64(&x).expect("scalar is representable as f64")
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::{DHJoint, JointLimits};

    fn puma() -> Kinematics<6, f64> {
        let pi = std::f64::consts::PI;
        let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
        let a = [0.0, 0.4318, -0.0203, 0.0, 0.0, 0.0];
        let d = [0.6718, 0.1397, 0.0, 0.4318, 0.0, 0.0565];
        let joints = std::array::from_fn(|i| DHJoint {
            a: a[i],
            alpha: alpha[i],
            d: d[i],
            theta_offset: 0.0,
        });
        Kinematics::from_dh(joints, JointLimits::symmetric(100.0), &[])
    }

    #[test]
    fn cached_path_matches_uncached_bit_for_bit() {
        let chain = puma();
        let r = chain.ik_resolved();
        assert!(
            r.cache.is_some(),
            "static Puma geometry must populate the reduced cache"
        );

        for cfg in [
            [0.0, 0.3, -0.4, 0.2, 0.5, -0.1],
            [-1.0, 0.7, -0.3, 1.2, -0.9, 0.4],
            [0.8, -0.6, 1.1, -0.2, 0.9, 1.3],
        ] {
            let target = kinspec_fk(&r.spec, &cfg);

            let cached = solve_ruled(r, &target);

            let frozen: [Option<f64>; 6] = [None; 6];
            let mut expected: Vec<[f64; 6]> = Vec::new();
            for full in solve_reduced(r, &frozen, &target) {
                push_filtered(r, full, &mut expected);
            }

            assert_eq!(
                cached.len(),
                expected.len(),
                "solution count mismatch for {cfg:?}"
            );
            for (c, e) in cached.iter().zip(expected.iter()) {
                for k in 0..6 {
                    assert_eq!(c[k].to_bits(), e[k].to_bits(), "bit mismatch for {cfg:?}");
                }
            }
        }
    }
}
