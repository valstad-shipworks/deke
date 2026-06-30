//! reaik — pure-Rust inverse kinematics for serial manipulators.
//!
//! [`Kinematics`] is the single API surface. Build it from Denavit-Hartenberg,
//! Hayati-Paul, or URDF parameters, or directly from a
//! [`KinSpec`](deke_types::KinSpec); it implements forward kinematics
//! ([`FKChain`](deke_types::FKChain)) and inverse kinematics
//! ([`IkSolver`](deke_types::IkSolver)).
//!
//! Every constructor requires per-joint [`JointLimits`] and a slice of
//! [`IkRules`]. At construction the chain eagerly resolves *how* it can be
//! inverted and caches the verdict ([`Kinematics::ik_diagnostic`]):
//!
//! - a known closed-form subproblem decomposition (analytical, 1–6R), else
//! - the general Raghavan–Roth/Manocha–Canny eigenvalue solver for an
//!   all-revolute 6-DOF chain ([`rr_ik`]), else
//! - a rule-reduced sub-problem ([`IkStrategy::Ruled`]) when [`IkRules`]
//!   constrain an over-actuated chain (>6 DOF, or with linear axes) down to a
//!   solvable ≤6R, else
//! - **not viable** — a chain the rules don't constrain enough, for which every
//!   [`ik`](deke_types::IkSolver::ik) call returns
//!   [`DekeError::IkNotViable`](deke_types::DekeError::IkNotViable). The chain
//!   still constructs and its forward kinematics still work.
//!
//! [`IkRules`] enable over-actuated chains: [`IkRules::FixedAxis`] folds a joint
//! out at a fixed value, [`IkRules::DiscreteAxis`] sweeps a joint across its
//! limits (one solve per sample — ideal for a linear rail), and
//! [`IkRules::IncludeWrapped`] additionally emits ±2π variants within limits.
//! IK output is filtered to the joint limits.
//!
//! Joint configurations are [`SRobotQ<N, F>`](deke_types::SRobotQ) (`F` is
//! `f32` or `f64`); IK targets are `AAffine3<F>` end-effector poses.
//!
//! # Example
//!
//! ```no_run
//! use deke_kin::{Kinematics, DHJoint, JointLimits};
//! use deke_kin::deke_types::SRobotQ;
//! use deke_kin::deke_types::{FKChain, IkSolver};
//!
//! let pi = std::f64::consts::PI;
//! // Puma 560 DH parameters.
//! let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
//! let a = [0.0, 0.43180, -0.02032, 0.0, 0.0, 0.0];
//! let d = [0.67183, 0.13970, 0.0, 0.43180, 0.0, 0.0565];
//! let robot = Kinematics::<6, f64>::from_dh(
//!     std::array::from_fn(|i| DHJoint { a: a[i], alpha: alpha[i], d: d[i], theta_offset: 0.0 }),
//!     JointLimits::symmetric(pi),       // per-joint limits (required)
//!     &[],                              // no rules: a plain 6R
//! );
//!
//! // Inspect the resolved strategy.
//! println!("{:?}", robot.ik_diagnostic().strategy);
//!
//! let q = SRobotQ::<6, f64>::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
//! let target = robot.fk_end(&q).unwrap();
//! let outcome = robot.ik(target).unwrap();
//! let solutions = outcome.unwrap(); // IkSolutions<6, f64> (stack-optimized)
//! ```

pub use deke_types;
pub use glam;

mod ik;
pub(crate) mod ik_geo;
mod kinematics;
mod remodel;
pub mod rr_ik;
pub mod snap;
mod solver;

pub use ik::{IkRules, IkSolverDiagnostic, IkStrategy};
pub use kinematics::{
    DHJoint, HPJoint, JointLimits, Kinematics, URDFBuildError, URDFJoint, URDFJointType,
};
pub use rr_ik::{RrConfig, RrSpecError, solve_kinspec};

use deke_types::{KinScalar, SRobotQ};
use glam::{DMat3, DMat4, DVec3};

#[allow(type_alias_bounds)]
pub(crate) type AAffine3<F: KinScalar> = F::AAffine3;
#[allow(type_alias_bounds)]
pub(crate) type AMat3<F: KinScalar> = F::AMat3;
#[allow(type_alias_bounds)]
pub(crate) type AVec3<F: KinScalar> = F::AVec3;

/// Errors produced by reaik operations.
#[derive(Debug, Clone)]
pub enum Error {
    /// Dimensions of H/P inputs are inconsistent at runtime (e.g. `p.len() != N + 1`).
    DimensionMismatch(String),
    /// More than 6 (or fewer than 1) effective joints after locking.
    UnsupportedDof(usize),
    /// IK requested on a kinematic class without a known analytical decomposition.
    UnknownDecomposition,
    /// An [`AnalyticIk`] was constructed from a chain containing a non-revolute
    /// (prismatic) joint, which the analytical solver does not support.
    NonRevoluteJoint,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch(s) => write!(f, "dimension mismatch: {s}"),
            Self::UnsupportedDof(n) => write!(
                f,
                "{n} effective joints — only 1..=6 are supported (lock redundant axes)"
            ),
            Self::UnknownDecomposition => {
                f.write_str("no analytical subproblem decomposition is known for this manipulator")
            }
            Self::NonRevoluteJoint => f.write_str("analytical IK supports revolute joints only"),
        }
    }
}

impl std::error::Error for Error {}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// A joint that should be locked at a fixed angle (e.g. for redundant manipulators).
#[derive(Debug, Clone, Copy)]
pub struct FixedAxis {
    /// Zero-indexed joint to lock in the original chain.
    pub joint: usize,
    /// Angle (radians) to lock the joint at.
    pub angle: f64,
}

impl FixedAxis {
    pub const fn new(joint: usize, angle: f64) -> Self {
        Self { joint, angle }
    }
}

/// A single IK solution: a statically-sized joint configuration in
/// original-robot order (i.e. including any locked / fixed joints).
pub type IkSolution<const N: usize> = SRobotQ<N, f64>;

/// A 1–6R serial manipulator with `N` joints (original chain, including any
/// locked axes) that can compute analytical IK.
///
/// Internal analytical engine behind [`Kinematics`]'s [`IkSolver`] impl;
/// construct chains through [`Kinematics`] rather than this type directly.
///
/// [`IkSolver`]: deke_types::IkSolver
///
/// Several fields/accessors are read only by the engine's own test suite; in a
/// non-test build the production path (`ik::resolve_ik`) uses just construction
/// plus `ik`/`has_known_decomposition`/`kinematic_family`/`n_effective_joints`.
#[allow(dead_code)]
pub(crate) struct Robot<const N: usize> {
    original_h: Vec<DVec3>,
    original_p: Vec<DVec3>,
    /// End-effector orientation relative to frame `N` in the **original** chain.
    r6t: DMat3,

    remodeled_h: Vec<DVec3>,
    remodeled_p: Vec<DVec3>,
    /// End-effector orientation relative to the last joint of the reduced
    /// (partially-locked) chain.
    r6t_partial: DMat3,

    /// Locked joints in ascending order of `joint`. Their angles are inserted
    /// back into the joint vector after IK is solved on the reduced chain.
    fixed_axes: Vec<FixedAxis>,

    solver: solver::Solver,
}

#[allow(dead_code)]
impl<const N: usize> Robot<N> {
    /// Construct a robot from standard Denavit-Hartenberg parameters.
    pub fn from_dh(alpha: &[f64; N], a: &[f64; N], d: &[f64; N]) -> Result<Self> {
        Self::from_dh_with(alpha, a, d, DMat3::IDENTITY, &[])
    }

    /// Construct a DH robot with end-effector offset and locked joints.
    pub fn from_dh_with(
        alpha: &[f64; N],
        a: &[f64; N],
        d: &[f64; N],
        r6t: DMat3,
        fixed_axes: &[FixedAxis],
    ) -> Result<Self> {
        let (h, p, r_dh) = kinematics::dh_to_hp(alpha, a, d);
        let r6t_total = r6t * r_dh;
        Self::build(h, p, r6t_total, fixed_axes)
    }

    /// Construct a robot from H (joint axes) and P (offsets) vectors.
    ///
    /// `h` has length `N` (one axis per joint); `p` must have length `N + 1`
    /// (one offset between consecutive frames, with the first being the base
    /// offset and the last being the end-effector offset). The length of `p`
    /// is checked at runtime because stable Rust cannot yet express `N + 1`
    /// as a const expression.
    pub fn from_hp(h: &[DVec3; N], p: &[DVec3]) -> Result<Self> {
        Self::from_hp_with(h, p, DMat3::IDENTITY, &[])
    }

    /// Construct an HP robot with end-effector offset and locked joints.
    pub fn from_hp_with(
        h: &[DVec3; N],
        p: &[DVec3],
        r6t: DMat3,
        fixed_axes: &[FixedAxis],
    ) -> Result<Self> {
        if p.len() != N + 1 {
            return Err(Error::DimensionMismatch(format!(
                "expected p.len() == {} (N + 1), got {}",
                N + 1,
                p.len()
            )));
        }
        Self::build(h.to_vec(), p.to_vec(), r6t, fixed_axes)
    }

    /// Like [`Self::from_hp_with`] but with caller-supplied tolerances. The
    /// default constructors use ZERO_THRESHOLD = 1e-13 and
    /// AXIS_INTERSECT_THRESHOLD = 1e-9 (mirroring the C++ EAIK constants);
    /// pass wider values here to accept a chain whose axes are only
    /// approximately parallel / intersecting (e.g. a calibrated robot whose
    /// nominal-spherical wrist has small numerical residuals).
    pub fn from_hp_with_thresholds(
        h: &[DVec3; N],
        p: &[DVec3],
        r6t: DMat3,
        fixed_axes: &[FixedAxis],
        zero_threshold: f64,
        axis_intersect_threshold: f64,
    ) -> Result<Self> {
        if p.len() != N + 1 {
            return Err(Error::DimensionMismatch(format!(
                "expected p.len() == {} (N + 1), got {}",
                N + 1,
                p.len()
            )));
        }
        Self::build_with(
            h.to_vec(),
            p.to_vec(),
            r6t,
            fixed_axes,
            zero_threshold,
            axis_intersect_threshold,
        )
    }

    fn build(h: Vec<DVec3>, p: Vec<DVec3>, r6t: DMat3, fixed_axes: &[FixedAxis]) -> Result<Self> {
        Self::build_with(h, p, r6t, fixed_axes, 1e-13, 1e-9)
    }

    fn build_with(
        mut h: Vec<DVec3>,
        p: Vec<DVec3>,
        r6t: DMat3,
        fixed_axes: &[FixedAxis],
        zero_threshold: f64,
        axis_intersect_threshold: f64,
    ) -> Result<Self> {
        debug_assert_eq!(h.len(), N);
        debug_assert_eq!(p.len(), N + 1);

        let zero_thresh = zero_threshold;
        let axis_thresh = axis_intersect_threshold;

        // Pre-normalise so every downstream `axis_angle` call can skip it.
        remodel::normalise_axes(&mut h);

        let (h_reduced, p_reduced, r6t_partial) = if !fixed_axes.is_empty() {
            kinematics::partial_joint_parametrization(&h, &p, fixed_axes, &r6t)
        } else {
            (h.clone(), p.clone(), r6t)
        };

        let p_remodeled =
            remodel::remodel_kinematics(&h_reduced, &p_reduced, zero_thresh, axis_thresh);

        let n_eff = h_reduced.len();
        if !(1..=6).contains(&n_eff) {
            return Err(Error::UnsupportedDof(n_eff));
        }

        let solver = solver::Solver::build(&h_reduced, &p_remodeled, zero_thresh, axis_thresh);

        let mut sorted_fixed = fixed_axes.to_vec();
        sorted_fixed.sort_by_key(|f| f.joint);

        Ok(Self {
            original_h: h,
            original_p: p,
            r6t,
            remodeled_h: h_reduced,
            remodeled_p: p_remodeled,
            r6t_partial,
            fixed_axes: sorted_fixed,
            solver,
        })
    }

    /// Total joints in the original chain — equal to the const generic `N`.
    pub const fn n_joints(&self) -> usize {
        N
    }

    /// Joints remaining after locking fixed axes — what the underlying solver
    /// operates on (always in `1..=6`).
    pub fn n_effective_joints(&self) -> usize {
        self.remodeled_h.len()
    }

    /// `true` if the robot has an analytically-solvable kinematic decomposition.
    pub fn has_known_decomposition(&self) -> bool {
        self.solver.has_known_decomposition()
    }

    /// `true` if the robot has a spherical wrist.
    pub fn is_spherical(&self) -> bool {
        self.solver.is_spherical()
    }

    /// Descriptive string for the robot's kinematic class.
    pub fn kinematic_family(&self) -> String {
        self.solver.kinematic_family()
    }

    /// Joint axes after kinematic remodeling.
    pub fn remodeled_h(&self) -> &[DVec3] {
        &self.remodeled_h
    }

    /// Joint offsets after kinematic remodeling.
    pub fn remodeled_p(&self) -> &[DVec3] {
        &self.remodeled_p
    }

    /// Original (un-remodelled) joint axes.
    pub fn original_h(&self) -> &[DVec3] {
        &self.original_h
    }

    /// Original (un-remodelled) joint offsets.
    pub fn original_p(&self) -> &[DVec3] {
        &self.original_p
    }

    /// Forward kinematics. `q` includes any locked joints.
    pub fn fk(&self, q: &SRobotQ<N, f64>) -> DMat4 {
        let pose = kinematics::fwdkin(&self.original_h, &self.original_p, q.as_slice());
        kinematics::apply_r6t(pose, &self.r6t)
    }

    /// Compute all analytical IK solutions for `pose`.
    pub fn ik(&self, mut pose: DMat4) -> Result<Vec<IkSolution<N>>> {
        if !self.solver.has_known_decomposition() {
            return Err(Error::UnknownDecomposition);
        }

        // R'_06 = R_06 · R6Tᵀ — strip the end-effector offset so the solver
        // sees orientation relative to the reduced chain's last joint.
        let r = DMat3::from_cols(
            pose.x_axis.truncate(),
            pose.y_axis.truncate(),
            pose.z_axis.truncate(),
        ) * self.r6t_partial.transpose();
        pose.x_axis = r.x_axis.extend(0.0);
        pose.y_axis = r.y_axis.extend(0.0);
        pose.z_axis = r.z_axis.extend(0.0);

        let raw = self.solver.solve(&pose);
        Ok(raw
            .into_iter()
            .map(|reduced| self.assemble_solution(reduced))
            .collect())
    }

    /// Walk a reduced-chain solution (always 6-slot [`solver::Joints`] padded
    /// with zeros after `n_eff` entries) and re-insert locked-axis values to
    /// produce a full N-joint configuration.
    fn assemble_solution(&self, reduced: solver::Joints) -> SRobotQ<N, f64> {
        let n_eff = self.remodeled_h.len();
        let active = &reduced.as_slice()[..n_eff];

        if self.fixed_axes.is_empty() {
            debug_assert_eq!(active.len(), N);
            let mut arr = [0.0f64; N];
            arr.copy_from_slice(active);
            return SRobotQ::from_array(arr);
        }

        let mut buf: Vec<f64> = active.to_vec();
        for fa in &self.fixed_axes {
            if fa.joint <= buf.len() {
                buf.insert(fa.joint, fa.angle);
            } else {
                buf.push(fa.angle);
            }
        }
        debug_assert_eq!(buf.len(), N, "assembled solution has wrong DOF");
        let mut arr = [0.0f64; N];
        arr.copy_from_slice(&buf);
        SRobotQ::from_array(arr)
    }
}

/// Build an analytical [`Robot`] of *runtime* DOF from screws, for classifying a
/// reduced chain. Returns `None` if the DOF is out of `1..=6`. Used by the
/// rule-driven IK path in [`ik`].
pub(crate) fn solver_robot(h: &[DVec3], p: &[DVec3], r6t: DMat3) -> Option<RuntimeRobot> {
    if !(1..=6).contains(&h.len()) || p.len() != h.len() + 1 {
        return None;
    }
    Some(RuntimeRobot::build(h.to_vec(), p.to_vec(), r6t))
}

/// A runtime-DOF analytical solver over a reduced (already-folded) revolute
/// chain. Mirrors [`Robot`]'s internals without the const-generic `N`.
pub(crate) struct RuntimeRobot {
    r6t_partial: DMat3,
    solver: solver::Solver,
}

impl RuntimeRobot {
    fn build(mut h: Vec<DVec3>, p: Vec<DVec3>, r6t: DMat3) -> Self {
        remodel::normalise_axes(&mut h);
        let p_remodeled = remodel::remodel_kinematics(&h, &p, 1e-13, 1e-9);
        let solver = solver::Solver::build(&h, &p_remodeled, 1e-13, 1e-9);
        Self {
            r6t_partial: r6t,
            solver,
        }
    }

    pub(crate) fn has_known_decomposition(&self) -> bool {
        self.solver.has_known_decomposition()
    }

    pub(crate) fn kinematic_family(&self) -> String {
        self.solver.kinematic_family()
    }

    /// Solve the reduced chain at `pose`. Each solution is padded to 6 slots;
    /// only the first effective-DOF entries are meaningful.
    pub(crate) fn solve(&self, mut pose: DMat4) -> solver::Solutions {
        if !self.solver.has_known_decomposition() {
            return solver::Solutions::new();
        }
        let r = DMat3::from_cols(
            pose.x_axis.truncate(),
            pose.y_axis.truncate(),
            pose.z_axis.truncate(),
        ) * self.r6t_partial.transpose();
        pose.x_axis = r.x_axis.extend(0.0);
        pose.y_axis = r.y_axis.extend(0.0);
        pose.z_axis = r.z_axis.extend(0.0);
        self.solver.solve(&pose)
    }
}

/// Solve a reduced free-joint chain at `target`, returning per-solution
/// free-joint value vectors. Uses the analytical solver when the chain has a
/// known decomposition, else the generic 6R eigenvalue solver when the reduced
/// chain is exactly 6R. `joints`/`base`/`end_to_ee` describe the reduced chain
/// for the generic fallback.
pub(crate) fn solve_reduced_chain(
    h: &[DVec3],
    p: &[DVec3],
    r6t: DMat3,
    joints: &[(glam::DAffine3, deke_types::JointSpec<f64>)],
    base: glam::DAffine3,
    end_to_ee: glam::DAffine3,
    target: &DMat4,
) -> solver::Solutions {
    let Some(robot) = solver_robot(h, p, r6t) else {
        return solver::Solutions::new();
    };
    if robot.has_known_decomposition() {
        return robot.solve(*target);
    }
    let n = joints.len();
    if n == 6 {
        let spec_joints: [(glam::DAffine3, deke_types::JointSpec<f64>); 6] =
            std::array::from_fn(|i| joints[i]);
        let spec = deke_types::KinSpec::new(base, spec_joints, end_to_ee);
        if let Ok(sols) = rr_ik::solve_kinspec(&spec, *target, &rr_ik::RrConfig::default()) {
            return sols.into_iter().collect();
        }
    } else if n == 5 {
        return solve_generic_5r(joints, base, end_to_ee, target);
    }
    solver::Solutions::new()
}

/// One virtual revolute joint that lifts a 5R chain to a generic 6R for the
/// eigenvalue fallback. At a zero joint angle it contributes only its constant
/// frame (which the caller cancels into `end_to_ee`), so a lifted solution with
/// the virtual joint at zero is exactly a solution of the original 5R. The
/// constants are arbitrary, chosen so the lifted 6R carries no
/// parallel/intersecting-axis structure that would defeat the general solver.
fn virtual_pad_joint() -> (glam::DAffine3, deke_types::JointSpec<f64>) {
    use glam::DAffine3;
    let g = DAffine3::from_translation(DVec3::new(0.137, -0.211, 0.173))
        * DAffine3::from_axis_angle(DVec3::new(0.41, 0.57, 0.71).normalize(), 0.61);
    (
        g,
        deke_types::JointSpec::Revolute {
            axis_local: DVec3::new(0.33, 0.62, 0.71).normalize(),
        },
    )
}

/// Solve a non-decomposable 5R reduced chain by lifting it to a generic 6R with
/// a virtual joint, solving with the eigenvalue solver, and keeping only the
/// solutions whose virtual joint returns to zero (the embeddings of the 5R).
fn solve_generic_5r(
    joints: &[(glam::DAffine3, deke_types::JointSpec<f64>)],
    base: glam::DAffine3,
    end_to_ee: glam::DAffine3,
    target: &DMat4,
) -> solver::Solutions {
    let v = virtual_pad_joint();
    let spec_joints: [(glam::DAffine3, deke_types::JointSpec<f64>); 6] =
        std::array::from_fn(|i| if i < 5 { joints[i] } else { v });
    let spec = deke_types::KinSpec::new(base, spec_joints, v.0.inverse() * end_to_ee);
    let Ok(sols) = rr_ik::solve_kinspec(&spec, *target, &rr_ik::RrConfig::default()) else {
        return solver::Solutions::new();
    };

    const VIRT_TOL: f64 = 1e-4;
    let mut out = solver::Solutions::new();
    for s in sols {
        let a = s.0;
        if a[5].abs() < VIRT_TOL {
            let mut buf = [0.0f64; 6];
            buf[..5].copy_from_slice(&a[..5]);
            out.push(SRobotQ::from_array(buf));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! Tests for the internal analytical [`Robot`] engine. The public IK surface
    //! is [`Kinematics`] (see `src/ik.rs` and `kinematics/fk_chain.rs` tests);
    //! these exercise the engine directly, which is what those public paths
    //! delegate to.
    use super::*;
    use glam::DAffine3;

    fn xs(s: &mut u64) -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s >> 11) as f64 / (1u64 << 53) as f64
    }
    fn rr(s: &mut u64, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * xs(s)
    }
    fn raxis(s: &mut u64) -> DVec3 {
        DVec3::new(rr(s, -1.0, 1.0), rr(s, -1.0, 1.0), rr(s, -1.0, 1.0)).normalize()
    }
    fn raff(s: &mut u64) -> DAffine3 {
        DAffine3::from_axis_angle(raxis(s), rr(s, -2.5, 2.5))
            * DAffine3::from_translation(DVec3::new(
                rr(s, -0.3, 0.3),
                rr(s, -0.3, 0.3),
                rr(s, 0.05, 0.3),
            ))
    }
    fn axis_of(j: &(DAffine3, deke_types::JointSpec<f64>)) -> DVec3 {
        match j.1 {
            deke_types::JointSpec::Revolute { axis_local } => axis_local,
            _ => unreachable!("generic padded test uses revolute joints only"),
        }
    }
    fn fk_nr(
        joints: &[(DAffine3, deke_types::JointSpec<f64>)],
        base: DAffine3,
        end: DAffine3,
        q: &[f64],
    ) -> DMat4 {
        let mut t = base;
        for (i, j) in joints.iter().enumerate() {
            t = t * j.0 * DAffine3::from_axis_angle(axis_of(j), q[i]);
        }
        DMat4::from(t * end)
    }

    /// Directly exercise the generic 5R padding fallback over random non-DH 5R
    /// chains: plant a configuration, then require that the solve recovers it and
    /// that every returned solution reproduces the target pose.
    #[test]
    fn generic_5r_recovers_planted() {
        let mut s = 0xC0FFEE_u64;
        for _ in 0..120 {
            let joints: Vec<(DAffine3, deke_types::JointSpec<f64>)> = (0..5)
                .map(|_| {
                    (
                        raff(&mut s),
                        deke_types::JointSpec::Revolute {
                            axis_local: raxis(&mut s),
                        },
                    )
                })
                .collect();
            let base = DAffine3::IDENTITY;
            let end = raff(&mut s);
            let q: Vec<f64> = (0..5).map(|_| rr(&mut s, -2.5, 2.5)).collect();
            let target = fk_nr(&joints, base, end, &q);

            let sols = solve_generic_5r(&joints, base, end, &target);
            assert!(
                !sols.is_empty(),
                "5R fallback returned no solution for a reachable target"
            );
            let tc = target.to_cols_array();
            for sol in &sols {
                let got = fk_nr(&joints, base, end, &sol.as_slice()[..5]).to_cols_array();
                assert!(
                    got.iter().zip(tc.iter()).all(|(p, q)| (p - q).abs() < 1e-6),
                    "5R solution does not reach the target"
                );
            }
            let recovered = sols.iter().any(|sol| {
                (0..5).all(|i| {
                    let d = (sol.as_slice()[i] - q[i]).rem_euclid(std::f64::consts::TAU);
                    d < 1e-5 || (std::f64::consts::TAU - d) < 1e-5
                })
            });
            assert!(recovered, "5R fallback did not recover planted q={q:?}");
        }
    }

    fn puma_dh() -> Robot<6> {
        let pi = std::f64::consts::PI;
        let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
        let a = [0.0, 0.43180, -0.02032, 0.0, 0.0, 0.0];
        let d = [0.67183, 0.13970, 0.0, 0.43180, 0.0, 0.0565];
        Robot::<6>::from_dh(&alpha, &a, &d).expect("Puma 560 should construct")
    }

    fn q<const N: usize>(arr: [f64; N]) -> SRobotQ<N, f64> {
        SRobotQ::from_array(arr)
    }

    #[test]
    fn dh_construction() {
        let robot = puma_dh();
        assert!(robot.has_known_decomposition());
        assert_eq!(robot.n_joints(), 6);
        assert_eq!(robot.n_effective_joints(), 6);
        assert!(robot.is_spherical(), "Puma 560 has a spherical wrist");
    }

    #[test]
    fn fk_pose_bottom_row() {
        let robot = puma_dh();
        let pose = robot.fk(&q([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]));
        let cols = pose.to_cols_array_2d();
        assert!(cols[0][3].abs() < 1e-10);
        assert!(cols[1][3].abs() < 1e-10);
        assert!(cols[2][3].abs() < 1e-10);
        assert!((cols[3][3] - 1.0).abs() < 1e-10);
    }

    fn assert_roundtrip<const N: usize>(robot: &Robot<N>, q_in: SRobotQ<N, f64>) {
        let pose = robot.fk(&q_in);
        let solutions = robot.ik(pose).unwrap();
        assert!(!solutions.is_empty(), "no IK solutions for q={:?}", q_in.0);
        let pose_cols = pose.to_cols_array();
        let matched = solutions.iter().any(|sol| {
            let recon = robot.fk(sol).to_cols_array();
            pose_cols
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max)
                < 1e-6
        });
        assert!(
            matched,
            "no IK solution reconstructed pose for q={:?}",
            q_in.0
        );
    }

    #[test]
    fn ik_roundtrip_puma() {
        assert_roundtrip(&puma_dh(), q([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]));
    }

    #[test]
    fn hp_construction_2r() {
        let h = [DVec3::Z, DVec3::Z];
        let p = [DVec3::ZERO, DVec3::X, DVec3::X];
        let robot = Robot::<2>::from_hp(&h, &p).unwrap();
        assert_eq!(robot.n_joints(), 2);
        assert!(robot.has_known_decomposition());
    }

    #[test]
    fn kinematic_family_string() {
        let family = puma_dh().kinematic_family();
        assert!(family.contains("6R"), "family was {family}");
    }

    #[test]
    fn ur5_roundtrip() {
        let pi = std::f64::consts::PI;
        let alpha = [pi / 2.0, 0.0, 0.0, pi / 2.0, -pi / 2.0, 0.0];
        let a = [0.0, -0.612, -0.573, 0.0, 0.0, 0.0];
        let d = [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922];
        let robot = Robot::<6>::from_dh(&alpha, &a, &d).unwrap();
        assert!(robot.has_known_decomposition(), "UR5 should be solvable");
        assert_roundtrip(&robot, q([0.1, -1.0, 1.5, 0.5, -0.3, 0.2]));
        assert_roundtrip(&robot, q([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn irb_roundtrip() {
        let pi = std::f64::consts::PI;
        let alpha = [-pi / 2.0, 0.0, -pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
        let a = [0.320, 1.28, 0.20, 0.0, 0.0, 0.0];
        let d = [0.780, 0.0, 0.0, 1.142, 0.0, 0.2];
        let robot = Robot::<6>::from_dh(&alpha, &a, &d).unwrap();
        assert!(robot.has_known_decomposition(), "IRB-style should solve");
        assert_roundtrip(&robot, q([0.4, -0.5, 0.6, 0.2, -0.3, 0.4]));
    }

    #[test]
    fn panda_fixed_q7_roundtrip() {
        // 7-DOF Panda, q7 locked at 0 to reduce to 6 effective joints.
        let pi = std::f64::consts::PI;
        let alpha = [
            pi / 2.0,
            -pi / 2.0,
            -pi / 2.0,
            pi / 2.0,
            -pi / 2.0,
            pi / 2.0,
            0.0,
        ];
        let a = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088];
        let d = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107];
        let robot =
            Robot::<7>::from_dh_with(&alpha, &a, &d, DMat3::IDENTITY, &[FixedAxis::new(6, 0.0)])
                .unwrap();
        assert_eq!(robot.n_joints(), 7);
        assert_eq!(robot.n_effective_joints(), 6);

        let q_in = q([0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.0]);
        let pose = robot.fk(&q_in);
        let solutions = robot.ik(pose).unwrap();
        assert!(!solutions.is_empty());
        for sol in &solutions {
            // Locked joint must hold its value.
            assert!((sol.as_slice()[6] - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn hp_4r_roundtrip() {
        let ex = DVec3::X;
        let ey = DVec3::Y;
        let ez = DVec3::Z;
        let h = [ez, ey, ex, ey];
        let p = [ex, ex + ey, ey + ex, ex + ez, ex + ez];
        let robot = Robot::<4>::from_hp(&h, &p).unwrap();
        assert!(robot.has_known_decomposition());
        assert_roundtrip(&robot, q([0.3, 0.4, -0.5, 0.6]));
    }
}
