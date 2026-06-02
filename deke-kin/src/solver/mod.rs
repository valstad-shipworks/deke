//! IK solver dispatch — one specialised solver per effective DOF (1..=6).

use deke_types::SRobotQ;
use glam::{DMat4, DVec3};

mod r1;
mod r2;
mod r3;
mod r4;
mod r5;
mod r6;

/// Const-generic kinematic chain for the IK hot path.
///
/// `N` is the number of joints, `M = N + 1` is the number of P offsets.
/// Stored as fixed-size `[DVec3; N]` / `[DVec3; M]` arrays so the per-IK
/// workload never touches the heap.
///
/// Stable Rust can't yet express `M = N + 1` in the type, so callers must
/// provide both as separate const params (`Chain<6, 7>`, `Chain<5, 6>`, …).
/// The `from_slices` constructor enforces the dimensional invariant at runtime.
#[derive(Clone, Debug)]
pub struct Chain<const N: usize, const M: usize> {
    pub h: [DVec3; N],
    pub p: [DVec3; M],
}

impl<const N: usize, const M: usize> Chain<N, M> {
    pub fn from_slices(h: &[DVec3], p: &[DVec3]) -> Self {
        debug_assert_eq!(h.len(), N);
        debug_assert_eq!(p.len(), M);
        debug_assert_eq!(M, N + 1);
        let mut h_arr = [DVec3::ZERO; N];
        let mut p_arr = [DVec3::ZERO; M];
        h_arr.copy_from_slice(h);
        p_arr.copy_from_slice(p);
        Self { h: h_arr, p: p_arr }
    }
}

pub type Chain1 = Chain<1, 2>;
pub type Chain2 = Chain<2, 3>;
pub type Chain3 = Chain<3, 4>;
pub type Chain4 = Chain<4, 5>;
pub type Chain5 = Chain<5, 6>;
pub type Chain6 = Chain<6, 7>;

/// Reduced-chain joint solution, padded to 6 slots so the per-DOF solver
/// variants share a single transport type through the [`Solver`] enum.
/// Only the first `n_effective_joints` entries are meaningful; the rest
/// are zero-padding. The outer [`crate::Robot::ik`] knows the effective DOF
/// and slices off the padding before re-inserting any locked axes.
pub(crate) type Joints = SRobotQ<6, f64>;

/// Build a [`Joints`] from a slice of up to 6 raw joint values, padding the
/// remaining slots with `0.0`.
#[inline]
pub(crate) fn pack(arr: &[f64]) -> Joints {
    debug_assert!(arr.len() <= 6);
    let mut buf = [0.0f64; 6];
    buf[..arr.len()].copy_from_slice(arr);
    SRobotQ::from_array(buf)
}

pub enum Solver {
    R1(r1::R1),
    R2(r2::R2),
    R3(r3::R3),
    R4(r4::R4),
    R5(r5::R5),
    R6(r6::R6),
}

impl Solver {
    pub fn build(h: &[DVec3], p: &[DVec3], zero_thresh: f64, axis_intersect_thresh: f64) -> Self {
        match h.len() {
            1 => Solver::R1(r1::R1::new(h, p)),
            2 => Solver::R2(r2::R2::new(h, p, zero_thresh)),
            3 => Solver::R3(r3::R3::new(h, p, zero_thresh)),
            4 => Solver::R4(r4::R4::new(h, p, zero_thresh, axis_intersect_thresh)),
            5 => Solver::R5(r5::R5::new(h, p, zero_thresh, axis_intersect_thresh)),
            6 => Solver::R6(r6::R6::new(h, p, zero_thresh, axis_intersect_thresh)),
            n => panic!("Solver::build called with unsupported DOF {n}"),
        }
    }

    pub fn has_known_decomposition(&self) -> bool {
        match self {
            Solver::R1(_) | Solver::R2(_) | Solver::R3(_) => true,
            Solver::R4(s) => s.has_known_decomposition(),
            Solver::R5(s) => s.has_known_decomposition(),
            Solver::R6(s) => s.has_known_decomposition(),
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_spherical(&self) -> bool {
        match self {
            Solver::R6(s) => s.is_spherical(),
            _ => false,
        }
    }

    pub fn kinematic_family(&self) -> String {
        match self {
            Solver::R1(_) => "1R".into(),
            Solver::R2(_) => "2R".into(),
            Solver::R3(_) => "3R".into(),
            Solver::R4(s) => s.kinematic_family(),
            Solver::R5(s) => s.kinematic_family(),
            Solver::R6(s) => s.kinematic_family(),
        }
    }

    pub fn solve(&self, pose: &DMat4) -> Vec<Joints> {
        match self {
            Solver::R1(s) => s.solve(pose),
            Solver::R2(s) => s.solve(pose),
            Solver::R3(s) => s.solve(pose),
            Solver::R4(s) => s.solve(pose),
            Solver::R5(s) => s.solve(pose),
            Solver::R6(s) => s.solve(pose),
        }
    }
}
