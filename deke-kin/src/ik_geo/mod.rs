// Vendored fork: a few matrix helpers and subproblem variants are retained for
// fidelity to upstream even though reaik's solvers don't call all of them.
#![allow(dead_code)]

//! Vendored fork of the `ik-geo` crate (v0.1.2, MIT — see LICENSE in this
//! directory), slimmed to the analytical [`subproblems`] kernels (sp1..6),
//! the small [`solutionset`] enum used for their return types, and the
//! rectangular matrix helpers in [`math`] they need. These are the Paden–Kahan
//! primitives behind reaik's own 6R dispatch in `solver/r1..r6`; the vendored
//! numerical `robot`/`inverse_kinematics` layer has been removed in favour of
//! the in-house Raghavan–Roth solver in [`crate::rr_ik`].

pub mod math;
pub mod solutionset;
pub mod subproblems;
