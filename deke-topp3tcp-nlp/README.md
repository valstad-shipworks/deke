# deke-topp3tcp-nlp

## Overview

Time-optimal path-parameterization (TOPP-3TCP) retimers for N-DOF robot arms. Both retimers take
a fixed geometric path of joint-space waypoints and emit a uniformly time-sampled `SRobotTraj`
that minimises total traversal time subject to per-joint and per-TCP velocity, acceleration, and
jerk bounds plus start/end boundary conditions. They share the path-densification and
boundary-projection plumbing in the `common` module and differ only in NLP formulation, solved by
the `hafgufa` (sleipnir) interior-point solver.

The entry types `Topp3Tcp6` and `Topp3Tcp6Discrete` are re-exported at the crate root. Types whose
names collide between the two formulations (`JointLimits`, `TcpLimits`, `SolveStatus`, the
diagnostic sub-types) are reached through the `continuous` and `discrete` modules.

## Continuous

The continuous formulation optimises the `(sd, sdd, sddd, dt)` profile at each densified knot and
resamples the converged solution onto the output grid. A two-stage warm start solves the
TCP-disabled problem first to seed the TCP-enabled solve.

```rust
use deke_topp3tcp_nlp::continuous::{Topp3Tcp6, Topp3Tcp6Constraints, TcpLimits};
use deke_types::Retimer;

let mut constraints = Topp3Tcp6Constraints::symmetric(2.0, 8.0, 80.0);
constraints.tcp = Some(TcpLimits { v_max: 1.0, a_max: 5.0, j_max: 50.0 });

let (traj, diag) = Topp3Tcp6::new(&chain).retime(&constraints, &path, &validator, &ctx);
```

## Discrete

The discrete formulation optimises the per-sample arc-length values `σ[i]` directly. Each
kinematic-bound row is a backward-difference of chord-linear joint positions over the σ chain —
the same finite difference the consumer computes on the output — so the IPM enforces exactly what
the consumer measures. A K-bisection driver searches for the smallest feasible sample count, and
the strict verifier in `discrete::verify` confirms the output to within solver tolerance.

```rust
use deke_topp3tcp_nlp::discrete::{Topp3Tcp6Discrete, Topp3Tcp6DiscreteConstraints, TcpLimits};
use deke_types::Retimer;

let mut constraints = Topp3Tcp6DiscreteConstraints::symmetric(2.0, 8.0, 80.0);
constraints.tcp = Some(TcpLimits { v_max: 1.0, a_max: 5.0, j_max: 50.0 });

let (traj, diag) = Topp3Tcp6Discrete::new(&chain).retime(&constraints, &path, &validator, &ctx);
```

## Notes

- Output is a uniformly time-sampled `SRobotTraj` at `1 / sample_rate_hz` spacing.
- The first `locked_prefix` joints can be held at their starting value for mobile bases or locked
  base rails.
- The `valuable` feature adds `valuable::Valuable` impls for the diagnostic and constraint types.
