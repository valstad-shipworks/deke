# deke-linear

Constant-TCP-speed Cartesian polyline following for serial manipulators.

Where the `deke-topp*` retimers are **time-optimal** (maximise speed under v/a/j
caps), `deke-linear` holds a **constant TCP travel speed** — the requirement for
welding and similar process motions — and degrades gracefully near singularities
rather than failing. It is a CNC-style constant-feedrate interpolator in three
stages, the latter two of which implement the `deke_types` `Planner` / `Retimer`
traits so they compose with the rest of the ecosystem.

## Pipeline

```
&[DAffine3] poses + FollowConfig
   │
 [A] path::condition   → Vec<CartesianRun>      split at sharp corners; shallow
   │                                            corners smoothed (squiggle Catmull–Rom),
   │                                            arc-length parameterised
 [B] CartesianLinearPlanner : Planner           dense IK → DP branch track
   │                                            (analytic IK, manipulability-weighted,
   │                                            no Jacobian inversion → singularity-safe)
 [C] ConstantSpeedRetimer  : Retimer            constant feedrate held where the joint
   │                                            v/a/j MVC allows; smooth dips elsewhere
 LinearFollower::follow → SRobotTraj            per-run trajectories stitched at rest
```

- **Corners (hybrid).** A vertex whose turn angle exceeds
  `PathConditioning::sharp_corner_angle` is *sharp*: the path splits there into runs
  that start and stop at rest, keeping the vertex exact. Shallower corners are
  smoothed by the Catmull–Rom spline and traversed without stopping.
- **`forbid_interior_dips`.** By default the speed dips smoothly where the joint
  v/a/j geometry forces it (a shallow corner, a near-singular patch). Set this flag
  to require the commanded speed to hold flat through a run's interior — the only
  sub-commanded speed allowed is the rest ramp at each run's start and end. If a dip
  would be forced anywhere in the interior, the retime fails with
  `SpeedDipRequired { run, s, feasible_speed, commanded }` instead of slowing down.
- **Singularity tolerance.** The planner inverts each pose with the chain's analytic
  IK (every branch, limit-filtered, no Jacobian inversion) and routes a continuous
  joint track that maximises manipulability `√det(J·Jᵀ)`. The retimer's
  feasible-speed ceiling is `min_j v_max,j / |q'_j(s)|`, so as a singularity is
  approached the commanded speed dips smoothly to zero instead of demanding infinite
  joint speed.

## Example

```rust
use deke_linear::{LinearFollower, FollowConfig};
use deke_types::glam::DAffine3;

let follower = LinearFollower::new(&chain);            // any ContinuousFKChain + IkSolver
let (traj, diag) = follower.follow(&poses, &cfg)?;     // poses: &[DAffine3], cfg: FollowConfig<N>
```

## Free tool-axis yaw (functional redundancy)

A tool that is rotationally symmetric about one of its axes — a welding torch,
spray head — leaves the rotation about that axis free (a 5-DOF task on a 6-DOF
arm). Declaring it lets the planner spend that DOF to steer around singularities.

```rust
use deke_linear::{FollowConfig, JointLimits, RedundantAxis, RedundantOptions};
use std::time::Duration;

let cfg = FollowConfig::weld(35.0, joint_limits, Duration::from_millis(8))  // 35 in/min
    .with_redundancy(RedundantOptions {
        axis: RedundantAxis::PosZ,                 // ±X/±Y/±Z or Custom(unit vec), tool frame
        yaw_window: (-45f64.to_radians(), 45f64.to_radians()),
        ..RedundantOptions::default()
    });
```

Yaw is a smooth scalar, so it is gridded coarsely and resolved by a **single global
DP** over `(station) × (yaw × branch)` — exact, so it finds the globally optimal yaw
track in one pass. A manipulability node cost steers off singularities, a yaw-rate
edge penalty keeps the spin smooth, and the velocity reconfiguration test rejects
discontinuous edges; the yaw may sweep freely across the window (bounded only by
per-step `max_yaw_step`). That coarse `ψ(s)` schedule is then refined at fine
arc-length spacing, with analytic IK placing the arm exactly each step
(predictor–corrector). The free axis is configurable because tool frames differ
(this project is Z-forward; others are X-forward).

## Reconfiguration by joint velocity

The planner takes a `max_velocity` and per-joint velocity ceilings: any edge that
would drive **any** joint past a configurable fraction (default 90%) of its limit
*at that speed* is treated as a reconfiguration/discontinuity and rejected. At weld
speeds (20–50 IPM) a joint approaching its velocity limit is the signature of a
singularity or wrist flip, so this cleanly separates "needs to slow down" from
"can't be done continuously." `FollowConfig::weld()` wires it up; otherwise set
`PlannerOptions::{max_velocity, joint_v_max, reconfig_vel_fraction}`.

## Linear rail (7th external axis)

A 6-DOF arm mounted on a prismatic linear rail can resolve the rail position as a
second smooth redundant DOF, the same way the tool yaw is resolved. The output
widens to a **rail-first** `SRobotQ<7>` = `[x_rail, q1..q6]` that flows through the
same `ConstantSpeedRetimer` unchanged.

```rust
use deke_linear::{
    ConstantSpeedRetimer, PlannerOptions, RailAxis, RailConfig, RailLinearPlanner,
    RailMountedChain, RailOptions, RailRefine,
};
use deke_types::{Planner, Retimer};

// `arm` is any ContinuousFKChain<6> + IkSolver<6> (the unmodified 6-DOF arm).
let planner = RailLinearPlanner::<6, 7, _>::new(&arm);
let chain = RailMountedChain::<6, 7, _>::new(&arm, RailAxis::PosX); // 7-DOF FK for the retimer
let retimer = ConstantSpeedRetimer::new(&chain);

let cfg = RailConfig::<6, 7> {
    planner: PlannerOptions {
        // 7-wide branch-tracking knobs; put the rail's velocity ceiling in slot 0
        // of `joint_v_max` for the reconfiguration test.
        ..PlannerOptions::default()
    },
    rail: RailOptions {
        axis: RailAxis::PosX,             // world-frame rail axis (PosX/PosY/PosZ/Custom)
        window: (-0.5, 0.5),              // rail travel limits (m)
        samples: 21,                      // rail grid resolution for the DP
        refine: RailRefine::Pchip,        // monotone (Fritsch–Carlson) rail schedule, no overshoot
        ..RailOptions::default()
    },
};
// for run in condition(&poses, &cond)? { planner.plan(&cfg, &run, ...); retimer.retime(&cons7, ...); }
```

- **Base-shift IK, inverse-free.** The rail is a world translation of the arm base:
  the target is shifted by `−x·â` and solved by the unchanged 6-DOF analytic IK
  (every branch, limit-filtered). `RailMountedChain` is a 7-DOF `ContinuousFKChain`
  (FK / Jacobian / manipulability) but deliberately does **not** implement
  `IkSolver<7>` — the redundancy is resolved in the planner, not by an
  underdetermined 7-DOF solve.
- **Arm-conditioned.** The DP scores the *arm's* manipulability so the rail is
  spent to keep the arm off its singular sets; a 7-DOF measure would let the rail's
  prismatic Jacobian column mask an arm singularity and the rail would never be
  recruited.
- **Smooth schedule.** The coarse rail track is refined with a monotone PCHIP
  (`RailRefine::Pchip`, the default) so the slow heavy axis never overshoots — fed
  to the retimer as dense knots, never differentiated for jerk.
- **Rail + yaw.** `RailYawPlanner` composes the rail with the free tool yaw
  hierarchically (rail first, then yaw on the fixed rail schedule).
- **Same guarantees.** The rail's v/a/j ceilings sit in slot 0 of `JointLimits<7>`;
  `is_reconfiguration` and the retimer's finite-difference verify already iterate
  every joint, so the rail is bounded by construction like any other axis.

## Scope / notes

- **Orientation** is a full TCP pose per vertex (slerped along the path). For a
  symmetric tool, declare the free axis via `with_redundancy` (see above) to let the
  planner resolve the yaw.
- Joint (and TCP) **velocity/acceleration/jerk** are honoured as bounds on the
  *finite differences* of the emitted samples — exactly what the controller
  reconstructs — by the discrete-LP retimer, with a verify-and-Err backstop against
  the true limits. They hold by construction rather than by a continuous-derivative
  approximation.
- Geometry runs in `f32` (squiggle); kinematics and timing in `f64`.

## License

Apache-2.0.
