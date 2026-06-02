# deke-topp-speed

Real-time, jerk-limited trajectory shaping for joint paths. Produces
time-optimal trajectories respecting velocity, acceleration, and jerk ceilings,
optionally through intermediate waypoints, plus a live follower for moving goals.

Joint count `N` is a const generic; numerics are generic over `F: KinScalar`
(`f32` or `f64`).

## Surface

- `ToppSolver`: offline path-to-trajectory solver. Implements
  `deke_types::Retimer`.
- `Pursuer`: real-time tracker that adapts the goal each control cycle to follow
  a moving `PursuitTarget`.
- `MotionSpec`: kinematic limits and goal description for a solve.
- `MotionSample`, `Extent`, `StepStatus`, and the mode enums (`ControlMode`,
  `Coordination`, `DurationGrid`, `FollowMode`, `GoalOutOfBounds`).

## Example

`ToppSolver` implements `Retimer` with `MotionSpec` as its constraints type:

```rust
use deke_topp_speed::{MotionSpec, ToppSolver};
use deke_types::Retimer;
use std::time::Duration;

let solver = ToppSolver::new(Duration::from_millis(8), &chain);   // output dt + FK chain
let spec = MotionSpec { /* per-axis v/a/j limits, goal, modes */ };
let (traj, diag) = solver.retime(&spec, &path, &validator, &ctx);
let traj = traj?;
```

For live tracking, `Pursuer` consumes a `PursuitTarget` each control cycle and
adapts the goal, returning a `MotionSample` and `StepStatus`.

## License

See workspace.
