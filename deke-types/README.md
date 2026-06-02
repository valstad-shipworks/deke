# deke-types

Shared types and interfaces for the deke crates. Joint count `N` is a const
generic; the scalar `F` is generic over `f32`/`f64` via `KinScalar`.

## Surface

- `RobotQ` / `SRobotQ`: joint configurations (heap and stack-array forms).
- `RobotPath` / `SRobotPath`: ordered joint-space waypoints.
- `RobotTraj` / `SRobotTraj`: time-sampled trajectories.
- `FKChain`, `StrictFKChain`, `BoxFK`: forward-kinematics chains; `JointSpec`,
  `KinSpec` describe them.
- `Validator` and combinators (`ValidatorAnd`, `ValidatorOr`, `ValidatorNot`,
  `JointValidator`, `DynamicJointValidator`).
- `Planner`, `Retimer`: the interfaces implemented by `deke-rrt` and the
  `deke-topp` crates.
- `DekeError` / `DekeResult`.

## Example

```rust
use deke_types::{JointValidator, SRobotQ, Validator};

let lower = SRobotQ::<6, f64>::from_array([-3.14; 6]);
let upper = SRobotQ::<6, f64>::from_array([3.14; 6]);
let validator = JointValidator::new(lower, upper);

let q = SRobotQ::<6, f64>::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
let ret = validator.validate(&q, &ctx);   // DekeResult<R> per the Validator impl
```

`Planner::plan` and `Retimer::retime` each return their result paired with a
diagnostic; `Validator::validate` / `validate_motion` return `DekeResult<R>`.

## Features

- `valuable`: derive `valuable::Valuable` for structured logging.

## License

Apache-2.0
