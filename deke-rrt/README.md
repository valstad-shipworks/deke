# deke-rrt

Sampling-based planners over the `deke_types::Validator` interface. Produces a
joint-space `SRobotPath` between a start and goal configuration.

## Planners

Each planner is a marker struct implementing `Planner<N, f64>`, paired with a
settings struct as its `Config`:

- `RrtcPlanner` / `RrtcSettings`: RRT-Connect (bidirectional).
- `AorrtcPlanner` / `AorrtcSettings`: asymptotically optimal RRT-Connect with
  anytime shortcutting and a geometric lower-bound stop.
- `KrrtcPlanner` / `KrrtcSettings`: kinodynamic variant.

Settings configure range, connection radius, iteration and sample budgets,
dynamic-domain sampling, and the randomizer (`HaltonRand`, `DekeRand`).
Termination is reported via `RrtTermination` with per-run counters for
diagnosing failures.

## Example

```rust
use deke_rrt::{AorrtcPlanner, AorrtcSettings, StartEnd};
use deke_types::Planner;

let config = AorrtcSettings { ..Default::default() };
let waypoints = StartEnd::new(start, goal)?;
let (path, diag) = AorrtcPlanner::default().plan(&config, &waypoints, &validator, &ctx);
let path = path?;
```

S-curve helpers (`scurve` module: `JointKinLimits`, `KinematicLimits`,
`direction_cosine`) support kinodynamic steering.

## Features

- `valuable`: derive `valuable::Valuable` for structured logging.

## License

Apache-2.0
