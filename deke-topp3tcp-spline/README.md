# deke-topp3tcp-spline

Discrete TOPP-3TCP retimer using a B-spline path and a depth-first search over
discrete jerk candidates. Implements `deke_types::Retimer<N, f64>`.

The path is a clamped B-spline interpolated through the input waypoints, with
support-point density refined until the spline stays inside a configurable
deviation tube around the original polyline. The search state is the
path-parameter tuple `(s, s', s'', s''')` at uniform time steps `dt`. At each
step it enumerates a cosine-spaced fan of jerk candidates, prunes any that
violate per-axis joint v/a/j or Cartesian TCP v/a/j limits (via the position
rows of the geometric Jacobian), and closes with a three-segment
boundary-condition solve once `s >= 0.7`.

## Example

```rust
use deke_topp3tcp_spline::{Topp3TcpSpline, Topp3TcpSplineConstraints};
use deke_types::Retimer;

let constraints = Topp3TcpSplineConstraints::symmetric(/* joint + TCP v/a/j */);
// adjust constraints.search and constraints.spline as needed
let (traj, diag) =
    Topp3TcpSpline::new(&chain).retime(&constraints, &path, &validator, &ctx);
let traj = traj?;
```

`BSpline` and `SplineInterpolatedRobotPath` expose the path representation;
`Topp3TcpSplineDiagnostic` reports the solve status.

## License

See workspace.
