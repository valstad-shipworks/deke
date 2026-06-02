# deke-bench-retimers

Comparative harness across the four `deke_types::Retimer` implementations:

- `deke_topp3tcp_nlp::continuous::Topp3Tcp6` (continuous NLP)
- `deke_topp3tcp_nlp::discrete::Topp3Tcp6Discrete` (discrete NLP)
- `deke_topp3tcp_spline::Topp3TcpSpline` (B-spline + DFS over jerk)
- `deke_topp_speed::ToppSolver` (real-time jerk-limited shaper)

A `BenchProblem` bundles a joint-space waypoint path with per-joint v/a/j limits
and a TCP velocity limit. `run_all` runs each retimer against a problem and
returns a `BenchResult` per retimer. Metrics: solve wall time, trajectory
duration, joint v/a/j peaks (via the same backward-difference stencils a consumer
would apply to the output), TCP linear-velocity peak (forward difference on
FK-evaluated positions), and per-limit utilization (peak over limit).

All problems target a fixed production 6-DOF URDF chain so the comparison runs
against one FK and one set of ceilings.

Analysis only, unpublished. Run the sweep through the test binary:

```sh
cargo test -p deke-bench-retimers --release -- --nocapture
```

## License

Unpublished (`publish = false`).
