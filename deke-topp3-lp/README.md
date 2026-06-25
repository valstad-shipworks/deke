# deke-topp3-lp

Joint-space, path-exact, jerk-limited TOPP-3 retimer via a discrete convex LP.
Implements `deke_types::Retimer<N, f64>`.

Given an untimed joint-space polyline (`SRobotPath`) and per-axis
velocity/acceleration/jerk limits, it emits a uniform-`dt` trajectory whose
samples lie *exactly* on the input chord and whose backward finite differences —
the quantities a downstream controller reconstructs — stay under the limits.

## How it works

The timing is a discrete convex program. The single scalar decision variable
`σ[k]` is the arc length reached at output tick `k·dt`. Because the path is
chord-linear, the `m`-th finite difference of every joint is exactly
`secantᵦ · Δᵐσ` within a segment, so each per-joint v/a/j limit becomes a
*linear* bound on a difference of the `σ`s — including jerk, which is what breaks
convexity in free-time TOPP. The program maximises progress (so it runs at the
limits wherever it can) and is solved with a small banded LP (Clarabel).

- **Zero chord deviation.** Every output sample is reconstructed as a point on
  the input polyline, and the endpoints are pinned bit-exact to the first/last
  waypoints. Unlike a smoothing retimer, the geometry is never rounded.
- **Hard finite-difference bound.** A final check recomputes the output's
  backward FD against the *true* limits; `retime` returns `Ok` only if it passes
  (verify-and-Err), so an over-limit trajectory is never emitted — a path that
  cannot be timed under the limits fails instead.
- **Sharp corners.** A hard joint-space kink can only be traversed on-chord with
  bounded jerk by stopping on it, so a vertex whose turn angle exceeds
  `sharp_corner_angle` is split into separate rest-to-rest runs that stop exactly
  on the corner. Shallower corners the LP can dip through stay in one run.

The exact per-joint rows fire only where a finite-difference stencil straddles a
corner; in segment interiors a single scalar row (equivalent, since the secant is
constant) keeps the program small.

## Retimers

- `Topp3Lp<N>` — joint-only; needs no FK. Rejects a TCP cap.
- `Topp3LpTcp<'a, N, FK>` — adds a Cartesian TCP **linear-velocity** cap. The cap
  becomes a per-segment ceiling on `σ̇` from the FK Jacobian (`‖J_lin·secant‖`),
  and the realised FK speed is verified against the true cap. Works on rail-mounted
  chains — a prismatic axis is just another DOF with its own limits, and its
  Jacobian column drives the TCP cap.

## Example

```rust
use deke_topp3_lp::{Topp3Lp, Topp3LpConstraints};
use deke_types::Retimer;

let retimer = Topp3Lp::<6>::new();
let constraints =
    Topp3LpConstraints::symmetric(2.0, 8.0, 80.0, std::time::Duration::from_millis(8));
let (traj, _diag) = retimer.retime(&constraints, &path, &validator, &());
let traj = traj?;
```

With a TCP velocity cap (needs an FK chain):

```rust
use deke_topp3_lp::{Topp3LpConstraints, Topp3LpTcp};
use deke_types::Retimer;

let constraints = Topp3LpConstraints::symmetric(2.0, 8.0, 80.0, dt).with_tcp_speed(0.25);
let (traj, diag) = Topp3LpTcp::new(&chain).retime(&constraints, &path, &validator, &());
```

`Topp3LpDiagnostic` reports the output sample count, duration, joint-space arc
length, and the peak per-joint v/a/j and TCP speed.

## Limitations

- Constant `dt`; the discrete LP is ill-conditioned at extreme output rates
  (≳ few kHz). Targets ≤ ~1 kHz today.
- TCP cap is velocity-only (no TCP acceleration/jerk).
- The "minimum time" objective is a max-progress heuristic on a fixed grid, not a
  true minimum-time bisection.

## License

Apache-2.0. See [CITATIONS.md](../CITATIONS.md) for the convex-TOPP / TOPP3
provenance.
