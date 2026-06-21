# deke

Motion-planning crates for serial manipulators: forward and inverse kinematics,
collision validation, sampling-based planning, and a family of time-optimal
path-parameterization retimers.

Two principles run through every crate:

- **Pluggability first.** Each stage is a trait — `FKChain`, `IkSolver`,
  `Validator`, `Planner`, `Retimer` — generic over the joint count `N` (a const
  generic) and the scalar `F` (`f32` or `f64`). Any implementation drops in for
  any other; validators compose with `&`/`|`/`!`, and `BoxFK` type-erases a
  chain when you need a trait object.
- **Performance throughout.** Const-generic `N` keeps joint configs, FK frames,
  and IK solutions in stack arrays. Leveraging SIMD where possible, and always
  reaching towards performance first algorithms.

## Kinematics

[`deke_kin::Kinematics`] is a single API surface for both FK and IK. Build it
from Denavit–Hartenberg, Hayati–Paul, or URDF parameters, or directly from a
[`KinSpec`] with arbitrary joint axes:

```rust
use deke_kin::{Kinematics, DHJoint, JointLimits};
use deke_kin::deke_types::{FKChain, IkSolver, SRobotQ};

let robot = Kinematics::<6, f64>::from_dh(dh_joints, JointLimits::symmetric(PI), &[]);

let q = SRobotQ::<6, f64>::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
let pose = robot.fk_end(&q)?;          // forward kinematics
let outcome = robot.ik(pose)?;         // IkOutcome::Solved(..) holds every isolated solution
```

**Forward kinematics** comes off the `FKChain` trait: `fk` for per-link frames,
`fk_end` for the end-effector, `all_fk` for base + links + end in one pass, plus
a geometric Jacobian derived from the chain structure.

**Inverse kinematics** resolves its strategy *once* at construction and caches
the verdict ([`ik_diagnostic`]), so every `ik` call takes the fastest viable
path:

1. a closed-form subproblem decomposition (analytical, 1R–6R), else
2. the general Raghavan–Roth / Manocha–Canny **eigenvalue solver** for an
   all-revolute 6-DOF chain — every isolated solution, no seed, no iteration
   ([`rr_ik`]), else
3. a **rule-reduced** solve for over-actuated chains, else
4. *not viable* — FK still works; `ik` returns `IkNotViable`.

`IkRules` make >6-DOF and linear-axis chains solvable by folding them down to a
≤6R sub-problem: `FixedAxis` pins a joint, `DiscreteAxis` sweeps one across its
limits (one solve per sample — ideal for a linear rail), `IncludeWrapped` emits
±2π variants that stay in limits. Every returned solution is filtered to the
joint limits.

## Crates

| Crate | Purpose |
|-------|---------|
| `deke-types` | Shared types: joint configs, paths, trajectories, FK chains, validators, and the `Validator` / `Planner` / `Retimer` interfaces. |
| `deke-kin` | Forward and inverse kinematics for serial manipulators: analytical 1R–6R, general-6R eigenvalue solver, and rule-reduced over-actuated chains. |
| `deke-wreck` | Collision validation backed by the `wreck` library. |
| `deke-rrt` | RRT-Connect, AORRTC, and KRRTC planners over the validator interface. |
| `deke-multipath` | Optimal ordering and orientation of required paths (asymmetric generalized TSP), stitched with planned or straight-line connectors. |
| `deke-topp3tcp-nlp` | Continuous- and discrete-NLP TOPP-3TCP retimers. |
| `deke-topp3tcp-spline` | B-spline path representation with depth-first search over jerk candidates. |
| `deke-topp-speed` | Real-time jerk-limited shaper plus a live goal-tracking pursuer. |
| `deke-linear` | Constant-TCP-speed Cartesian polyline following: conditions a polyline, branch-tracks IK around singularities, and holds commanded speed within joint limits. |
| `deke-bench-retimers` | Comparative harness across the retimer family. Internal, unpublished. |

## Pipeline

Plan a joint-space path with `deke-rrt` against a `Validator` (typically
`deke-wreck`), then retime it to a sampled trajectory with one of the `deke-topp`
crates. `deke-kin` supplies FK/IK; `deke-types` defines the interfaces every
stage shares — swap any implementation without touching the others.

```rust
use deke_types::{Planner, Retimer};
use deke_rrt::StartEnd;

let (path, _diag) = planner.plan(&config, &StartEnd::new(start, goal)?, &validator, &ctx);  // deke-rrt
let path = path?;
// The retimer is constructed with the FK chain (e.g. `Topp3Tcp6::new(&chain)`).
let (traj, _diag) = retimer.retime(&constraints, &path, &validator, &ctx); // deke-topp*
```

`plan` and `retime` each return the result paired with a diagnostic.

The `valuable` feature, where present, derives `valuable::Valuable` for
structured logging.

## License

Each crate carries its own `LICENSE`. All are Apache-2.0 except `deke-kin`.
Vendored third-party licenses are collected under
[`vendored_license/`](vendored_license/); see [CITATIONS.md](CITATIONS.md) for
upstream provenance.
