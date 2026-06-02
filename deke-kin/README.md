# deke-kin

Pure-Rust inverse kinematics for serial manipulators. Analytical FK/IK for 1R–6R
chains via subproblem decomposition, with a complete general-6R eigenvalue
fallback for chains that have no closed form.

[`Kinematics`] is the single API surface. Build it from Denavit-Hartenberg,
Hayati-Paul, or URDF parameters, or from a `KinSpec`; it implements both forward
kinematics (`FKChain`) and inverse kinematics (`IkSolver`). Joint count `N` is a
const generic; configurations are `SRobotQ<N, F>` (`F` is `f32` or `f64`).

At construction each chain eagerly resolves how it can be inverted and caches the
verdict (`ik_diagnostic()`):

- a known closed-form decomposition (analytical, 1–6R), else
- the general Raghavan–Roth / Manocha–Canny eigenvalue solver for an
  all-revolute 6-DOF chain (`rr_ik`), else
- **not viable** — e.g. a chain with a prismatic joint, for which every `ik`
  call returns `DekeError::IkNotViable`.

## Example

```rust
use deke_kin::{Kinematics, DHJoint};
use deke_types::SRobotQ;
use deke_types::fk::{FKChain, IkSolver};

let pi = std::f64::consts::PI;
// Puma 560
let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
let a = [0.0, 0.4318, -0.0203, 0.0, 0.0, 0.0];
let d = [0.6718, 0.1397, 0.0, 0.4318, 0.0, 0.0565];
let robot = Kinematics::<6, f64>::from_dh(std::array::from_fn(|i| DHJoint {
    a: a[i], alpha: alpha[i], d: d[i], theta_offset: 0.0,
}));

println!("{:?}", robot.ik_diagnostic().strategy); // Analytic { family: "6R …" }

let q = SRobotQ::<6, f64>::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
let target = robot.fk_end(&q)?;
let solutions = robot.ik(target)?.unwrap();
```

## Layout

- `Kinematics`: the FK/IK chain; constructors `from_dh` / `from_hp` /
  `from_urdf` / `from_kinspec`.
- `IkSolverDiagnostic` / `IkStrategy`: the resolved IK strategy.
- `rr_ik`: the general 6R Raghavan–Roth/Manocha–Canny eigenvalue solver
  (`solve_kinspec`), used as the fallback and available directly.
- `solver::r1`..`r6`: the analytical per-DOF subproblem decompositions.
- `ik_geo`: vendored Paden–Kahan subproblem kernels behind the analytical path.
- `snap`: nearest-solution selection.

## License

MIT. Vendors a fork of `ik-geo` (see `src/ik_geo/LICENSE`). See the root
[CITATIONS.md](../CITATIONS.md).

[`Kinematics`]: https://docs.rs/deke-kin
