# Rail-axis redundancy (7th-axis planning) — design note

Status: **planned, not started.** Scope captured here so we can resume cold.

Target: a 6-DOF arm mounted on a linear rail (prismatic, along world X). We want
`deke-linear` to plan the rail position *together with* the arm, resolved on a
coarse grid and refined smoothly, so the rail never injects discontinuities into
the constant-speed retimer.

This note covers the **rail-only** step (one redundant DOF). Combining it with the
existing tool-yaw redundancy is a separate, riskier follow-up — see *Risks → Why
not combinable yet*.

---

## What it is

The rail is a **smooth scalar redundant DOF**, structurally identical to the
tool-yaw DOF the crate already resolves in `src/redundant.rs`. The plan is to add a
second redundancy *kind* — a base/prismatic sweep — that reuses ~90% of the
existing `RedundantLinearPlanner` machinery (coarse global DP → smooth refinement).

The output joint vector widens from 6 to **7**: `q = [x_rail, q1..q6]` (final order
TBD — see *Open questions*). That 7-vector is what flows to the validator, the
manipulability cost, and the `ConstantSpeedRetimer`.

The only structural difference from yaw is **where the DOF enters the IK target**:

- **Yaw** post-multiplies the orientation: `R_ref(s) · Rot(â, ψ)`, translation
  unchanged (`redundant.rs:172`, `:229`).
- **Rail** shifts the arm base. For a fixed world TCP pose `T_world(s)` and rail
  position `x`, the pose seen by the existing 6-DOF analytic IK is

  ```
  T_arm(s, x) = (Trans(x · x̂) · base0)⁻¹ · T_world(s)
  ```

  i.e. translate the target by `−x` along the rail, then call the unchanged 6-DOF
  `ik`. Every branch, limit-filtered, no Jacobian inversion → still
  singularity-safe.

Everything else carries over: sweep `x ∈ [rail_min, rail_max]` instead of `ψ`, a
`rail_rate_weight` edge penalty in place of `yaw_rate_weight`, a `max_rail_step`
cap in place of `max_yaw_step`, manipulability node cost unchanged.

---

## What it solves

1. **Workspace extension along the weld.** Long seams that exceed the arm's reach
   become feasible because the base travels with the TCP.
2. **Singularity / reach relief.** The rail is a second way (besides tool yaw) to
   keep the arm well-conditioned. Crucially, when manipulability is scored on the
   **7-DOF** chain, the rail's prismatic Jacobian column raises `det(J Jᵀ)`, so the
   DP is naturally rewarded for using the rail to escape arm singularities — the
   6-DOF-arm-only manipulability cannot see this.
3. **No retimer discontinuities without dense sampling.** This is the whole point.
   The existing two-tier structure already does exactly what's needed:
   - **Coarse DP** over `(station) × (rail_sample × IK_branch)` on a coarse grid
     (`dp_ds` ≈ 5 mm, ~10–25 rail samples), continuity enforced by `max_rail_step`,
     the rate penalty, and `is_reconfiguration`.
   - **Fine refinement** interpolates the coarse `x_rail(s)` up to the retimer
     spacing (`planner.sample_ds`) and re-solves arm IK exactly at each fine
     station, predictor–corrector (`refine`, `redundant.rs:202`).
   The retimer only ever sees the smooth interpolant, never the coarse grid.

---

## How to do it

Three pieces of work.

### 1. A 7-DOF `RailMountedChain` wrapping the 6-DOF arm

Implement `ContinuousFKChain<7, f64>` by prepending one
`JointSpec::Prismatic { axis_local: X }` to the arm's `KinSpec`. `forward_pass`,
`jacobian`, and `manipulability` already handle prismatic columns
(`deke-types/src/fk.rs:432`, `:180`), so the **correct 7-column Jacobian (including
the rail column) comes for free**.

- Use this chain for FK / Jacobian / manipulability / validation / retiming.
- **Do not** implement `IkSolver<7>` — a 7-DOF redundant solve is underdetermined;
  resolving it in the planner is the entire point. IK is always the inner 6-DOF
  arm solve on the base-shifted target.

### 2. Generalize the redundancy resolver

Mirror `RedundantLinearPlanner` for a base/prismatic sweep. Cleanest shape: a
`RedundantKind` enum (or a parallel resolver) so the station builder branches on
how the DOF enters the target:

- `build_stations`: sweep `x ∈ [rail_min, rail_max]`; for each sample compute
  `T_arm(s, x)` via the base shift above, run the **6-DOF** arm `ik`, validate,
  assemble each solution into a `SRobotQ<7>` `[x_rail, q_arm…]`, score with 7-DOF
  manipulability.
- `solve_global`: same `ladder_dp`; edge cost adds `rail_rate_weight · |Δx|/Δs`;
  reject edges past `max_rail_step` or `is_reconfiguration`.
- `refine`: interpolate `x_rail(s)`, re-IK the arm with the rail baked into the
  base shift, pick the branch nearest the previous step.

Because `is_reconfiguration` already loops over all `N` joints with per-axis
`joint_v_max` (`planner.rs:145`), **once the path is 7-wide with the rail's velocity
limit in its slot, the rail is automatically guarded by the same velocity/reconfig
test as the arm joints.** No new test needed.

### 3. Config plumbing

- `JointLimits<7>` / `PlannerOptions<7>` / `LinearConstraints<7>` carry the rail's
  own v/a/j ceilings; the `ConstantSpeedRetimer` then caps rail speed and
  acceleration like any other DOF.
- New `RailOptions { axis, window: (f64, f64), samples, dp_ds, rate_weight,
  max_step }`, wired onto `FollowConfig` alongside `redundant`, and a
  `with_rail(...)` builder mirroring `with_redundancy`.

### Critical change for smoothness — do not skip

`refine` currently interpolates with **linear `interp`** (`redundant.rs:225`).
Linear is **C⁰**: position-continuous but piecewise-constant velocity, so the rail
velocity kinks at every coarse knot (an acceleration spike). For yaw that's
filtered through arm IK + the jerk-limited integrator and goes unnoticed; for a
**prismatic axis fed straight into the retimer it is exactly the discontinuity we
are trying to avoid.**

Fix: fit the coarse `x_rail(s)` knots with a **Catmull–Rom / monotone cubic**
(`squiggle` is already a dependency and already smooths the Cartesian path) for a
**C¹/C²** rail schedule. This is the single change that actually keeps the retimer
happy, and it's worth backporting to the yaw path too.

Optional: a small **rail-centering node cost** (keep the rail near the path
centroid / arm-reach mid-range) so the global DP doesn't park the rail at a window
extreme and strand the arm.

---

## Risks

### Rail recruitment can throttle the weld feed rate

The rail is **heavy and slow** — its v/a/j ceilings are far below the wrist's. The
retimer's feasible-speed ceiling is `min_j v_max,j / |q'_j(s)|`. Once the rail is a
real DOF, a track that uses the rail with high `|x_rail'(s)|` to dodge a
singularity can **throttle the entire TCP speed**. This is invisible until the
retimer runs. Mitigate with a firm `rail_rate_weight` and the centering cost so the
rail moves smoothly and only when it's actually paying for itself.

### Linear-interp discontinuity (covered above)

If the Catmull–Rom refinement is skipped, the rail injects acceleration spikes at
coarse knots — the exact failure mode this feature is supposed to prevent.

### Refinement / branch-snap mismatch

The coarse DP commits to a `(rail, branch)` track; the fine predictor–corrector can
snap the arm to a different branch than assumed, surfacing as `NoContinuousTrack`
or a residual kink. Milder for one DOF than two, but watch it on near-singular
patches. Keep `dp_ds` fine enough that adjacent coarse stations are unambiguous.

### Cost: product of samples × quadratic DP

`ladder_dp` is O(stations × layer²) and `build_stations` does one IK per
`(station × rail_sample)`. Rail-only multiplies IK count by `rail_samples` (~10–25)
vs yaw-only — acceptable. (The blow-up that makes the *combinable* version risky is
the product `rail_samples × yaw_samples`; see below.)

### Why not combinable (rail + yaw in one DP) yet

A joint 2-D `(x_rail, ψ)` DP is the riskiest config and is deliberately **out of
scope** for this step:

- **Product grid × quadratic DP × multiplied IK** → ~200×+ cost vs yaw-only.
- **Flat/degenerate cost landscape** near singularities (both DOFs relieve the same
  problem) → weight-sensitive, sloshing, hard to tune.
- **2-D coarse grid refined as two independent schedules** approximates the true
  optimum worse → *more* `NoContinuousTrack` / kinks, not fewer.
- **Poor failure attribution** (rail window? yaw window? coupling?).

If both DOFs are eventually needed, prefer **hierarchical** resolution (rail first
with yaw pinned, then yaw given the rail schedule) over the joint DP. Only fall back
to the true 2-D DP if a real weld case demonstrably fails hierarchically.

---

## Open questions (resolve before coding)

1. **Joint order** in the 7-vector: rail first (`[rail, q1..q6]`) or rail last?
   Must match the downstream robot/retimer convention.
2. **Rail axis**: fixed world `+X`, or expose a general unit axis like
   `RedundantAxis::Custom`? (Spec says X; general is cheap.)
3. **Rail window**: absolute machine limits, or a soft window around the path
   centroid per run?
4. Whether to backport the Catmull–Rom refinement to the existing yaw path in the
   same change.
