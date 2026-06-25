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
   keep the arm well-conditioned. **Score the node cost on the *arm's* 6-DOF
   manipulability, not the 7-DOF chain's** (see *Implementation notes* — this is the
   reverse of the original guess here): the rail's prismatic Jacobian column keeps
   `det(J Jᵀ)` high even when the arm itself is singular, so a 7-DOF measure *masks*
   an arm singularity and the DP parks the rail instead of recruiting it. Scoring
   the arm alone makes the DP spend the rail to keep the arm off its singular sets.
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

---

## Research-backed decisions + sources

A literature/OSS survey (June 2026) confirms the architecture above and settles
the open questions. The one substantive correction is the smoothing choice: the
rail wants a **monotone** smoother (PCHIP), not Catmull–Rom.

### Decisions

1. **Resolve the rail in the planner DP; never as a retimer LP variable.** The
   `ConstantSpeedRetimer` is the Verscheure / TOPP-RA convex family: per-joint
   v/a/j become *linear* bounds on the finite differences of `σ` **only because
   the per-segment secant `dq/ds` (`retimer.rs`) is a precomputed constant along a
   fixed path.** The rail enters IK through the nonlinear base shift
   `T_arm = (Trans(x·x̂)·base0)⁻¹·T_world`, so a variable `x_rail` makes `dq/ds`
   (and every constraint coefficient) nonlinear in it and destroys convexity. The
   bi-level paper proves exactly this for a redundant DOF; redundantly-*actuated*
   TOPP only redistributes forces on a fixed path and cannot absorb kinematic
   redundancy. ⇒ Choose `x_rail(s)` upstream, bake it into a 7-wide `q(s)`, and the
   retimer sees only **more (tighter) linear FD rows on the same `σ`** — reused
   unchanged. `is_reconfiguration` and `verify_fd` already loop over all `N`
   joints, so once the path is 7-wide with the rail's v/a/j in slot 0 the rail is
   guarded by construction (hard FD invariant intact).

2. **Combine rail + yaw hierarchically, not as a 2-D DP.** Run the 1-D resolver
   twice: rail DP with yaw pinned (rail = coarse workspace/reach + singularity
   relief), then yaw DP on the fixed rail schedule (yaw = fine conditioning). Cost
   stays additive. The joint `(rail × yaw × branch)` DP is `O(L·S²)` with
   `S = m_rail·m_yaw·branches` (~200×+), and its flat/degenerate cost basin near
   singularities (both DOFs relieve the same problem) makes it slosh and produce
   *more* `NoContinuousTrack`/kinks. Graph-of-Convex-Sets (Drake GCS) is the
   principled scalable unification — deferred until a real weld case fails
   hierarchically.

3. **Smoothing — rail ≠ yaw.** Rail → **Fritsch–Carlson monotone cubic (PCHIP)**:
   a heavy slow axis must not overshoot a sampled value (overshoot = wasted travel
   that throttles feedrate). PCHIP is only C¹, so feed the retimer densely
   resampled knots and let the chord-linear secant absorb the knot-curvature jump
   at bin boundaries — never differentiate the raw curve for jerk. Yaw →
   **centripetal Catmull–Rom (α=0.5)** (cusp/self-intersection-free; correct for a
   symmetric DOF with no monotonicity to preserve). **Do NOT use Catmull–Rom for
   the rail** — it is not monotone and wiggles past samples.

4. **Feedrate-throttle is the one genuinely new failure mode.** The retimer's
   ceiling is `minⱼ v_max,j/|q'_j(s)|`; a rail schedule with large `|x_rail'(s)|`
   silently caps TCP speed. Mitigate with a firm `rail_rate_weight` + a
   rail-centering node cost (recruit the rail only when it pays for itself); the
   principled escape when the rail truly can't keep up is a minimal feedrate
   breakpoint, not a global derate.

5. **Solver follow-ups (not required for v1):** keep Clarabel; the iterated
   derate loop re-factorizes near-identical LPs, so warm-started OSQP, or a banded
   LDLᵀ / Riccati factorization exploiting the block-tridiagonal `σ[k]` KKT with
   the `Σσ` cap as a low-rank arrow row, are the speed paths.

### Open questions — resolved

1. **Joint order:** rail **first** — `q = [x_rail, q1..q6]`.
2. **Rail axis:** configurable world unit axis (default `+X`), used for both the
   base shift and the prismatic Jacobian column.
3. **Rail window:** per-run soft window around the path centroid, clamped to
   absolute machine limits.
4. **Yaw Catmull–Rom backport:** deferred — it would change existing yaw output
   and the hard requirement is *no regression*. Tracked as a follow-up.

### Sources

Foundational (why the rail can't be an LP variable):
- Bi-Level Optimization for Redundant Manipulators — arXiv 2412.07859
- Verscheure et al., Time-Optimal Path Tracking: A Convex Optimization Approach — IEEE 5256286
- Pham & Pham, TOPP-RA (T-RO 2018) — arXiv 1707.07239
- Pham & Stasse, TOPP for redundantly-actuated robots — ntu.edu.sg/cuong/docs/overactuated.pdf

Redundant / external-axis resolution & welding:
- Effective path planning for robot welding w/ redundant kinematics (RCIM 2024) — sciencedirect S0141635924002356
- Kinematic aspects of a robot-positioner system in arc welding — sciencedirect S0967066102001776
- Co-Optimization of tool orientations, kinematic redundancy & timing — arXiv 2409.13448
- Real-time feedrate optimization for laser processes w/ redundant axes (2025) — sciencedirect S0890695525000975
- DP-based redundancy resolution considering breakpoints — arXiv 2411.17034
- eSNS hierarchical redundancy resolution — arXiv 2204.03974
- Expansion-GRR — arXiv 2405.13770

OSS planners:
- swri-robotics/descartes_light — github.com/swri-robotics/descartes_light
- tesseract-robotics/tesseract_planning
- Drake GcsTrajectoryOptimization / Marcucci et al. — arXiv 2205.04422

Smoothing & solvers:
- Monotone cubic (Fritsch–Carlson / PCHIP) — Wikipedia
- Centripetal Catmull–Rom spline — Wikipedia
- OSQP — osqp.org
- Clarabel — arXiv 2405.12762
- Generalized Riccati for equality-constrained LQ control / HPIPM–BLASFEO — arXiv 2302.14836

---

## Implementation notes (as built)

Shipped in `src/rail.rs` (+ `pchip` in `src/util.rs`, `RailDiagnostic`, exports).
`RailMountedChain<A,N>` implements `ContinuousFKChain<N>` only (never `IkSolver<N>`);
the 7-DOF path flows through the **unchanged** `ConstantSpeedRetimer`. Two findings
diverged from the pre-implementation plan:

1. **Score the arm's manipulability, not the chain's.** The decisive correctness
   fix. With 7-DOF scoring the rail column masks an arm (e.g. elbow) singularity, so
   the DP parked the rail and the arm hit a near-singular start on the weld-Y ×
   rail-45° cell — a `q'''` boundary layer that spiked joint jerk ~5.8× and the
   retimer (correctly) refused it. Scoring `arm.manipulability(q_arm)` makes the DP
   move the rail (e.g. 0.24 m → 0.03 m on that cell) to keep the elbow bent; the
   start becomes smooth and the pristine retimer times it in ~4 s.

2. **Refine with a branch ladder-DP, not greedy nearest-branch.** With the rail
   schedule fixed, the arm is an ordinary branch-tracking problem; the greedy walk
   from the yaw planner was myopic and stranded the diagonal-rail cell with
   `NoContinuousTrack`. Refine now runs the same `ladder_dp` the base planner uses.

Tried and rejected: extending the retimer's adaptive derate loop to the joint caps
(global derate blew up the LP horizon; per-segment derate still failed at the start
and ran ~130 s/cell). The real defect was upstream (finding 1); the retimer needed
no change. The yaw-Catmull-Rom backport remains deferred (would alter existing yaw
output; the hard requirement is no regression).

**Verification.** Test matrix `tests/rail_matrix.rs`: weld ∈ {X, Y, 45°} × rail ∈
{X, Y, 45°} × refine ∈ {Linear, PCHIP} = 18 cells, all pass with high rail limits
(v 1 m/s, a 20 m/s², j 2000 m/s³). Each asserts FD v/a/j ≤ limit·(1+1e-9) on every
joint (rail included), cruise speed within 2 %, deviation < 2 mm, rail within window.
Realized joint-jerk peaks ~0.97× limit, accel ~0.10–0.23×, deviation ~0.1 µm.
`tests/rail_yaw.rs` proves hierarchical rail+yaw composes. No existing test regressed;
`cargo clippy --all-targets` clean.

### Workspace extension (over-reach seam) — and fast scans

`tests/rail_reach.rs` runs a straight seam **longer than the arm's reach**: the
fixed-base 6-DOF planner fails (the far end has no IK), while the rail carries the
base along the seam and the whole pass is planned and timed within all joint limits
(seam 1.38 m vs ~1.68 m reach upper bound; rail travels the full seam; TCP exact to
~0.3 µm). It runs at **0.25 m/s scan feed** and at **30 IPM weld feed** (two tests).
Getting a fast traverse to time cleanly drove four changes, all isolated to the rail
planner (the retimer stayed pristine — fixing it there is what makes a long, fast
scan honour the per-tick jerk limit by construction):

- **Edge cost measures arm motion, not the 7-vector.** `solve_global` originally
  scored each DP edge by `q.distance` over all 7 joints, counting rail traverse as
  "joint motion" to minimise — on a long seam it stalled the rail and made the arm
  over-reach. The edge now measures **arm** joint motion only; rail travel is
  governed by `rate_weight`, `is_reconfiguration`, and the schedule smoothing. For a
  pure traverse set `rate_weight = 0` (the rail must follow the TCP).
- **Continuous polish + slope-preserving smoothing of the rail schedule.** The rail
  position only changes *how* the arm reaches the target — the inner IK still hits
  the exact weld pose for any `x` — so `x(s)` can be smoothed freely (the TCP stays
  exact). Each coarse knot is polished to its continuous arm-manipulability optimum
  (golden section; on a straight scan that is exactly `x = s − const`), then the fine
  schedule is box-smoothed with linearly-extrapolated ends (a naïve filter
  back-averages the ends and *flattens* the ramp, which makes the arm reconfigure).
- **Even fine-station spacing.** Clamping `i·sample_ds` to `length` leaves a short
  remainder as the final interval, whose large secant the retimer reads as a sharp
  end segment; the stations are now spaced evenly over `[0, length]`.
- **Joint-jitter removal.** The independent analytic IK solves carry ~1e-5 rad of
  floating-point jitter (sub-mm at the TCP). At weld feed it is invisible; cubed by a
  0.25 m/s scan it becomes a spurious jerk spike, so the planned joint columns get a
  narrow slope-preserving filter before retiming (well within path tolerance).

Net: rail v/a/j ceilings (`a 20 m/s²`, `j 2000 m/s³` here — set high; the higher the
better) are honoured by construction, and the matrix's short cells are unaffected
(the rail parks there).

**The grid spacing is derived, not configured.** `RailOptions::samples` is only a
floor; the planner raises it so the spacing meets both the reconfiguration limit
(`≤ frac·v_rail·dp_ds/speed`, ≈ 0.018 m at 0.25 m/s — needed so the rail can advance
a sample per DP station) and a resolution limit (`≲ 3·dp_ds`, so the coarse DP can
resolve a long traverse rather than routing a degenerate, arm-over-reaching track).
The caller never sizes the grid to the scan speed.
