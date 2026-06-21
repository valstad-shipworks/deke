# Weave / torch oscillation — design note

Status: **planned, not started.** Scope captured so we can resume cold. Related:
`docs/rail-axis-redundancy.md` (both are about what the *path primitive* can express
and how that flows into the constant-speed retimer).

Target: let `deke-linear` follow a seam while the torch oscillates *transverse* to
travel (weave beading) — the standard technique for filling wide joints, ensuring
sidewall fusion, and welding out-of-position.

---

## What it is

A weave is a deterministic lateral oscillation superimposed on the nominal seam
path. The TCP traces a side-to-side pattern (sine, triangle, zig-zag, crescent,
figure-8) as it travels down the joint. It is parameterised by:

- **Amplitude** `A` — peak-to-peak transverse width.
- **Wavelength** `λ` (spatial) or **frequency** `f` (temporal) — oscillations per
  unit seam length, or per second.
- **Pattern shape** — sine / triangle / etc.
- **Dwell** — a pause at the weave extremes (and sometimes centre) for edge fusion.
  *Temporal* concept; see the phasing below.
- **Weave plane** — the transverse direction `n̂`, perpendicular to travel.

Mechanically it is a **Stage A (`path`) concern, not a retimer concern** — a weave
is a richer path geometry, not a different time law — *with one semantic change to
Stage C* (what "constant speed" means) and *one important exception* (dwell).

### The semantic fork: spatial vs. temporal weave

This is the crux of the design.

- **Spatial weave** — the oscillation is a function of **seam arc length** `u`:
  `λ` is fixed in space, the bead pattern is locked to the workpiece. Pure geometry.
  Fits the constant-feedrate model cleanly. **This is Phase 1.**
- **Temporal weave with dwell** — the oscillation is a function of **time**, with
  explicit dwell *durations* at the edges. Dwell decouples lateral motion from seam
  progress and is inherently a *time* concept, so it cannot live purely in Stage A —
  it must be handled by the retimer (Stage C), which owns time. **This is Phase 2,
  deferred.**

Recommendation: ship spatial weave (no dwell) first. It covers the bulk of weave
needs (wide bead, fill, sidewall fusion via pattern shape) and slots into the
existing architecture with minimal disruption. Dwell is a separate, more invasive
follow-up.

---

## What it solves

- **Wide joints / fill passes.** A straight pass lays a narrow bead; weaving spreads
  deposit across the joint width — needed for wide root gaps and bevel fills.
- **Sidewall fusion & heat distribution.** Pattern shape (and, later, dwell at the
  toes) wets the puddle into both plates without overheating the centre.
- **Out-of-position control.** Vertical-up / overhead beads rely on weave patterns
  to let the puddle freeze at the edges.

---

## How to do it (Phase 1 — spatial weave)

### 1. Define the weave plane from the **tool frame** (not Frenet)

The transverse direction `n̂(u)` must be robust. **Do not** use the Frenet normal:
it is undefined on straight sections (zero curvature — i.e. exactly the seams you
weave most) and flips/twists at inflection points on 3D seams.

Instead define the weave axis in the **tool frame**, mirroring the existing
`RedundantAxis` API (`src/redundant.rs:27`): the run already carries an orientation
quaternion `R(u)` (slerped `vtx_q`, `path.rs:65`), so

```
n̂(u) = R(u) · â_tool        // â_tool e.g. tool +Y, configurable
```

This is degeneracy-free, physically meaningful (weave ⟂ the wire), and consistent
with how tool-yaw redundancy already names axes.

### 2. Overlay the weave in Stage A

The weaving pose at seam length `u` is

```
P(u) = Trans( A(u) · shape(2π u / λ) · n̂(u) ) · seam_pose(u)
```

where `seam_pose(u) = CartesianRun::eval(u)` (`path.rs:37`) is today's pose, and
`A(u)` **tapers to zero at each run's start/end** so the weave vanishes into the
rest ramps (the conditioner already stops runs at rest at sharp corners —
`path.rs:94`). Cleanest implementation: a `WeavingRun` wrapper (or a `weave:
Option<WeaveOptions>` field on `CartesianRun`) whose `eval` applies the overlay, so
Stage B/C are unchanged structurally and the planner keeps IK-ing `run.eval(u)`.

### 3. Re-base the retimer on the **seam** parameter (the one Stage-C change)

Today the retimer derives arc length from the **FK end positions of the joint
path** (`retimer.rs:52–60`) and holds `tcp_speed` along *that* length. With a weave
baked in, that would hold constant *total* TCP speed — torch tip constant including
the cross-strokes — which is **not** what a welder wants. Welders hold constant
**travel speed** (seam progress `u̇`), because heat input per unit *seam* length is
what governs the bead.

Fix: parameterise the retimer by the seam length `u` instead of the weaving-path
length:

- pass the per-sample `u[i]` array alongside the path (instead of recomputing `s`
  from FK positions),
- compute `q'(u)` by central difference over `u` (`retimer.rs:77`),
- command constant `u̇ = travel_speed`.

The whole MVC machinery then works unchanged: `project_min(q'(u), v_max)`
(`retimer.rs:192`) already gives `min_j v_max,j / |q'_j(u)|`, which is the ceiling
on `u̇`. The weave makes `|q'(u)|` larger (lateral motion adds to `dq/du`), so the
achievable travel speed is **naturally throttled by the weave's joint demand** —
correct behaviour. When `A = 0`, `u ≡ s` and the path must be bit-identical to today
(regression guard).

### 4. Config

A `WeaveOptions { pattern, amplitude, wavelength, axis: RedundantAxis, taper }` on
`FollowConfig` (alongside `redundant`), with a `with_weave(...)` builder mirroring
`with_redundancy` (`constraints.rs:135`). Travel speed reuses `tcp_speed` (now
defined as seam-travel speed).

---

## Risks

### Sampling resolution / Nyquist — the big one, and exactly your standing concern

A weave injects geometry at wavelength `λ`. To resolve it without aliasing, the
planner's `sample_ds` must be **≪ λ** — at least ~10–20 samples per weave cycle.
That multiplies the IK count in Stage B and the sample count in Stage C. There is no
coarse-grid escape here (unlike the rail DOF) because the weave *is* the
high-frequency content of the path, not a smooth scalar to be interpolated. Budget
for it: `sample_ds ≤ λ / 15` or so, set automatically from `WeaveOptions`.

### Curvature breaks the retimer's `q''·ṡ²` approximation

The retimer deliberately approximates acceleration/jerk via tangent projection and
**omits the `q''(s)·ṡ²` curvature cross-term** (`retimer.rs:13–14`), noting it is
negligible at process speeds. **Weaving violates that assumption:** weave curvature
is `~ A·(2π/λ)²` and dominates the path — the cross-term is no longer negligible, so
the a/j ceilings can be under-predicted and the real joint acceleration on the
cross-strokes can exceed limits. Likely need to **promote that approximation to an
exact `q''(u)·u̇²` term for the weaving case** (compute `q''(u)` by second difference
and fold it into `a_path`). This is the subtlest correctness risk.

### Amplitude/frequency are limited by joint a/j, not just reach

High `A` at high `f` demands large transverse joint acceleration. The feasible weave
envelope is bounded by `a_max`/`j_max`, not just `v_max` and workspace — so a weave
that "fits" geometrically can still be infeasible dynamically, surfacing as a
collapsed `u̇` ceiling or `Stalled`. Diagnostics should report the binding axis.

### Singularity / reconfiguration interaction

Cross-strokes add joint motion the straight-seam analysis never sees. Near a
singularity or workspace edge the lateral excursions can trip `is_reconfiguration`
(`planner.rs:135`) or collapse the speed ceiling mid-weave even where the seam alone
was fine. Weaving near reach limits is the danger zone.

### Orientation weave couples with the yaw-redundancy resolver

Some procedures also oscillate torch *orientation* (work/travel angle to the
sidewall). That overlays on the same orientation channel the tool-yaw redundancy
resolver manipulates — same class of coupling risk as the rail+yaw combinable case
(`docs/rail-axis-redundancy.md`). Keep positional weave and orientation weave
separate; treat orientation weave as its own later phase.

### Corner / run-boundary behaviour

Weave through or near a sharp corner is ill-defined. The amplitude taper (step 2)
must drive `A → 0` into every run boundary so the weave never fights the rest ramp.

### Dwell is temporal — it breaks the pure geometric model (Phase 2)

Dwell holds position for a *time* at the weave edge; it cannot be expressed as
`f(u)`. It must be inserted by the retimer as a timed hold at the parameter values
where the weave is at an extreme, which means Stage C has to know the weave phase.
Deferred, and flagged as the reason temporal weave is a separate phase.

---

## Open questions (resolve before coding)

1. **Spatial vs. temporal first** — confirm Phase 1 = spatial, no dwell. Does the
   target controller/process actually need true timed dwell, or is pattern shape
   (triangle/crescent with edge-biased shaping) enough?
2. **Weave axis source** — tool-frame axis (recommended, robust) vs. a supplied
   per-vertex weld coordinate frame? Tool-frame needs the right `â_tool` for the
   torch convention (this project is Z-forward).
3. **Pattern set** — which shapes to ship first (sine + triangle likely enough)?
4. **Does `tcp_speed` get redefined as travel speed**, or add a distinct
   `travel_speed` field and keep `tcp_speed` meaning total-tip speed? (Affects API
   back-compat.)
5. **Promote the `q''·ṡ²` term globally or only under weave?** Doing it only for the
   weave case avoids regressing the validated straight-weld path.
