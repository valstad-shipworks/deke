# deke-topp3-lp — findings & future work

Notes from a production integration (orchestra `nanopanel-material`, a 7-DOF RTU
robot: prismatic rail at q[0] in metres carrying a 6-DOF arm). Two issues surfaced
on long single-chord transitions that slide the rail several metres *and*
reconfigure the arm a lot at once, so the **TCP linear-speed cap is the binding
constraint** (the TCP sweeps ~3–4 m of straight-line distance).

The reproductions are real, self-contained tests in this crate:
- `tests/rtu_tcp_cap_long_chord.rs` — BUG-12 regression (fixed; see below).
- `tests/collinear_perf.rs` — perf characterization of the Collinear slowness
  (`#[ignore]`; run with `--ignored --nocapture`).
- `tests/common/mod.rs::material_7dof()` + `MATERIAL_PATH_A/B` — the extracted
  KinSpec and the two failing chords.

---

## FIXED — BUG-12: per-segment TCP `kappa` under-estimate on long chords

**Symptom.** `Topp3LpTcp::retime` returned `TcpLimitExceeded` (e.g. realised peak
`2.0984 > 2.0000 m/s`, ~5% over) on a chord the cap should comfortably admit by
slowing down. It only triggered when the arm could move fast (the realised TCP
peak occurs mid-chord), so it appeared with the fast uniform limits but not the
slower per-axis material limits.

**Root cause.** `kappa_per_segment` computed `κ = ‖J_lin·secant‖` (TCP speed per
unit `σ̇`) using the Jacobian at each segment's **start knot only**. The Jacobian
varies along a segment, so on a long chord that reconfigures the arm, the start
knot under-estimates the mid-chord `κ`. The σ-LP then plans `σ̇` against a cap that
is too loose, the realised FK secant speed overshoots, and the `time_run` derate
loop (6 passes) cannot claw back a few-percent overshoot.

**Fix.** Make `κ` conservative: sample the Jacobian at several interior points of
each segment and take the max, so the per-segment cap bounds the worst-case TCP
gain and the first solve already respects the cap. Cheap under `Collinear`
(segments are short there); 5 samples is enough for `Raw` long chords. Verified:
all four `rtu_tcp_cap_long_chord` cases pass and the existing suite stays green.

**Production note.** orchestra uses `Conditioning::Raw` (one segment per planner
edge) + this conservative `κ`. That is accurate and fast (1–3 s per retime) and is
the recommended configuration — *not* `Collinear` (see below).

---

## OPEN — Collinear conditioning is pathologically slow (the "hang")

**Symptom.** With `Conditioning::Collinear(res)` a single long-chord retime takes
**5–16 s** (and, before BUG-12, the un-converging derate loop multiplied that into
a minutes-long apparent hang). It is **not** an infinite loop — every loop is
bounded and Clarabel is capped at `max_iter = 200` — it is pathological re-solve
blowup.

**Measured** (real material chain, Path B, see `tests/collinear_perf.rs`):

| conditioning      | TCP cap | time   |
|-------------------|---------|--------|
| Raw               | yes     | ~3.0 s |
| Collinear(0.20)   | yes     | ~11.8 s|
| Collinear(0.10)   | yes     | ~6.4 s |
| Collinear(0.05)   | yes     | ~16.5 s|
| Collinear(0.05)   | no      | ~5.2 s |
| Raw               | no      | ~1.3 s |

**Root cause.** Collinear densifies the chord into many knots (≈56 at res=0.05),
and the σ-LP is re-solved from scratch across three nested loops with **no
warm-starting**:
- `time_run` TCP-derate loop (≤6), each re-runs all of `time_chord`;
- `time_chord` horizon-grow loop (≤8), each a fresh solve;
- `time_chord` bin-convergence passes (≤8) + a boxed re-solve per grow.

Densification inflates both the per-solve size (more ticks/bins) and the number of
grow/box re-solves (denser bins make `verify_joint_fd` straddle more sub-bins,
forcing more horizon growth). Evidence it tracks re-solve **count**, not knot
count: the res→time curve is **non-monotonic** (0.20→11.8 s, 0.10→6.4 s,
0.05→16.5 s). The TCP-derate loop multiplies it further (`tcp` 16.5 s vs `nocap`
5.2 s at res=0.05).

**Why it matters.** A consumer that reaches for `Collinear` to fix the per-segment
`κ` accuracy (before BUG-12 was understood) pays this cost on every transition; in
a batched run that reads as a hang.

**Future fix options (none implemented):**
1. **Warm-start Clarabel** across the grow/derate re-solves (reuse the previous
   `σ` as the IPM start, or only re-solve when the TCP cap actually scales). The
   nested loops currently throw away every prior solve.
2. **Decouple the TCP cap from densification.** With conservative `κ` (BUG-12),
   `Raw` already enforces the cap accurately, so densification is not needed for
   correctness — only `Collinear`'s separate purpose (genuinely curved
   *multi-segment* inputs) should pay for it.
3. **Cap/short-circuit horizon growth** when the FD-verify violation is tiny and
   not improving, instead of growing the grid and re-solving.

Given (2), the pragmatic recommendation stands: prefer `Raw` + conservative `κ`;
treat `Collinear` as opt-in for curved inputs and only after the warm-start work.
