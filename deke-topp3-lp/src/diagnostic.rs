use std::fmt;
use std::time::Duration;

/// Which backstops the retimer had to engage to time the path, in escalating
/// order of severity. An all-`default` value means the σ-LP timed every run in one
/// pass at the requested caps — no recovery needed.
#[derive(Clone, Debug, Default)]
pub struct RetimeRecovery {
    /// Per-joint planned-limit derates the precision backstop applied, as
    /// `(joint, kind, factor)` where `kind` is `"velocity"`/`"acceleration"`/
    /// `"jerk"` and `factor < 1` is how far the *planned* cap was tightened below
    /// the true cap so the reconstruction's realized finite difference lands under
    /// it (the σ-LP solves the tiny jerk rows only to convex tolerance). Cheap —
    /// only the binding ramp is slowed.
    pub derates: Vec<(usize, &'static str, f64)>,
    /// The uniform time-stretch backstop fired on at least one run: the σ-LP
    /// profile overran its own caps and neither derating nor horizon growth could
    /// bring it under, so the whole run was slowed to clear the worst FD. A last
    /// resort — a `true` here means a run is slower than time-optimal.
    pub time_scaled: bool,
    /// Rest stops the bisection backstop inserted: a run too curved/tight to time
    /// in one rest-to-rest pass was split at an on-chord waypoint into halves timed
    /// independently (recursively). Each count is one added momentary stop on the
    /// planned path. `0` = no segmentation.
    pub bisections: usize,
}

impl RetimeRecovery {
    /// `true` when no backstop was engaged — the path timed cleanly at the caps.
    pub fn is_clean(&self) -> bool {
        self.derates.is_empty() && !self.time_scaled && self.bisections == 0
    }

    pub(crate) fn merge(&mut self, other: RetimeRecovery) {
        self.derates.extend(other.derates);
        self.time_scaled |= other.time_scaled;
        self.bisections += other.bisections;
    }
}

/// Outcome of a convex-LP retime.
#[derive(Clone, Debug)]
pub struct Topp3LpDiagnostic {
    pub output_samples: usize,
    pub duration: Duration,
    /// Joint-space arc length of the timed path.
    pub arc_length: f64,
    /// Peak per-axis finite-difference velocity/acceleration/jerk of the output.
    pub peak_joint_vel: f64,
    pub peak_joint_accel: f64,
    pub peak_joint_jerk: f64,
    /// Peak TCP linear speed (m/s) — `0.0` when retimed without an FK chain.
    pub peak_tcp_speed: f64,
    /// Which recovery backstops (if any) the retimer engaged to time this path.
    pub recovery: RetimeRecovery,
}

impl Topp3LpDiagnostic {
    pub(crate) fn zeroed() -> Self {
        Self {
            output_samples: 0,
            duration: Duration::ZERO,
            arc_length: 0.0,
            peak_joint_vel: 0.0,
            peak_joint_accel: 0.0,
            peak_joint_jerk: 0.0,
            peak_tcp_speed: 0.0,
            recovery: RetimeRecovery::default(),
        }
    }
}

impl fmt::Display for Topp3LpDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "topp3-lp: {} samples over {:.3}s, {:.4} arc, peak v/a/j {:.3}/{:.3}/{:.3}, tcp {:.4} m/s",
            self.output_samples,
            self.duration.as_secs_f64(),
            self.arc_length,
            self.peak_joint_vel,
            self.peak_joint_accel,
            self.peak_joint_jerk,
            self.peak_tcp_speed,
        )?;
        let r = &self.recovery;
        if r.is_clean() {
            return Ok(());
        }
        write!(f, " [recovery:")?;
        if r.bisections > 0 {
            write!(f, " {} bisection rest-stop(s)", r.bisections)?;
        }
        if r.time_scaled {
            write!(f, " time-scaled")?;
        }
        if !r.derates.is_empty() {
            write!(f, " derated")?;
            for (j, kind, factor) in &r.derates {
                write!(f, " j{j}.{kind}×{factor:.3}")?;
            }
        }
        write!(f, "]")
    }
}
