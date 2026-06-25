use std::fmt;
use std::time::Duration;

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
        )
    }
}
