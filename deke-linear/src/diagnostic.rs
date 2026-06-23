use std::fmt;
use std::time::Duration;

/// Outcome of the Stage B branch-tracking plan over one run.
#[derive(Clone, Debug)]
pub struct LinearPlannerDiagnostic {
    pub samples: usize,
    pub min_manipulability: f64,
    pub total_cost: f64,
}

impl fmt::Display for LinearPlannerDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "planner: {} samples, min manipulability {:.3e}, route cost {:.3}",
            self.samples, self.min_manipulability, self.total_cost
        )
    }
}

/// Outcome of the Stage C constant-speed retime over one run.
#[derive(Clone, Debug)]
pub struct LinearRetimerDiagnostic {
    pub output_samples: usize,
    pub duration: Duration,
    pub arc_length: f64,
    pub commanded_speed: f64,
    pub peak_speed: f64,
    /// Peak continuous per-joint acceleration `|q'·a + q''·v²|` over the run.
    pub peak_joint_accel: f64,
    /// Peak continuous per-joint jerk `|q'·j_s + 3·q''·a·v + q'''·v³|` over the
    /// run. Bounded by the joint jerk limit by construction (the FD third
    /// difference of the dt-sampled output can read higher at the jerk steps a
    /// jerk-limited profile necessarily takes).
    pub peak_joint_jerk: f64,
}

impl fmt::Display for LinearRetimerDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "retimer: {} samples over {:.3}s, {:.4}m, commanded {:.4} m/s, peak {:.4} m/s",
            self.output_samples,
            self.duration.as_secs_f64(),
            self.arc_length,
            self.commanded_speed,
            self.peak_speed
        )
    }
}

/// Outcome of the redundancy-resolving (free-yaw) plan over one run.
#[derive(Clone, Debug)]
pub struct RedundantDiagnostic {
    pub samples: usize,
    pub min_manipulability: f64,
    /// (min, max) resolved yaw about the tool axis, radians.
    pub yaw_range: (f64, f64),
}

impl fmt::Display for RedundantDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "redundant: {} samples, min manipulability {:.3e}, yaw ∈ [{:.1}°, {:.1}°]",
            self.samples,
            self.min_manipulability,
            self.yaw_range.0.to_degrees(),
            self.yaw_range.1.to_degrees()
        )
    }
}
