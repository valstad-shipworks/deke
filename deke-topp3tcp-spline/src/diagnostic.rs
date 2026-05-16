//! Diagnostics returned alongside the retimer output.

use std::fmt;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    NotAttempted,
    Success,
    SearchExhausted,
}

impl fmt::Display for SolveStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveStatus::NotAttempted => f.write_str("NotAttempted"),
            SolveStatus::Success => f.write_str("Success"),
            SolveStatus::SearchExhausted => f.write_str("SearchExhausted"),
        }
    }
}

impl Default for SolveStatus {
    fn default() -> Self {
        SolveStatus::NotAttempted
    }
}

/// Diagnostic data produced by every retimer call.
#[derive(Debug, Clone, Default)]
pub struct Topp3TcpSplineDiagnostic {
    pub status: SolveStatus,
    pub input_waypoints: usize,
    pub deduplicated_waypoints: usize,
    /// Number of discrete trajectory states (including start and end).
    pub output_states: usize,
    /// Total trajectory duration.
    pub total_time: Duration,
    /// Wall-clock spent inside the optimizer.
    pub solve_time: Duration,
    /// Optional human-readable message (typically the error reason).
    pub message: Option<String>,
}

impl fmt::Display for Topp3TcpSplineDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Topp3TcpSpline[status={}, in_wps={}, dedup_wps={}, states={}, total={:?}, solve={:?}",
            self.status,
            self.input_waypoints,
            self.deduplicated_waypoints,
            self.output_states,
            self.total_time,
            self.solve_time,
        )?;
        if let Some(msg) = &self.message {
            write!(f, ", msg={:?}", msg)?;
        }
        f.write_str("]")
    }
}
