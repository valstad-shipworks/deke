#![cfg(feature = "valuable")]
//! `valuable::Valuable` impls for the public diagnostics, settings, and
//! constraint types.

use std::time::Duration;

use ::valuable::{
    EnumDef, Enumerable, Fields, NamedField, NamedValues, StructDef, Structable, Valuable, Value,
    Variant, VariantDef, Visit,
};

use super::constraints::{
    BoundaryConditions, DensificationOptions, JointLimits, SolverOptions, TcpLimits,
    Topp3Tcp6Constraints,
};
use super::diagnostic::{
    BoundarySlackUsage, ConstraintCounts, DerivativeStats, InitialGuessStats, LimitingGroup,
    PathStats, PeakLocation, PhaseTiming, SolveStatus, TcpStats, Topp3Tcp6Diagnostic,
};
use crate::common::boundary::ProjectedBoundary;

/// Renders a `Duration` as its `f64` second count for inspection purposes.
fn duration_secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

const LIMITING_GROUP_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("JointVelocity", Fields::Unnamed(0)),
    VariantDef::new("JointAcceleration", Fields::Unnamed(0)),
    VariantDef::new("JointJerk", Fields::Unnamed(0)),
    VariantDef::new("TcpVelocity", Fields::Unnamed(0)),
    VariantDef::new("TcpAcceleration", Fields::Unnamed(0)),
    VariantDef::new("TcpJerk", Fields::Unnamed(0)),
    VariantDef::new("BoundaryCondition", Fields::Unnamed(0)),
    VariantDef::new("TimestepLowerBound", Fields::Unnamed(0)),
];

impl Valuable for LimitingGroup {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        visit.visit_unnamed_fields(&[]);
    }
}

impl Enumerable for LimitingGroup {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("LimitingGroup", LIMITING_GROUP_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            Self::JointVelocity => 0,
            Self::JointAcceleration => 1,
            Self::JointJerk => 2,
            Self::TcpVelocity => 3,
            Self::TcpAcceleration => 4,
            Self::TcpJerk => 5,
            Self::BoundaryCondition => 6,
            Self::TimestepLowerBound => 7,
        };
        Variant::Static(&LIMITING_GROUP_VARIANTS[idx])
    }
}

const SOLVE_STATUS_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("Success", Fields::Unnamed(0)),
    VariantDef::new("CallbackRequestedStop", Fields::Unnamed(0)),
    VariantDef::new("TooFewDofs", Fields::Unnamed(0)),
    VariantDef::new("LocallyInfeasible", Fields::Unnamed(0)),
    VariantDef::new("GloballyInfeasible", Fields::Unnamed(0)),
    VariantDef::new("FactorizationFailed", Fields::Unnamed(0)),
    VariantDef::new("LineSearchFailed", Fields::Unnamed(0)),
    VariantDef::new("FeasibilityRestorationFailed", Fields::Unnamed(0)),
    VariantDef::new("NonfiniteInitialGuess", Fields::Unnamed(0)),
    VariantDef::new("DivergingIterates", Fields::Unnamed(0)),
    VariantDef::new("MaxIterationsExceeded", Fields::Unnamed(0)),
    VariantDef::new("Timeout", Fields::Unnamed(0)),
    VariantDef::new("NotAttempted", Fields::Unnamed(0)),
];

impl Valuable for SolveStatus {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        visit.visit_unnamed_fields(&[]);
    }
}

impl Enumerable for SolveStatus {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("SolveStatus", SOLVE_STATUS_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            Self::Success => 0,
            Self::CallbackRequestedStop => 1,
            Self::TooFewDofs => 2,
            Self::LocallyInfeasible => 3,
            Self::GloballyInfeasible => 4,
            Self::FactorizationFailed => 5,
            Self::LineSearchFailed => 6,
            Self::FeasibilityRestorationFailed => 7,
            Self::NonfiniteInitialGuess => 8,
            Self::DivergingIterates => 9,
            Self::MaxIterationsExceeded => 10,
            Self::Timeout => 11,
            Self::NotAttempted => 12,
        };
        Variant::Static(&SOLVE_STATUS_VARIANTS[idx])
    }
}

const DIAG_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("status"),
    NamedField::new("iterations"),
    NamedField::new("solve_time_secs"),
    NamedField::new("solver_tolerance_used"),
    NamedField::new("densified_samples"),
    NamedField::new("output_samples"),
    NamedField::new("total_time_secs"),
    NamedField::new("peak_joint_velocity"),
    NamedField::new("peak_joint_acceleration"),
    NamedField::new("peak_joint_jerk"),
    NamedField::new("peak_tcp_velocity"),
    NamedField::new("peak_tcp_acceleration"),
    NamedField::new("peak_tcp_jerk"),
    NamedField::new("average_utilization"),
    NamedField::new("boundary_projection_residual"),
    NamedField::new("limiting_constraint"),
    NamedField::new("limiting_sample"),
    NamedField::new("message"),
    NamedField::new("path_stats"),
    NamedField::new("derivative_stats"),
    NamedField::new("tcp_stats"),
    NamedField::new("peak_joint_velocity_at"),
    NamedField::new("peak_joint_acceleration_at"),
    NamedField::new("peak_joint_jerk_at"),
    NamedField::new("peak_tcp_velocity_at"),
    NamedField::new("peak_tcp_acceleration_at"),
    NamedField::new("peak_tcp_jerk_at"),
    NamedField::new("constraint_counts"),
    NamedField::new("initial_guess"),
    NamedField::new("boundary_slack_usage"),
    NamedField::new("phase_timing"),
];

impl Valuable for Topp3Tcp6Diagnostic {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let solve_secs = duration_secs(self.solve_time);
        let total_secs = duration_secs(self.total_time);
        let limiting = match &self.limiting_constraint {
            Some(g) => g.as_value(),
            None => Value::Unit,
        };
        let limiting_sample = match self.limiting_sample {
            Some(s) => Value::U64(s as u64),
            None => Value::Unit,
        };
        let message = match &self.message {
            Some(s) => Value::String(s.as_str()),
            None => Value::Unit,
        };
        let values = [
            self.status.as_value(),
            self.iterations.as_value(),
            solve_secs.as_value(),
            self.solver_tolerance_used.as_value(),
            self.densified_samples.as_value(),
            self.output_samples.as_value(),
            total_secs.as_value(),
            self.peak_joint_velocity.as_value(),
            self.peak_joint_acceleration.as_value(),
            self.peak_joint_jerk.as_value(),
            self.peak_tcp_velocity.as_value(),
            self.peak_tcp_acceleration.as_value(),
            self.peak_tcp_jerk.as_value(),
            self.average_utilization.as_value(),
            self.boundary_projection_residual.as_value(),
            limiting,
            limiting_sample,
            message,
            self.path_stats.as_value(),
            self.derivative_stats.as_value(),
            self.tcp_stats.as_value(),
            self.peak_joint_velocity_at.as_value(),
            self.peak_joint_acceleration_at.as_value(),
            self.peak_joint_jerk_at.as_value(),
            self.peak_tcp_velocity_at.as_value(),
            self.peak_tcp_acceleration_at.as_value(),
            self.peak_tcp_jerk_at.as_value(),
            self.constraint_counts.as_value(),
            self.initial_guess.as_value(),
            self.boundary_slack_usage.as_value(),
            self.phase_timing.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(DIAG_FIELDS, &values));
    }
}

impl Structable for Topp3Tcp6Diagnostic {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("Topp3Tcp6Diagnostic", Fields::Named(DIAG_FIELDS))
    }
}

const PEAK_LOCATION_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("value"),
    NamedField::new("sample"),
    NamedField::new("joint"),
];

impl Valuable for PeakLocation {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let joint = match self.joint {
            Some(j) => Value::U64(j as u64),
            None => Value::Unit,
        };
        let values = [self.value.as_value(), self.sample.as_value(), joint];
        visit.visit_named_fields(&NamedValues::new(PEAK_LOCATION_FIELDS, &values));
    }
}

impl Structable for PeakLocation {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("PeakLocation", Fields::Named(PEAK_LOCATION_FIELDS))
    }
}

const PATH_STATS_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("input_waypoints"),
    NamedField::new("merged_waypoints"),
    NamedField::new("chord_length"),
    NamedField::new("min_segment_length"),
    NamedField::new("max_segment_length"),
    NamedField::new("segment_length_ratio"),
];

impl Valuable for PathStats {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.input_waypoints.as_value(),
            self.merged_waypoints.as_value(),
            self.chord_length.as_value(),
            self.min_segment_length.as_value(),
            self.max_segment_length.as_value(),
            self.segment_length_ratio.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(PATH_STATS_FIELDS, &values));
    }
}

impl Structable for PathStats {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("PathStats", Fields::Named(PATH_STATS_FIELDS))
    }
}

const DERIVATIVE_STATS_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("max_abs_qpp"),
    NamedField::new("max_abs_qpp_sample"),
    NamedField::new("max_abs_qpp_joint"),
    NamedField::new("max_abs_qppp"),
    NamedField::new("max_abs_qppp_sample"),
    NamedField::new("max_abs_qppp_joint"),
    NamedField::new("min_qp_norm_relative_sq"),
    NamedField::new("min_qp_norm_sample"),
    NamedField::new("degenerate_qp_samples"),
];

impl Valuable for DerivativeStats {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.max_abs_qpp.as_value(),
            self.max_abs_qpp_sample.as_value(),
            self.max_abs_qpp_joint.as_value(),
            self.max_abs_qppp.as_value(),
            self.max_abs_qppp_sample.as_value(),
            self.max_abs_qppp_joint.as_value(),
            self.min_qp_norm_relative_sq.as_value(),
            self.min_qp_norm_sample.as_value(),
            self.degenerate_qp_samples.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(DERIVATIVE_STATS_FIELDS, &values));
    }
}

impl Structable for DerivativeStats {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("DerivativeStats", Fields::Named(DERIVATIVE_STATS_FIELDS))
    }
}

const TCP_STATS_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("max_abs_pp"),
    NamedField::new("max_abs_ppp"),
    NamedField::new("max_abs_pppp"),
    NamedField::new("min_abs_pp_per_axis"),
    NamedField::new("max_abs_pp_per_axis"),
];

impl Valuable for TcpStats {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let min_axis = self.min_abs_pp_per_axis.as_slice();
        let max_axis = self.max_abs_pp_per_axis.as_slice();
        let values = [
            self.max_abs_pp.as_value(),
            self.max_abs_ppp.as_value(),
            self.max_abs_pppp.as_value(),
            min_axis.as_value(),
            max_axis.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(TCP_STATS_FIELDS, &values));
    }
}

impl Structable for TcpStats {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("TcpStats", Fields::Named(TCP_STATS_FIELDS))
    }
}

const CONSTRAINT_COUNTS_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("joint_v"),
    NamedField::new("joint_a"),
    NamedField::new("joint_j"),
    NamedField::new("tcp_v"),
    NamedField::new("tcp_a"),
    NamedField::new("tcp_j"),
];

impl Valuable for ConstraintCounts {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.joint_v.as_value(),
            self.joint_a.as_value(),
            self.joint_j.as_value(),
            self.tcp_v.as_value(),
            self.tcp_a.as_value(),
            self.tcp_j.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(CONSTRAINT_COUNTS_FIELDS, &values));
    }
}

impl Structable for ConstraintCounts {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("ConstraintCounts", Fields::Named(CONSTRAINT_COUNTS_FIELDS))
    }
}

const INITIAL_GUESS_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("end_sd_residual"),
    NamedField::new("end_sdd_residual"),
    NamedField::new("max_sddd"),
    NamedField::new("max_sddd_segment"),
];

impl Valuable for InitialGuessStats {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.end_sd_residual.as_value(),
            self.end_sdd_residual.as_value(),
            self.max_sddd.as_value(),
            self.max_sddd_segment.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(INITIAL_GUESS_FIELDS, &values));
    }
}

impl Structable for InitialGuessStats {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("InitialGuessStats", Fields::Named(INITIAL_GUESS_FIELDS))
    }
}

const BOUNDARY_SLACK_USAGE_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("start_sd"),
    NamedField::new("start_sdd"),
    NamedField::new("end_sd"),
    NamedField::new("end_sdd"),
];

impl Valuable for BoundarySlackUsage {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.start_sd.as_value(),
            self.start_sdd.as_value(),
            self.end_sd.as_value(),
            self.end_sdd.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(BOUNDARY_SLACK_USAGE_FIELDS, &values));
    }
}

impl Structable for BoundarySlackUsage {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static(
            "BoundarySlackUsage",
            Fields::Named(BOUNDARY_SLACK_USAGE_FIELDS),
        )
    }
}

const PHASE_TIMING_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("densify_secs"),
    NamedField::new("derivatives_secs"),
    NamedField::new("nlp_build_secs"),
    NamedField::new("nlp_solve_secs"),
    NamedField::new("resample_secs"),
];

impl Valuable for PhaseTiming {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let densify = duration_secs(self.densify);
        let derivatives = duration_secs(self.derivatives);
        let nlp_build = duration_secs(self.nlp_build);
        let nlp_solve = duration_secs(self.nlp_solve);
        let resample = duration_secs(self.resample);
        let values = [
            densify.as_value(),
            derivatives.as_value(),
            nlp_build.as_value(),
            nlp_solve.as_value(),
            resample.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(PHASE_TIMING_FIELDS, &values));
    }
}

impl Structable for PhaseTiming {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("PhaseTiming", Fields::Named(PHASE_TIMING_FIELDS))
    }
}

const JOINT_LIMITS_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("q_min"),
    NamedField::new("q_max"),
    NamedField::new("v_max"),
    NamedField::new("a_max"),
    NamedField::new("j_max"),
];

impl<const N: usize> Valuable for JointLimits<N> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.q_min.as_value(),
            self.q_max.as_value(),
            self.v_max.as_value(),
            self.a_max.as_value(),
            self.j_max.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(JOINT_LIMITS_FIELDS, &values));
    }
}

impl<const N: usize> Structable for JointLimits<N> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("JointLimits", Fields::Named(JOINT_LIMITS_FIELDS))
    }
}

const TCP_LIMITS_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("v_max"),
    NamedField::new("a_max"),
    NamedField::new("j_max"),
];

impl Valuable for TcpLimits {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.v_max.as_value(),
            self.a_max.as_value(),
            self.j_max.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(TCP_LIMITS_FIELDS, &values));
    }
}

impl Structable for TcpLimits {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("TcpLimits", Fields::Named(TCP_LIMITS_FIELDS))
    }
}

const BOUNDARY_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("v_start"),
    NamedField::new("a_start"),
    NamedField::new("v_end"),
    NamedField::new("a_end"),
    NamedField::new("projection_tolerance"),
];

impl<const N: usize> Valuable for BoundaryConditions<N> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.v_start.as_value(),
            self.a_start.as_value(),
            self.v_end.as_value(),
            self.a_end.as_value(),
            self.projection_tolerance.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(BOUNDARY_FIELDS, &values));
    }
}

impl<const N: usize> Structable for BoundaryConditions<N> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("BoundaryConditions", Fields::Named(BOUNDARY_FIELDS))
    }
}

const DENSIFICATION_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("max_segment_step"),
    NamedField::new("max_samples"),
    NamedField::new("min_samples"),
];

impl Valuable for DensificationOptions {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let max_step = match self.max_segment_step {
            Some(v) => Value::F64(v),
            None => Value::Unit,
        };
        let values = [
            max_step,
            self.max_samples.as_value(),
            self.min_samples.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(DENSIFICATION_FIELDS, &values));
    }
}

impl Structable for DensificationOptions {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("DensificationOptions", Fields::Named(DENSIFICATION_FIELDS))
    }
}

const SOLVER_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("tolerance"),
    NamedField::new("max_iterations"),
    NamedField::new("timeout_secs"),
    NamedField::new("diagnostics"),
];

impl Valuable for SolverOptions {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let timeout = match self.timeout {
            Some(d) => Value::F64(duration_secs(d)),
            None => Value::Unit,
        };
        let values = [
            self.tolerance.as_value(),
            self.max_iterations.as_value(),
            timeout,
            self.diagnostics.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(SOLVER_FIELDS, &values));
    }
}

impl Structable for SolverOptions {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("SolverOptions", Fields::Named(SOLVER_FIELDS))
    }
}

const CONSTRAINTS_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("joint"),
    NamedField::new("tcp"),
    NamedField::new("boundary"),
    NamedField::new("densification"),
    NamedField::new("solver"),
    NamedField::new("sample_rate_hz"),
    NamedField::new("locked_prefix"),
    NamedField::new("post_validation"),
    NamedField::new("check_output_dynamics"),
];

impl<const N: usize> Valuable for Topp3Tcp6Constraints<N> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let tcp_value = match &self.tcp {
            Some(t) => t.as_value(),
            None => Value::Unit,
        };
        let values = [
            self.joint.as_value(),
            tcp_value,
            self.boundary.as_value(),
            self.densification.as_value(),
            self.solver.as_value(),
            self.sample_rate_hz.as_value(),
            self.locked_prefix.as_value(),
            self.post_validation.as_value(),
            self.check_output_dynamics.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(CONSTRAINTS_FIELDS, &values));
    }
}

impl<const N: usize> Structable for Topp3Tcp6Constraints<N> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("Topp3Tcp6Constraints", Fields::Named(CONSTRAINTS_FIELDS))
    }
}

const PROJ_BOUNDARY_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("sd"),
    NamedField::new("sdd"),
    NamedField::new("velocity_residual"),
    NamedField::new("acceleration_residual"),
];

impl Valuable for ProjectedBoundary {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.sd.as_value(),
            self.sdd.as_value(),
            self.velocity_residual.as_value(),
            self.acceleration_residual.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(PROJ_BOUNDARY_FIELDS, &values));
    }
}

impl Structable for ProjectedBoundary {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("ProjectedBoundary", Fields::Named(PROJ_BOUNDARY_FIELDS))
    }
}
