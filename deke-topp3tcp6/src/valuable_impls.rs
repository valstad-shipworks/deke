#![cfg(feature = "valuable")]
//! `valuable::Valuable` impls for the public diagnostics, settings, and
//! constraint types.

use std::time::Duration;

use ::valuable::{
    EnumDef, Enumerable, Fields, NamedField, NamedValues, StructDef, Structable, Valuable,
    Value, Variant, VariantDef, Visit,
};

use crate::{
    BoundaryConditions, DensificationOptions, JointLimits, LimitingGroup, SolveStatus,
    SolverOptions, TcpLimits, Topp3Tcp6Constraints, Topp3Tcp6Diagnostic,
    boundary::ProjectedBoundary,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Renders a `Duration` as its `f64` second count for inspection purposes.
fn duration_secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

// ---------------------------------------------------------------------------
// LimitingGroup
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// SolveStatus
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Topp3Tcp6Diagnostic
// ---------------------------------------------------------------------------

const DIAG_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("status"),
    NamedField::new("iterations"),
    NamedField::new("solve_time_secs"),
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
    NamedField::new("message"),
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
        let message = match &self.message {
            Some(s) => Value::String(s.as_str()),
            None => Value::Unit,
        };
        let values = [
            self.status.as_value(),
            self.iterations.as_value(),
            solve_secs.as_value(),
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
            message,
        ];
        visit.visit_named_fields(&NamedValues::new(DIAG_FIELDS, &values));
    }
}

impl Structable for Topp3Tcp6Diagnostic {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("Topp3Tcp6Diagnostic", Fields::Named(DIAG_FIELDS))
    }
}

// ---------------------------------------------------------------------------
// JointLimits<N>
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// TcpLimits
// ---------------------------------------------------------------------------

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
        let values = [self.v_max.as_value(), self.a_max.as_value(), self.j_max.as_value()];
        visit.visit_named_fields(&NamedValues::new(TCP_LIMITS_FIELDS, &values));
    }
}

impl Structable for TcpLimits {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("TcpLimits", Fields::Named(TCP_LIMITS_FIELDS))
    }
}

// ---------------------------------------------------------------------------
// BoundaryConditions<N>
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// DensificationOptions
// ---------------------------------------------------------------------------

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
        let values = [max_step, self.max_samples.as_value(), self.min_samples.as_value()];
        visit.visit_named_fields(&NamedValues::new(DENSIFICATION_FIELDS, &values));
    }
}

impl Structable for DensificationOptions {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("DensificationOptions", Fields::Named(DENSIFICATION_FIELDS))
    }
}

// ---------------------------------------------------------------------------
// SolverOptions
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Topp3Tcp6Constraints<N>
// ---------------------------------------------------------------------------

const CONSTRAINTS_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("joint"),
    NamedField::new("tcp"),
    NamedField::new("boundary"),
    NamedField::new("densification"),
    NamedField::new("solver"),
    NamedField::new("sample_rate_hz"),
    NamedField::new("locked_prefix"),
    NamedField::new("post_validation"),
];

impl<const N: usize> Valuable for Topp3Tcp6Constraints<N> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.joint.as_value(),
            self.tcp.as_value(),
            self.boundary.as_value(),
            self.densification.as_value(),
            self.solver.as_value(),
            self.sample_rate_hz.as_value(),
            self.locked_prefix.as_value(),
            self.post_validation.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(CONSTRAINTS_FIELDS, &values));
    }
}

impl<const N: usize> Structable for Topp3Tcp6Constraints<N> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("Topp3Tcp6Constraints", Fields::Named(CONSTRAINTS_FIELDS))
    }
}

// ---------------------------------------------------------------------------
// ProjectedBoundary
// ---------------------------------------------------------------------------

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
