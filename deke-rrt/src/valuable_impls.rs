#![cfg(feature = "valuable")]
//! `valuable::Valuable` impls for diagnostics, settings, and limit types.

use ::valuable::{
    EnumDef, Enumerable, Fields, NamedField, NamedValues, StructDef, Structable, Valuable, Value,
    Variant, VariantDef, Visit,
};

use crate::{
    AnytimeInfo, AorrtcSettings, ExtensionStats, JointKinLimits, KinematicLimits, KrrtcSettings,
    RandomizerType, RrtDiagnostic, RrtTermination, RrtcSettings,
};

const TERMINATION_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("NotStarted", Fields::Unnamed(0)),
    VariantDef::new("DegenerateStartGoal", Fields::Unnamed(0)),
    VariantDef::new("DirectConnection", Fields::Unnamed(0)),
    VariantDef::new("Solved", Fields::Unnamed(0)),
    VariantDef::new("MaxIterationsExceeded", Fields::Unnamed(0)),
    VariantDef::new("MaxSamplesExceeded", Fields::Unnamed(0)),
    VariantDef::new("Stalled", Fields::Unnamed(0)),
    VariantDef::new("OptimalReached", Fields::Unnamed(0)),
    VariantDef::new("InputInvalid", Fields::Unnamed(0)),
    VariantDef::new("NoInitialPath", Fields::Unnamed(0)),
];

impl Valuable for RrtTermination {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        visit.visit_unnamed_fields(&[]);
    }
}

impl Enumerable for RrtTermination {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("RrtTermination", TERMINATION_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            Self::NotStarted => 0,
            Self::DegenerateStartGoal => 1,
            Self::DirectConnection => 2,
            Self::Solved => 3,
            Self::MaxIterationsExceeded => 4,
            Self::MaxSamplesExceeded => 5,
            Self::Stalled => 6,
            Self::OptimalReached => 7,
            Self::InputInvalid => 8,
            Self::NoInitialPath => 9,
        };
        Variant::Static(&TERMINATION_VARIANTS[idx])
    }
}

const EXTENSION_STATS_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("extension_attempts"),
    NamedField::new("dynamic_domain_rejections"),
    NamedField::new("edge_validations"),
    NamedField::new("edge_validation_failures"),
    NamedField::new("successful_extensions"),
    NamedField::new("connect_attempts"),
    NamedField::new("connect_successes"),
];

impl Valuable for ExtensionStats {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.extension_attempts.as_value(),
            self.dynamic_domain_rejections.as_value(),
            self.edge_validations.as_value(),
            self.edge_validation_failures.as_value(),
            self.successful_extensions.as_value(),
            self.connect_attempts.as_value(),
            self.connect_successes.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(EXTENSION_STATS_FIELDS, &values));
    }
}

impl Structable for ExtensionStats {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("ExtensionStats", Fields::Named(EXTENSION_STATS_FIELDS))
    }
}

const ANYTIME_INFO_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("initial_cost"),
    NamedField::new("initial_iterations"),
    NamedField::new("improvements"),
    NamedField::new("iters_since_last_improvement"),
    NamedField::new("optimality_ratio"),
];

impl Valuable for AnytimeInfo {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.initial_cost.as_value(),
            self.initial_iterations.as_value(),
            self.improvements.as_value(),
            self.iters_since_last_improvement.as_value(),
            self.optimality_ratio.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(ANYTIME_INFO_FIELDS, &values));
    }
}

impl Structable for AnytimeInfo {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("AnytimeInfo", Fields::Named(ANYTIME_INFO_FIELDS))
    }
}

const RRT_DIAG_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("iterations"),
    NamedField::new("start_tree_size"),
    NamedField::new("goal_tree_size"),
    NamedField::new("path_cost"),
    NamedField::new("elapsed_ns"),
    NamedField::new("termination"),
    NamedField::new("extension_stats"),
    NamedField::new("c_min"),
    NamedField::new("closest_approach"),
    NamedField::new("anytime"),
];

impl Valuable for RrtDiagnostic {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let anytime_value = match &self.anytime {
            Some(a) => a.as_value(),
            None => Value::Unit,
        };
        let values = [
            self.iterations.as_value(),
            self.start_tree_size.as_value(),
            self.goal_tree_size.as_value(),
            self.path_cost.as_value(),
            self.elapsed_ns.as_value(),
            self.termination.as_value(),
            self.extension_stats.as_value(),
            self.c_min.as_value(),
            self.closest_approach.as_value(),
            anytime_value,
        ];
        visit.visit_named_fields(&NamedValues::new(RRT_DIAG_FIELDS, &values));
    }
}

impl Structable for RrtDiagnostic {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("RrtDiagnostic", Fields::Named(RRT_DIAG_FIELDS))
    }
}

const RANDOMIZER_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("Wyrand", Fields::Unnamed(0)),
    VariantDef::new("SplitMix", Fields::Unnamed(0)),
    VariantDef::new("Xorshift", Fields::Unnamed(0)),
    VariantDef::new("Halton", Fields::Unnamed(0)),
];

impl Valuable for RandomizerType {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        visit.visit_unnamed_fields(&[]);
    }
}

impl Enumerable for RandomizerType {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("RandomizerType", RANDOMIZER_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            RandomizerType::Wyrand => 0,
            RandomizerType::SplitMix => 1,
            RandomizerType::Xorshift => 2,
            RandomizerType::Halton => 3,
        };
        Variant::Static(&RANDOMIZER_VARIANTS[idx])
    }
}

const JOINT_KIN_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("v_max"),
    NamedField::new("a_max"),
    NamedField::new("j_max"),
];

impl Valuable for JointKinLimits {
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
        visit.visit_named_fields(&NamedValues::new(JOINT_KIN_FIELDS, &values));
    }
}

impl Structable for JointKinLimits {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("JointKinLimits", Fields::Named(JOINT_KIN_FIELDS))
    }
}

const KIN_LIMITS_FIELDS: &[NamedField<'static>] = &[NamedField::new("joints")];

impl<const N: usize> Valuable for KinematicLimits<N> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let joints: &[JointKinLimits] = &self.joints[..];
        let values = [joints.as_value()];
        visit.visit_named_fields(&NamedValues::new(KIN_LIMITS_FIELDS, &values));
    }
}

impl<const N: usize> Structable for KinematicLimits<N> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("KinematicLimits", Fields::Named(KIN_LIMITS_FIELDS))
    }
}

const RRTC_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("range"),
    NamedField::new("max_iterations"),
    NamedField::new("max_samples"),
    NamedField::new("joint_lower"),
    NamedField::new("joint_upper"),
    NamedField::new("dof_cost_weights"),
    NamedField::new("resolution"),
    NamedField::new("dynamic_domain"),
    NamedField::new("radius"),
    NamedField::new("alpha"),
    NamedField::new("min_radius"),
    NamedField::new("balance"),
    NamedField::new("tree_ratio"),
    NamedField::new("seed"),
    NamedField::new("randomizer"),
    NamedField::new("shortcut"),
    NamedField::new("bspline_steps"),
    NamedField::new("bspline_midpoint_interpolation"),
    NamedField::new("bspline_min_change"),
    NamedField::new("reduce_max_steps"),
    NamedField::new("reduce_range_ratio"),
];

impl<const N: usize> Valuable for RrtcSettings<N> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.range.as_value(),
            self.max_iterations.as_value(),
            self.max_samples.as_value(),
            self.joint_lower.as_value(),
            self.joint_upper.as_value(),
            self.dof_cost_weights.as_value(),
            self.resolution.as_value(),
            self.dynamic_domain.as_value(),
            self.radius.as_value(),
            self.alpha.as_value(),
            self.min_radius.as_value(),
            self.balance.as_value(),
            self.tree_ratio.as_value(),
            self.seed.as_value(),
            self.randomizer.as_value(),
            self.shortcut.as_value(),
            self.bspline_steps.as_value(),
            self.bspline_midpoint_interpolation.as_value(),
            self.bspline_min_change.as_value(),
            self.reduce_max_steps.as_value(),
            self.reduce_range_ratio.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(RRTC_FIELDS, &values));
    }
}

impl<const N: usize> Structable for RrtcSettings<N> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("RrtcSettings", Fields::Named(RRTC_FIELDS))
    }
}

const AORRTC_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("rrtc"),
    NamedField::new("max_iterations"),
    NamedField::new("max_samples"),
    NamedField::new("use_phs"),
    NamedField::new("cost_bound_resamples"),
    NamedField::new("stall_iterations"),
    NamedField::new("dof_cost_weights"),
    NamedField::new("penalize_static_dof"),
    NamedField::new("static_dof_penalty"),
    NamedField::new("static_dof_threshold"),
    NamedField::new("aux_randomizer"),
    NamedField::new("aux_seed"),
    NamedField::new("simplify_shortcut"),
    NamedField::new("simplify_bspline_steps"),
    NamedField::new("simplify_bspline_midpoint_interpolation"),
    NamedField::new("simplify_bspline_min_change"),
    NamedField::new("simplify_reduce_max_steps"),
    NamedField::new("simplify_reduce_range_ratio"),
];

impl<const N: usize> Valuable for AorrtcSettings<N> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.rrtc.as_value(),
            self.max_iterations.as_value(),
            self.max_samples.as_value(),
            self.use_phs.as_value(),
            self.cost_bound_resamples.as_value(),
            self.stall_iterations.as_value(),
            self.dof_cost_weights.as_value(),
            self.penalize_static_dof.as_value(),
            self.static_dof_penalty.as_value(),
            self.static_dof_threshold.as_value(),
            self.aux_randomizer.as_value(),
            self.aux_seed.as_value(),
            self.simplify_shortcut.as_value(),
            self.simplify_bspline_steps.as_value(),
            self.simplify_bspline_midpoint_interpolation.as_value(),
            self.simplify_bspline_min_change.as_value(),
            self.simplify_reduce_max_steps.as_value(),
            self.simplify_reduce_range_ratio.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(AORRTC_FIELDS, &values));
    }
}

impl<const N: usize> Structable for AorrtcSettings<N> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("AorrtcSettings", Fields::Named(AORRTC_FIELDS))
    }
}

const KRRTC_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("range"),
    NamedField::new("max_iterations"),
    NamedField::new("max_samples"),
    NamedField::new("joint_lower"),
    NamedField::new("joint_upper"),
    NamedField::new("kin_limits"),
    NamedField::new("resolution"),
    NamedField::new("dynamic_domain"),
    NamedField::new("radius"),
    NamedField::new("alpha"),
    NamedField::new("min_radius"),
    NamedField::new("balance"),
    NamedField::new("tree_ratio"),
    NamedField::new("seed"),
    NamedField::new("randomizer"),
    NamedField::new("shortcut_iterations"),
    NamedField::new("smoothing_iterations"),
];

impl<const N: usize> Valuable for KrrtcSettings<N> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [
            self.range.as_value(),
            self.max_iterations.as_value(),
            self.max_samples.as_value(),
            self.joint_lower.as_value(),
            self.joint_upper.as_value(),
            self.kin_limits.as_value(),
            self.resolution.as_value(),
            self.dynamic_domain.as_value(),
            self.radius.as_value(),
            self.alpha.as_value(),
            self.min_radius.as_value(),
            self.balance.as_value(),
            self.tree_ratio.as_value(),
            self.seed.as_value(),
            self.randomizer.as_value(),
            self.shortcut_iterations.as_value(),
            self.smoothing_iterations.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(KRRTC_FIELDS, &values));
    }
}

impl<const N: usize> Structable for KrrtcSettings<N> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("KrrtcSettings", Fields::Named(KRRTC_FIELDS))
    }
}
