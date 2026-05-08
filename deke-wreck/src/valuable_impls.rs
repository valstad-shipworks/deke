#![cfg(feature = "valuable")]
//! `valuable::Valuable` impls for the wreck-validator context and the small
//! POD types it surfaces.

use ::valuable::{
    Fields, NamedField, NamedValues, StructDef, Structable, Valuable, Value, Visit,
};

use crate::{CollisionFilter, WreckValidatorContext};

// ---------------------------------------------------------------------------
// CollisionFilter<N>
// ---------------------------------------------------------------------------

const FILTER_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("links"),
    NamedField::new("ee"),
    NamedField::new("base"),
    NamedField::new("obstacles"),
];

impl<const N: usize> Valuable for CollisionFilter<N> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let links: &[bool] = &self.links[..];
        let values = [
            links.as_value(),
            self.ee.as_value(),
            self.base.as_value(),
            self.obstacles.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(FILTER_FIELDS, &values));
    }
}

impl<const N: usize> Structable for CollisionFilter<N> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("CollisionFilter", Fields::Named(FILTER_FIELDS))
    }
}

// ---------------------------------------------------------------------------
// WreckValidatorContext<'a, N>
//
// `extra_attachments` and `environment` are surfaced as their counts/markers
// rather than full structures: `Attachment` and `Collider` are large composites
// (uuids, vec-of-colliders, etc.) that don't have `Valuable` impls and would
// pull this crate in deeper than warranted. Inspectors that need full detail
// can drill into `wreck::Collider` directly via its own `valuable` feature.
// ---------------------------------------------------------------------------

const CONTEXT_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("num_extra_attachments"),
    NamedField::new("self_collisions"),
];

impl<'a, const N: usize> Valuable for WreckValidatorContext<'a, N> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let num_extras: usize = self.extra_attachments.len();
        let values = [num_extras.as_value(), self.self_collisions.as_value()];
        visit.visit_named_fields(&NamedValues::new(CONTEXT_FIELDS, &values));
    }
}

impl<'a, const N: usize> Structable for WreckValidatorContext<'a, N> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("WreckValidatorContext", Fields::Named(CONTEXT_FIELDS))
    }
}
