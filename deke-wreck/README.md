# deke-wreck

Collision validation for serial chains, backed by the `wreck` library.
Implements `deke_types::Validator`, so it drops into `deke-rrt` and anywhere
else the validator interface is consumed.

Each link, the end effector, and the base carry `wreck` colliders. On every
check the colliders are transformed into the world frame from the chain's FK and
tested for self-collision and against static or dynamic environment geometry. A
collision filter suppresses adjacent-link and other allowed pairs. Per-thread
scratch buffers are reused across calls to keep validation allocation-free on the
hot path.

## Example

```rust
use deke_wreck::{CollisionBody, WreckValidator, WreckValidatorContext};
use deke_types::Validator;

// One CollisionBody per link plus EE; optional base and static world bodies.
let validator = WreckValidator::new(links, ee, base, world, chain);
let ctx = WreckValidatorContext::new(&environment);
let ret = validator.validate(&q, &ctx);
```

`DynamicWreckValidator` handles moving environment bodies whose transforms
change between checks.

## Features

- `valuable`: derive `valuable::Valuable` for structured logging.

## License

Apache-2.0
