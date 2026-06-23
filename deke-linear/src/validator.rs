use deke_types::{DekeError, DekeResult, SRobotQ, SRobotQLike, Validator};

/// A validator that accepts everything — for callers that handle collision
/// checking elsewhere (or not at all).
#[derive(Debug, Clone, Default)]
pub struct NoopValidator<const N: usize>;

impl<const N: usize> Validator<N, (), f64> for NoopValidator<N> {
    type Context<'ctx> = ();

    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<N, E, f64>>(
        &self,
        _q: A,
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        Ok(())
    }

    fn validate_motion<'ctx>(
        &self,
        _qs: &[SRobotQ<N, f64>],
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        Ok(())
    }
}
