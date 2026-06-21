use deke_types::DekeError;

/// Failure modes specific to multipath solving, layered over [`DekeError`] for
/// the underlying planning / validation / path-construction failures that flow
/// up from the deke primitives.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MultipathError {
    /// A required path was declared with zero options to choose from (an empty
    /// `ManyWays`), making the tour infeasible — there is nothing to traverse
    /// for that cluster.
    #[error("required path #{0} has no options")]
    EmptyOptions(usize),

    /// The solver could not find any ordering that visits every required path,
    /// e.g. because a supplied cost function reported every relevant transition
    /// as non-finite.
    #[error("no feasible ordering exists through the required paths")]
    NoFeasibleTour,

    /// An underlying deke operation failed: planning a connector, validating a
    /// straight-line connector, or constructing a path.
    #[error(transparent)]
    Deke(#[from] DekeError),
}

pub type MultipathResult<T> = Result<T, MultipathError>;
