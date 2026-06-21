use deke_types::SRobotPath;

use crate::error::{MultipathError, MultipathResult};

/// A path that must be traversed exactly once, with the directions or variants
/// the solver is free to choose between. Every variant collapses to a set of
/// directed option paths; the solver picks exactly one option per `ReqPath`.
pub enum ReqPath<const N: usize> {
    /// Must be traversed in the given direction.
    OneWay(SRobotPath<N, f64>),
    /// May be traversed forwards or backwards; the solver picks the cheaper
    /// orientation. The reverse is the waypoint-reversed path.
    Reversible(SRobotPath<N, f64>),
    /// Forwards and backwards are *distinct* pre-built realizations (e.g. a
    /// different joint resolution for the return direction). The solver picks
    /// one.
    BothWays(SRobotPath<N, f64>, SRobotPath<N, f64>),
    /// Arbitrary alternative realizations of the same required motion (e.g.
    /// redundant linear-axis variants). Must be non-empty.
    ManyWays(Vec<SRobotPath<N, f64>>),
}

/// One directed candidate for a required path. Its start is `path.first()` and
/// its end is `path.last()`. `cluster` ties it back to the `ReqPath` it came
/// from so the solver visits each required path exactly once.
pub(crate) struct DirectedOption<const N: usize> {
    pub path: SRobotPath<N, f64>,
    pub cluster: usize,
}

/// Flatten the required paths into directed options tagged by cluster index.
/// Returns the options and the cluster count. Errors if any `ManyWays` cluster
/// is empty.
pub(crate) fn expand<const N: usize>(
    req_paths: &[ReqPath<N>],
) -> MultipathResult<(Vec<DirectedOption<N>>, usize)> {
    let mut options = Vec::new();
    for (cluster, req) in req_paths.iter().enumerate() {
        let before = options.len();
        match req {
            ReqPath::OneWay(p) => options.push(DirectedOption {
                path: p.clone(),
                cluster,
            }),
            ReqPath::Reversible(p) => {
                options.push(DirectedOption {
                    path: p.clone(),
                    cluster,
                });
                options.push(DirectedOption {
                    path: p.reversed(),
                    cluster,
                });
            }
            ReqPath::BothWays(a, b) => {
                options.push(DirectedOption {
                    path: a.clone(),
                    cluster,
                });
                options.push(DirectedOption {
                    path: b.clone(),
                    cluster,
                });
            }
            ReqPath::ManyWays(ps) => {
                for p in ps {
                    options.push(DirectedOption {
                        path: p.clone(),
                        cluster,
                    });
                }
            }
        }
        if options.len() == before {
            return Err(MultipathError::EmptyOptions(cluster));
        }
    }
    Ok((options, req_paths.len()))
}
