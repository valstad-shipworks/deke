use deke_types::{SRobotPath, SRobotQ};

use crate::reqpath::DirectedOption;

/// How a transition between two configurations is scored when ordering the
/// required paths. This drives the AGTSP only — the connector paths in the
/// output are produced separately (by the planner or by straight-line
/// segments).
pub enum TransitionCost<'a, const N: usize> {
    /// Weighted joint-space straight-line distance: `sqrt(Σ (wᵢ·Δqᵢ)²)`.
    JointWeighted(SRobotQ<N, f64>),
    /// Caller-supplied cost from `from` to `to`. Wrap a planner here for
    /// obstacle-aware ordering, at the price of one plan per scored transition.
    Custom(&'a dyn Fn(&SRobotQ<N, f64>, &SRobotQ<N, f64>) -> f64),
}

impl<const N: usize> TransitionCost<'_, N> {
    pub(crate) fn eval(&self, from: &SRobotQ<N, f64>, to: &SRobotQ<N, f64>) -> f64 {
        match self {
            TransitionCost::JointWeighted(w) => weighted_distance(from, to, w),
            TransitionCost::Custom(f) => f(from, to),
        }
    }
}

/// Weighted joint-space distance `sqrt(Σ (wᵢ·(aᵢ-bᵢ))²)`. Note this is *not*
/// `SRobotQ::distance`, which is unweighted.
pub fn weighted_distance<const N: usize>(
    a: &SRobotQ<N, f64>,
    b: &SRobotQ<N, f64>,
    w: &SRobotQ<N, f64>,
) -> f64 {
    w.0.iter()
        .zip(a.0.iter())
        .zip(b.0.iter())
        .map(|((&wi, &ai), &bi)| {
            let d = wi * (ai - bi);
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Cost of traversing a path under `cost`: the sum over its segments. For the
/// weighted-joint model this is the weighted arc length.
fn traversal_cost<const N: usize>(path: &SRobotPath<N, f64>, cost: &TransitionCost<N>) -> f64 {
    path.segments().map(|(a, b)| cost.eval(a, b)).sum()
}

/// The cost matrices the AGTSP consumes. `transition` is a row-major
/// `options × options` matrix — contiguous rather than a `Vec<Vec<_>>` so the
/// Held–Karp inner loop, which is cache-bound on these reads, avoids a pointer
/// chase per access. Entry `(i, j)` already folds in option `j`'s own traversal
/// cost, so the solver prefers the cheaper realization, not just the cheaper
/// connector.
pub(crate) struct CostMatrices {
    /// Row-major `options × options`: `(i, j)` = move from option `i`'s end to
    /// option `j`'s start, then traverse `j`.
    pub transition: Vec<f64>,
    /// `start[i]` = move from the global start to option `i`'s start, then
    /// traverse `i`.
    pub start: Vec<f64>,
    /// `end[i]` = move from option `i`'s end to the global end (`0.0` when no
    /// end is requested).
    pub end: Vec<f64>,
}

pub(crate) fn build_matrices<const N: usize>(
    options: &[DirectedOption<N>],
    cost: &TransitionCost<N>,
    start_q: &SRobotQ<N, f64>,
    end_q: Option<&SRobotQ<N, f64>>,
) -> CostMatrices {
    let m = options.len();
    let traversal: Vec<f64> = options.iter().map(|o| traversal_cost(&o.path, cost)).collect();

    let mut transition = vec![0.0_f64; m * m];
    for (i, oi) in options.iter().enumerate() {
        let from = oi.path.last();
        let row = &mut transition[i * m..(i + 1) * m];
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = cost.eval(from, options[j].path.first()) + traversal[j];
        }
    }

    let start = options
        .iter()
        .enumerate()
        .map(|(i, o)| cost.eval(start_q, o.path.first()) + traversal[i])
        .collect();

    let end = options
        .iter()
        .map(|o| end_q.map_or(0.0, |e| cost.eval(o.path.last(), e)))
        .collect();

    CostMatrices { transition, start, end }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q<const N: usize>(a: [f64; N]) -> SRobotQ<N, f64> {
        SRobotQ(a)
    }

    #[test]
    fn weighted_distance_applies_weights() {
        let origin = q([0.0, 0.0]);
        let w = q([2.0, 1.0]);
        // Moving the high-weight joint costs twice the low-weight joint.
        assert!((weighted_distance(&origin, &q([1.0, 0.0]), &w) - 2.0).abs() < 1e-12);
        assert!((weighted_distance(&origin, &q([0.0, 1.0]), &w) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn traversal_sums_weighted_segments() {
        let path =
            SRobotPath::<2, f64>::try_new(vec![q([0.0, 0.0]), q([1.0, 0.0]), q([1.0, 1.0])])
                .unwrap();
        let cost = TransitionCost::JointWeighted(q([2.0, 1.0]));
        // 2·1 (joint 0) + 1·1 (joint 1) = 3.
        assert!((traversal_cost(&path, &cost) - 3.0).abs() < 1e-12);
    }
}
