//! Small numeric helpers shared across the stages.

/// Piecewise-linear lookup of `vals` against a sorted `grid`, clamped to the
/// grid's range. `grid` and `vals` are parallel and `grid` is non-empty and
/// ascending.
pub(crate) fn interp(grid: &[f64], vals: &[f64], x: f64) -> f64 {
    let x = x.clamp(grid[0], grid[grid.len() - 1]);
    match grid.binary_search_by(|v| v.total_cmp(&x)) {
        Ok(i) => vals[i],
        Err(i) => {
            if i == 0 {
                return vals[0];
            }
            if i >= grid.len() {
                return vals[grid.len() - 1];
            }
            let (g0, g1) = (grid[i - 1], grid[i]);
            let f = if g1 > g0 { (x - g0) / (g1 - g0) } else { 0.0 };
            vals[i - 1] + (vals[i] - vals[i - 1]) * f
        }
    }
}

/// Forward shortest path over a layered graph: choose one node per layer that
/// minimises the cumulative node + edge cost, skipping infeasible edges.
///
/// `node_cost(layer, idx)` is the node weight (an infinite weight prunes the
/// node); `edge(layer, prev_idx, idx)` returns the cost of the edge from
/// `prev_idx` in `layer - 1` to `idx` in `layer`, or `None` when that transition
/// is forbidden. Returns the chosen node index per layer and the total cost, or
/// `None` if no path reaches the final layer.
pub(crate) fn ladder_dp(
    layer_sizes: &[usize],
    node_cost: impl Fn(usize, usize) -> f64,
    edge: impl Fn(usize, usize, usize) -> Option<f64>,
) -> Option<(Vec<usize>, f64)> {
    let n = layer_sizes.len();
    if n == 0 {
        return Some((Vec::new(), 0.0));
    }

    let mut dp: Vec<Vec<(f64, usize)>> = Vec::with_capacity(n);
    dp.push(
        (0..layer_sizes[0])
            .map(|i| (node_cost(0, i), usize::MAX))
            .collect(),
    );
    for k in 1..n {
        let mut row = Vec::with_capacity(layer_sizes[k]);
        for ci in 0..layer_sizes[k] {
            let nc = node_cost(k, ci);
            let mut best = (f64::INFINITY, usize::MAX);
            for (pi, prev_cell) in dp[k - 1].iter().enumerate() {
                let prev = prev_cell.0;
                if !prev.is_finite() {
                    continue;
                }
                if let Some(e) = edge(k, pi, ci) {
                    let cand = prev + nc + e;
                    if cand < best.0 {
                        best = (cand, pi);
                    }
                }
            }
            row.push(best);
        }
        dp.push(row);
    }

    let last = n - 1;
    let (mut bi, total) = dp[last]
        .iter()
        .enumerate()
        .filter(|(_, e)| e.0.is_finite())
        .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
        .map(|(i, e)| (i, e.0))?;

    let mut chosen = vec![0usize; n];
    for k in (0..n).rev() {
        chosen[k] = bi;
        if k > 0 {
            bi = dp[k][bi].1;
            if bi == usize::MAX {
                return None;
            }
        }
    }
    Some((chosen, total))
}
