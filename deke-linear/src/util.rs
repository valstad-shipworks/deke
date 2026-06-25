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

/// Fritsch–Carlson monotone cubic (PCHIP) evaluation of `vals` at `x` against a
/// sorted ascending `grid`. The interpolant passes through every knot, stays
/// monotone on monotone data, and never overshoots the sample envelope — so a
/// smoothed schedule built from it inherits the DP track's bounds. `grid` and
/// `vals` are parallel and non-empty; `x` is clamped to the grid range.
pub(crate) fn pchip(grid: &[f64], vals: &[f64], x: f64) -> f64 {
    let n = grid.len();
    if n == 1 {
        return vals[0];
    }
    let x = x.clamp(grid[0], grid[n - 1]);

    let mut delta = vec![0.0f64; n - 1];
    let mut h = vec![0.0f64; n - 1];
    for k in 0..n - 1 {
        h[k] = (grid[k + 1] - grid[k]).max(1e-15);
        delta[k] = (vals[k + 1] - vals[k]) / h[k];
    }

    let mut m = vec![0.0f64; n];
    if n == 2 {
        m[0] = delta[0];
        m[1] = delta[0];
    } else {
        m[0] = endpoint_slope(h[1], h[0], delta[0], delta[1]);
        m[n - 1] = endpoint_slope(h[n - 3], h[n - 2], delta[n - 2], delta[n - 3]);
        for k in 1..n - 1 {
            let (d0, d1) = (delta[k - 1], delta[k]);
            if d0 * d1 <= 0.0 {
                m[k] = 0.0;
            } else {
                let w1 = 2.0 * h[k] + h[k - 1];
                let w2 = h[k] + 2.0 * h[k - 1];
                m[k] = (w1 + w2) / (w1 / d0 + w2 / d1);
            }
        }
    }

    let i = match grid.binary_search_by(|v| v.total_cmp(&x)) {
        Ok(j) => return vals[j],
        Err(j) => (j.max(1) - 1).min(n - 2),
    };
    let t = (x - grid[i]) / h[i];
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * vals[i] + h10 * h[i] * m[i] + h01 * vals[i + 1] + h11 * h[i] * m[i + 1]
}

/// One-sided endpoint slope with the standard Fritsch–Carlson shape-preserving
/// correction: a non-shape-preserving extrapolation is clamped to zero or to
/// three times the boundary secant.
fn endpoint_slope(h_far: f64, h_near: f64, d_near: f64, d_far: f64) -> f64 {
    let m = ((2.0 * h_near + h_far) * d_near - h_near * d_far) / (h_near + h_far);
    if m * d_near <= 0.0 {
        0.0
    } else if d_near * d_far <= 0.0 && m.abs() > 3.0 * d_near.abs() {
        3.0 * d_near
    } else {
        m
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

#[cfg(test)]
mod tests {
    use super::pchip;

    #[test]
    fn pchip_passes_through_knots() {
        let grid = [0.0, 1.0, 2.0, 3.0];
        let vals = [0.0, 0.5, 0.7, 2.0];
        for (g, v) in grid.iter().zip(vals.iter()) {
            assert!((pchip(&grid, &vals, *g) - v).abs() < 1e-12);
        }
    }

    #[test]
    fn pchip_no_overshoot_on_monotone_data() {
        let grid = [0.0, 1.0, 2.0, 3.0, 4.0];
        let vals = [0.0, 0.0, 0.0, 1.0, 1.0];
        let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut prev = pchip(&grid, &vals, 0.0);
        for i in 0..=400 {
            let x = 4.0 * i as f64 / 400.0;
            let y = pchip(&grid, &vals, x);
            assert!(y >= lo - 1e-12 && y <= hi + 1e-12, "overshoot at {x}: {y}");
            assert!(y >= prev - 1e-9, "non-monotone at {x}");
            prev = y;
        }
    }

    #[test]
    fn pchip_single_point() {
        assert_eq!(pchip(&[2.0], &[5.0], 9.0), 5.0);
    }
}
