//! Asymmetric generalized TSP over a precomputed cost matrix.
//!
//! Each "option" belongs to a cluster; a feasible tour selects exactly one
//! option per cluster and visits every cluster once. The tour is open (it has a
//! distinct start and end, biased by the `start` / `end` cost vectors), which
//! matches a robot moving from a start pose, through every required path, to an
//! optional end pose.
//!
//! Exact solving is Held–Karp bitmask DP keyed on `(visited_cluster_mask,
//! last_option)`; above a cell budget (or past the 32-cluster bitmask cap) it
//! falls back to a two-level heuristic that separates *cluster ordering* from
//! *option selection*:
//!
//!  1. order the clusters cheaply with a per-cluster distance surrogate
//!     (nearest-neighbour + 2-opt + Or-opt), then
//!  2. given that fixed cluster order, choose the optimal one-option-per-cluster
//!     assignment exactly via a layered-DAG shortest path ("cluster
//!     optimization") in `O(clusters · options²)`.
//!
//! This is the decomposition used by RoboTSP (Suárez-Ruiz & Pham, 2017) and the
//! cluster-optimization step of Karapetyan & Gutin's GTSP local search (2010) —
//! a cheap surrogate orders the clusters, an exact shortest path picks the
//! options. See `CITATIONS.md` (deke-multipath).
//!
//! The matrices are pure data, so this module is independent of any planner or
//! validator and is unit-tested with hand-built arrays.

use std::cmp::Ordering;

pub(crate) struct Solution {
    /// Option indices in traversal order, one per cluster.
    pub order: Vec<usize>,
    pub cost: f64,
}

pub(crate) struct Problem<'a> {
    /// `cluster_ids[i]` is the cluster of option `i`.
    pub cluster_ids: &'a [usize],
    pub n_clusters: usize,
    /// Row-major `options × options` transition costs (option `i` → option `j`,
    /// already including `j`'s traversal). Read via [`Problem::trans`].
    pub transition: &'a [f64],
    /// Row stride of `transition`, equal to the option count.
    pub options: usize,
    /// `start[i]` = cost to begin the tour at option `i`.
    pub start: &'a [f64],
    /// `end[i]` = cost to finish the tour at option `i`.
    pub end: &'a [f64],
}

impl Problem<'_> {
    /// Transition cost from option `i` to option `j`.
    #[inline]
    fn trans(&self, i: usize, j: usize) -> f64 {
        self.transition[i * self.options + j]
    }
}

/// Above this many DP cells, fall back to the heuristic. `states · options`
/// where `states = 1 << n_clusters`, so the table is exponential in cluster
/// count. 16M cells × 16 B ≈ 256 MB.
pub(crate) const DEFAULT_CELL_BUDGET: usize = 16 * 1024 * 1024;

/// The bitmask is a `u32`, so at most 32 clusters can be solved exactly.
const MAX_EXACT_CLUSTERS: usize = 32;

pub(crate) fn solve(problem: &Problem, cell_budget: usize) -> Option<Solution> {
    if problem.n_clusters == 0 {
        return Some(Solution { order: Vec::new(), cost: 0.0 });
    }
    debug_assert_eq!(
        problem.options,
        problem.cluster_ids.len(),
        "transition stride (problem.options) must equal the option count"
    );
    let options = problem.options;
    let too_big = problem.n_clusters > MAX_EXACT_CLUSTERS
        || (1usize << problem.n_clusters).saturating_mul(options) > cell_budget;
    if too_big {
        tracing::debug!(
            n_clusters = problem.n_clusters,
            options,
            "deke-multipath: AGTSP over cell budget; using cluster-optimization heuristic"
        );
        solve_heuristic(problem)
    } else {
        solve_exact(problem)
    }
}

/// Backpointer for one Held–Karp cell. Kept in a separate array from the cost
/// (structure-of-arrays): the inner loop reads/writes the dense `cost` array on
/// every candidate but touches a backpointer only on the rare improvement, so
/// splitting them keeps the hot `f64` array twice as dense per cache line.
#[derive(Clone, Copy)]
struct Back {
    prev_mask: u32,
    prev_opt: i32,
}

const NO_BACK: Back = Back { prev_mask: 0, prev_opt: -1 };

/// Group option indices by their cluster: `out[c]` lists every option in cluster
/// `c`.
fn group_by_cluster(problem: &Problem) -> Vec<Vec<usize>> {
    let mut by_cluster = vec![Vec::new(); problem.n_clusters];
    for (i, &c) in problem.cluster_ids.iter().enumerate() {
        by_cluster[c].push(i);
    }
    by_cluster
}

fn solve_exact(problem: &Problem) -> Option<Solution> {
    let options = problem.options;
    let states = 1usize << problem.n_clusters;
    let mut cost = vec![f64::INFINITY; states * options];
    let mut back = vec![NO_BACK; states * options];
    let idx = |mask: usize, opt: usize| mask * options + opt;

    let by_cluster = group_by_cluster(problem);

    for (i, &c) in problem.cluster_ids.iter().enumerate() {
        let s = problem.start[i];
        if !s.is_finite() {
            continue;
        }
        let id = idx(1usize << c, i);
        if s < cost[id] {
            cost[id] = s;
            back[id] = Back { prev_mask: u32::MAX, prev_opt: -1 };
        }
    }

    for mask in 1..states {
        // Only options whose cluster is already in `mask` can be a finite
        // predecessor; walk the set bits of `mask` instead of scanning every
        // option (most of which are unreachable, cold dp cells).
        let mut present = mask;
        while present != 0 {
            let cluster = present.trailing_zeros() as usize;
            present &= present - 1;
            for &opt in &by_cluster[cluster] {
                let base = cost[idx(mask, opt)];
                if !base.is_finite() {
                    continue;
                }
                for (next_cluster, opts) in by_cluster.iter().enumerate() {
                    let bit = 1usize << next_cluster;
                    if mask & bit != 0 {
                        continue;
                    }
                    let new_mask = mask | bit;
                    for &next in opts {
                        let step = problem.trans(opt, next);
                        if !step.is_finite() {
                            continue;
                        }
                        let new_cost = base + step;
                        let id = idx(new_mask, next);
                        if new_cost < cost[id] {
                            cost[id] = new_cost;
                            back[id] = Back { prev_mask: mask as u32, prev_opt: opt as i32 };
                        }
                    }
                }
            }
        }
    }

    let full = states - 1;
    let (best_opt, best_cost) = (0..options)
        .map(|opt| (opt, cost[idx(full, opt)] + problem.end[opt]))
        .filter(|(_, c)| c.is_finite())
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))?;

    let mut order = Vec::with_capacity(problem.n_clusters);
    let mut mask = full;
    let mut opt = best_opt;
    loop {
        order.push(opt);
        let b = back[idx(mask, opt)];
        if b.prev_opt < 0 {
            break;
        }
        mask = b.prev_mask as usize;
        opt = b.prev_opt as usize;
    }
    order.reverse();

    Some(Solution { order, cost: best_cost })
}

/// At most this many nearest-neighbour seeds (the clusters cheapest to start
/// from) are expanded into full tours; the best by true cost wins.
const MAX_HEURISTIC_STARTS: usize = 8;

/// Two-level heuristic: order clusters cheaply with the surrogate, then choose
/// options exactly for that order with [`cluster_optimize`]. A handful of
/// nearest-neighbour seeds are refined by surrogate 2-opt + Or-opt and the best
/// true (option-level) cost is returned.
fn solve_heuristic(problem: &Problem) -> Option<Solution> {
    let by_cluster = group_by_cluster(problem);
    let surrogate = Surrogate::build(problem, &by_cluster);

    // Seed from the clusters cheapest to begin at (RoboTSP orders clusters by a
    // cheap representative distance; ours is the surrogate). Trying a few seeds
    // costs little and guards against a single greedy misstep.
    let mut seeds: Vec<usize> = (0..problem.n_clusters)
        .filter(|&c| surrogate.start[c].is_finite())
        .collect();
    seeds.sort_by(|&a, &b| {
        surrogate.start[a]
            .partial_cmp(&surrogate.start[b])
            .unwrap_or(Ordering::Equal)
    });
    seeds.truncate(MAX_HEURISTIC_STARTS.max(1));

    // Each seed is an independent tour build + refine + cluster-optimize, so the
    // seeds fan out across the rayon pool when the `rayon` feature is on.
    let candidates: Vec<Solution> = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            seeds
                .par_iter()
                .filter_map(|&first| solve_from_seed(problem, &by_cluster, &surrogate, first))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            seeds
                .iter()
                .filter_map(|&first| solve_from_seed(problem, &by_cluster, &surrogate, first))
                .collect()
        }
    };

    candidates
        .into_iter()
        .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal))
}

/// Build a full solution from one nearest-neighbour seed cluster: order the
/// clusters with the surrogate (NN → 2-opt → Or-opt), then select options
/// exactly with [`cluster_optimize`].
fn solve_from_seed(
    problem: &Problem,
    by_cluster: &[Vec<usize>],
    surrogate: &Surrogate,
    first: usize,
) -> Option<Solution> {
    let order = surrogate.nearest_neighbour_order(problem.n_clusters, first);
    let order = surrogate.two_opt(order);
    let order = surrogate.or_opt(order);
    let (order, cost) = cluster_optimize(problem, by_cluster, &order)?;
    Some(Solution { order, cost })
}

/// Given a fixed cluster order, pick the optimal one-option-per-cluster
/// assignment by a forward shortest-path pass over the layered DAG whose layers
/// are the clusters' option sets — the "cluster optimization" subproblem. Cost
/// includes the start and end terms. `O(Σ |layer_i|·|layer_{i+1}|)`; returns the
/// option indices in order and the total cost, or `None` if every assignment is
/// non-finite.
fn cluster_optimize(
    problem: &Problem,
    by_cluster: &[Vec<usize>],
    cluster_order: &[usize],
) -> Option<(Vec<usize>, f64)> {
    if cluster_order.is_empty() {
        return Some((Vec::new(), 0.0));
    }
    let layers: Vec<&[usize]> =
        cluster_order.iter().map(|&c| by_cluster[c].as_slice()).collect();

    let mut cost: Vec<f64> = layers[0].iter().map(|&o| problem.start[o]).collect();
    let mut back: Vec<Vec<i32>> = vec![vec![-1; layers[0].len()]];
    for l in 1..layers.len() {
        let prev = layers[l - 1];
        let mut next_cost = vec![f64::INFINITY; layers[l].len()];
        let mut next_back = vec![-1i32; layers[l].len()];
        for (j, &oj) in layers[l].iter().enumerate() {
            for (i, &oi) in prev.iter().enumerate() {
                if !cost[i].is_finite() {
                    continue;
                }
                let candidate = cost[i] + problem.trans(oi, oj);
                if candidate < next_cost[j] {
                    next_cost[j] = candidate;
                    next_back[j] = i as i32;
                }
            }
        }
        cost = next_cost;
        back.push(next_back);
    }

    let last = layers.len() - 1;
    let (best_j, best_cost) = layers[last]
        .iter()
        .enumerate()
        .map(|(j, &o)| (j, cost[j] + problem.end[o]))
        .filter(|(_, c)| c.is_finite())
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))?;

    let mut layer_idx = vec![0usize; layers.len()];
    layer_idx[last] = best_j;
    for l in (1..layers.len()).rev() {
        let prev = back[l][layer_idx[l]];
        if prev < 0 {
            return None;
        }
        layer_idx[l - 1] = prev as usize;
    }
    let order = layers.iter().zip(&layer_idx).map(|(layer, &idx)| layer[idx]).collect();
    Some((order, best_cost))
}

/// Cheap cluster-to-cluster distances used to *order* clusters before any option
/// is chosen. Each entry is the best (minimum) over the option pairs, so the
/// surrogate is an optimistic estimate of the real ordering cost.
struct Surrogate {
    dist: Vec<Vec<f64>>,
    start: Vec<f64>,
    end: Vec<f64>,
}

impl Surrogate {
    fn build(problem: &Problem, by_cluster: &[Vec<usize>]) -> Self {
        let n = problem.n_clusters;
        let mut dist = vec![vec![f64::INFINITY; n]; n];
        for (a, opts_a) in by_cluster.iter().enumerate() {
            for (b, opts_b) in by_cluster.iter().enumerate() {
                if a == b {
                    continue;
                }
                let mut best = f64::INFINITY;
                for &i in opts_a {
                    for &j in opts_b {
                        best = best.min(problem.trans(i, j));
                    }
                }
                dist[a][b] = best;
            }
        }
        let reduce = |opts: &[usize], src: &[f64]| {
            opts.iter().map(|&i| src[i]).fold(f64::INFINITY, f64::min)
        };
        let start = by_cluster.iter().map(|o| reduce(o, problem.start)).collect();
        let end = by_cluster.iter().map(|o| reduce(o, problem.end)).collect();
        Surrogate { dist, start, end }
    }

    fn nearest_neighbour_order(&self, n_clusters: usize, first: usize) -> Vec<usize> {
        let mut visited = vec![false; n_clusters];
        let mut order = Vec::with_capacity(n_clusters);
        visited[first] = true;
        order.push(first);
        let mut current = first;
        for _ in 1..n_clusters {
            let mut best = f64::INFINITY;
            let mut pick: Option<usize> = None;
            for (cluster, &seen) in visited.iter().enumerate() {
                if seen {
                    continue;
                }
                if self.dist[current][cluster] < best {
                    best = self.dist[current][cluster];
                    pick = Some(cluster);
                }
            }
            // A disconnected surrogate (all-infinite) still has to place every
            // cluster; cluster_optimize will report infeasibility if it is real.
            let next = pick.or_else(|| visited.iter().position(|&v| !v)).unwrap();
            visited[next] = true;
            order.push(next);
            current = next;
        }
        order
    }

    /// Change in tour cost from reversing `order[i..=j]`. The tour is asymmetric,
    /// so the reversed segment's internal edges flip direction; only the two
    /// boundary edges and the segment's internal edges are touched, never the
    /// untouched prefix/suffix.
    fn reverse_delta(&self, order: &[usize], i: usize, j: usize) -> f64 {
        let n = order.len();
        let old_in = if i == 0 { self.start[order[i]] } else { self.dist[order[i - 1]][order[i]] };
        let new_in = if i == 0 { self.start[order[j]] } else { self.dist[order[i - 1]][order[j]] };
        let old_out =
            if j == n - 1 { self.end[order[j]] } else { self.dist[order[j]][order[j + 1]] };
        let new_out =
            if j == n - 1 { self.end[order[i]] } else { self.dist[order[i]][order[j + 1]] };
        let mut old_int = 0.0;
        let mut new_int = 0.0;
        for k in i..j {
            old_int += self.dist[order[k]][order[k + 1]];
            new_int += self.dist[order[k + 1]][order[k]];
        }
        (new_in + new_int + new_out) - (old_in + old_int + old_out)
    }

    /// Asymmetric 2-opt over the cluster order (length = clusters, not options).
    fn two_opt(&self, mut order: Vec<usize>) -> Vec<usize> {
        let n = order.len();
        if n < 4 {
            return order;
        }
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n - 1 {
                for j in i + 1..n {
                    if self.reverse_delta(&order, i, j) < -1e-9 {
                        order[i..=j].reverse();
                        improved = true;
                    }
                }
            }
        }
        order
    }

    /// Or-opt: relocate a single cluster to its best position while that strictly
    /// improves the surrogate cost. Complements 2-opt's segment reversals. Each
    /// candidate position is scored by an O(1) edge delta rather than a full
    /// tour re-walk.
    fn or_opt(&self, mut order: Vec<usize>) -> Vec<usize> {
        let n = order.len();
        if n < 3 {
            return order;
        }
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n {
                let node = order[i];
                let old_in = if i == 0 { self.start[node] } else { self.dist[order[i - 1]][node] };
                let old_out =
                    if i == n - 1 { self.end[node] } else { self.dist[node][order[i + 1]] };
                let bridge = if i == 0 {
                    self.start[order[1]]
                } else if i == n - 1 {
                    self.end[order[i - 1]]
                } else {
                    self.dist[order[i - 1]][order[i + 1]]
                };
                let removal = bridge - old_in - old_out;

                let mut without = order.clone();
                without.remove(i);
                let w = without.len();

                let mut best = -1e-9;
                let mut best_pos: Option<usize> = None;
                for p in 0..=w {
                    let removed = if p == 0 {
                        self.start[without[0]]
                    } else if p == w {
                        self.end[without[w - 1]]
                    } else {
                        self.dist[without[p - 1]][without[p]]
                    };
                    let add_l =
                        if p == 0 { self.start[node] } else { self.dist[without[p - 1]][node] };
                    let add_r = if p == w { self.end[node] } else { self.dist[node][without[p]] };
                    let delta = removal + (add_l + add_r - removed);
                    if delta < best {
                        best = delta;
                        best_pos = Some(p);
                    }
                }
                if let Some(p) = best_pos {
                    without.insert(p, node);
                    order = without;
                    improved = true;
                }
            }
        }
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem<'a>(
        cluster_ids: &'a [usize],
        n_clusters: usize,
        transition: &'a [f64],
        start: &'a [f64],
        end: &'a [f64],
    ) -> Problem<'a> {
        Problem { cluster_ids, n_clusters, transition, options: cluster_ids.len(), start, end }
    }

    /// Flatten a row-major matrix written as rows into the contiguous layout
    /// `Problem` expects.
    fn flat(rows: &[&[f64]]) -> Vec<f64> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    #[test]
    fn empty_problem_is_trivial() {
        let sol = solve(&problem(&[], 0, &[], &[], &[]), DEFAULT_CELL_BUDGET).unwrap();
        assert!(sol.order.is_empty());
        assert_eq!(sol.cost, 0.0);
    }

    #[test]
    fn picks_cheaper_start() {
        let cluster_ids = [0, 1];
        let transition = flat(&[&[0.0, 5.0], &[5.0, 0.0]]);
        let start = [1.0, 10.0];
        let end = [0.0, 0.0];
        let sol =
            solve(&problem(&cluster_ids, 2, &transition, &start, &end), DEFAULT_CELL_BUDGET)
                .unwrap();
        assert_eq!(sol.order, vec![0, 1]);
        assert!((sol.cost - 6.0).abs() < 1e-9);
    }

    #[test]
    fn generalized_picks_best_option_in_cluster() {
        // Cluster 0 = {opt0, opt1}, cluster 1 = {opt2}. opt1 is the cheap
        // realization of cluster 0 and connects cheaply to opt2.
        let cluster_ids = [0, 0, 1];
        let inf = f64::INFINITY;
        let transition = flat(&[&[0.0, inf, 9.0], &[inf, 0.0, 1.0], &[9.0, 1.0, 0.0]]);
        let start = [5.0, 1.0, 5.0];
        let end = [0.0, 0.0, 0.0];
        let sol =
            solve(&problem(&cluster_ids, 2, &transition, &start, &end), DEFAULT_CELL_BUDGET)
                .unwrap();
        assert_eq!(sol.order, vec![1, 2]);
        assert!((sol.cost - 2.0).abs() < 1e-9);
    }

    #[test]
    fn end_cost_breaks_the_tie() {
        let cluster_ids = [0, 0];
        let transition = flat(&[&[0.0, 0.0], &[0.0, 0.0]]);
        let start = [1.0, 1.0];
        let end = [5.0, 0.0];
        let sol =
            solve(&problem(&cluster_ids, 1, &transition, &start, &end), DEFAULT_CELL_BUDGET)
                .unwrap();
        assert_eq!(sol.order, vec![1]);
        assert!((sol.cost - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cluster_optimize_finds_layered_shortest_path() {
        // 2 clusters, 2 options each. Cluster 0 = {0,1}, cluster 1 = {2,3}.
        // The cheap chain is start opt1 (1) -> opt2 (1), end opt2 (0) = 2.
        let cluster_ids = [0, 0, 1, 1];
        let inf = f64::INFINITY;
        let transition = flat(&[
            &[0.0, 0.0, 8.0, 8.0],
            &[0.0, 0.0, 1.0, 9.0],
            &[inf, inf, 0.0, 0.0],
            &[inf, inf, 0.0, 0.0],
        ]);
        let start = [7.0, 1.0, inf, inf];
        let end = [inf, inf, 0.0, 5.0];
        let p = problem(&cluster_ids, 2, &transition, &start, &end);
        let by_cluster = group_by_cluster(&p);
        let (order, cost) = cluster_optimize(&p, &by_cluster, &[0, 1]).unwrap();
        assert_eq!(order, vec![1, 2]);
        assert!((cost - 2.0).abs() < 1e-9);
    }

    #[test]
    fn heuristic_matches_exact_generalized() {
        // Same instance as the exact generalized test, forced through the
        // heuristic with a zero cell budget.
        let cluster_ids = [0, 0, 1];
        let inf = f64::INFINITY;
        let transition = flat(&[&[0.0, inf, 9.0], &[inf, 0.0, 1.0], &[9.0, 1.0, 0.0]]);
        let start = [5.0, 1.0, 5.0];
        let end = [0.0, 0.0, 0.0];
        let sol = solve(&problem(&cluster_ids, 2, &transition, &start, &end), 0).unwrap();
        assert_eq!(sol.order, vec![1, 2]);
        assert!((sol.cost - 2.0).abs() < 1e-9);
    }

    #[test]
    fn heuristic_matches_exact_on_small_instance() {
        let cluster_ids = [0, 1, 2];
        let transition = flat(&[&[0.0, 2.0, 9.0], &[3.0, 0.0, 2.0], &[4.0, 6.0, 0.0]]);
        let start = [0.0, 7.0, 7.0];
        let end = [0.0, 0.0, 0.0];
        let exact =
            solve(&problem(&cluster_ids, 3, &transition, &start, &end), DEFAULT_CELL_BUDGET)
                .unwrap();
        // A zero cell budget forces the NN + 2-opt path.
        let heuristic =
            solve(&problem(&cluster_ids, 3, &transition, &start, &end), 0).unwrap();
        assert!(heuristic.cost + 1e-9 >= exact.cost, "heuristic beat optimal");
        assert!(
            (heuristic.cost - exact.cost).abs() < 1e-9,
            "heuristic {} != exact {}",
            heuristic.cost,
            exact.cost
        );
    }
}
