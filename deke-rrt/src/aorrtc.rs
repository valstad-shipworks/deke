use deke_types::{DekeResult, SRobotPath, SRobotQ, Validator};

use crate::randomizer::{DekeRng, RandomizerType};
use crate::rrtc::{
    RrtcSettings, path_cost, reduce, sample_uniform, shortcut, smooth_bspline,
    solve as rrtc_solve, validate_edge_stats, weighted_distance,
};
use crate::tree::RrtTree;
use crate::{AnytimeInfo, RrtDiagnostic, RrtTermination};

#[derive(Debug, Clone, Copy)]
pub struct AorrtcSettings<const N: usize> {
    pub rrtc: RrtcSettings<N>,
    pub max_iterations: usize,
    pub max_samples: usize,
    pub use_phs: bool,
    pub cost_bound_resamples: usize,
    pub stall_iterations: usize,
    pub dof_cost_weights: SRobotQ<N, f64>,
    pub penalize_static_dof: bool,
    pub static_dof_penalty: f64,
    pub static_dof_threshold: f64,
    /// RNG used for non-sampling work (PHS box-Muller draws, c_rand picks,
    /// cost-bound resampling). Kept separate from `rrtc.randomizer` so that
    /// e.g. joint sampling can use Halton while these auxiliary calls stay on
    /// a conventional PRNG and don't perturb the Halton stripe.
    pub aux_randomizer: RandomizerType,
    pub aux_seed: u64,
    pub simplify_shortcut: bool,
    pub simplify_bspline_steps: usize,
    pub simplify_bspline_midpoint_interpolation: f64,
    pub simplify_bspline_min_change: f64,
    pub simplify_reduce_max_steps: usize,
    pub simplify_reduce_range_ratio: f64,
}

impl<const N: usize> AorrtcSettings<N> {
    pub fn new(lower: SRobotQ<N, f64>, upper: SRobotQ<N, f64>) -> Self {
        Self {
            rrtc: RrtcSettings::new(lower, upper),
            max_iterations: 100_000,
            max_samples: 100_000,
            use_phs: true,
            cost_bound_resamples: 1000,
            stall_iterations: 50_000,
            dof_cost_weights: SRobotQ::splat(1.0),
            penalize_static_dof: false,
            static_dof_penalty: 100.0,
            static_dof_threshold: 1e-4,
            aux_randomizer: RandomizerType::Wyrand,
            aux_seed: 43,
            simplify_shortcut: true,
            simplify_bspline_steps: 5,
            simplify_bspline_midpoint_interpolation: 0.5,
            simplify_bspline_min_change: 1e-5,
            simplify_reduce_max_steps: 25,
            simplify_reduce_range_ratio: 0.8,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Phs<const N: usize> {
    center: [f64; N],
    c_min: f64,
    c_best: f64,
    basis: [[f64; N]; N],
    inv_sqrt_coeffs: [f64; N],
}

impl<const N: usize> Phs<N> {
    fn new(start: &SRobotQ<N, f64>, goal: &SRobotQ<N, f64>, coeffs: &[f64; N]) -> Self {
        let mut sqrt_coeffs = [0.0; N];
        let mut inv_sqrt_coeffs = [0.0; N];
        for i in 0..N {
            sqrt_coeffs[i] = coeffs[i].sqrt();
            inv_sqrt_coeffs[i] = 1.0 / sqrt_coeffs[i];
        }

        let mut scaled_start = [0.0; N];
        let mut scaled_goal = [0.0; N];
        let mut center = [0.0; N];
        for i in 0..N {
            scaled_start[i] = start.0[i] * sqrt_coeffs[i];
            scaled_goal[i] = goal.0[i] * sqrt_coeffs[i];
            center[i] = (scaled_start[i] + scaled_goal[i]) * 0.5;
        }

        let c_min = euclidean_dist(&scaled_start, &scaled_goal);
        let basis = build_orthonormal_basis(&scaled_start, &scaled_goal, c_min);

        Self {
            center,
            c_min,
            c_best: f64::INFINITY,
            basis,
            inv_sqrt_coeffs,
        }
    }

    fn set_cost(&mut self, c_best: f64) {
        self.c_best = c_best;
    }

    fn sample<S: DekeRng<N>, A: DekeRng<N>>(
        &self,
        sample_rng: &mut S,
        aux_rng: &mut A,
        lower: &SRobotQ<N, f64>,
        upper: &SRobotQ<N, f64>,
    ) -> SRobotQ<N, f64> {
        if self.c_best >= f64::INFINITY || self.c_best <= self.c_min + 1e-10 {
            return sample_uniform(sample_rng, lower, upper);
        }

        let r_major = self.c_best * 0.5;
        let r_minor = (self.c_best * self.c_best - self.c_min * self.c_min).sqrt() * 0.5;

        const MAX_REJECTIONS: u32 = 1000;
        for _ in 0..MAX_REJECTIONS {
            let ball = uniform_unit_ball::<N, _>(aux_rng);

            let mut scaled_ball = [0.0; N];
            scaled_ball[0] = ball[0] * r_major;
            for i in 1..N {
                scaled_ball[i] = ball[i] * r_minor;
            }

            let mut point = [0.0; N];
            #[allow(clippy::needless_range_loop)]
            for i in 0..N {
                for j in 0..N {
                    point[i] += self.basis[j][i] * scaled_ball[j];
                }
                point[i] += self.center[i];
            }

            let mut q = [0.0f64; N];
            let mut in_bounds = true;
            for i in 0..N {
                let qi = point[i] * self.inv_sqrt_coeffs[i];
                q[i] = qi;
                if qi < lower[i] || qi > upper[i] {
                    in_bounds = false;
                    break;
                }
            }

            if in_bounds {
                return SRobotQ::from_array(q);
            }
        }

        sample_uniform(sample_rng, lower, upper)
    }
}

fn euclidean_dist<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

fn build_orthonormal_basis<const N: usize>(
    start: &[f64; N],
    goal: &[f64; N],
    c_min: f64,
) -> [[f64; N]; N] {
    let mut basis = [[0.0; N]; N];

    if c_min < 1e-10 {
        #[allow(clippy::needless_range_loop)]
        for i in 0..N {
            basis[i][i] = 1.0;
        }
        return basis;
    }

    for i in 0..N {
        basis[0][i] = (goal[i] - start[i]) / c_min;
    }

    for i in 1..N {
        let mut v = [0.0; N];
        v[i] = 1.0;

        #[allow(clippy::needless_range_loop)]
        for j in 0..i {
            let dot: f64 = (0..N).map(|k| v[k] * basis[j][k]).sum();
            for k in 0..N {
                v[k] -= dot * basis[j][k];
            }
        }

        let norm: f64 = (0..N).map(|k| v[k] * v[k]).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for k in 0..N {
                basis[i][k] = v[k] / norm;
            }
        }
    }

    basis
}

fn box_muller<const N: usize, R: DekeRng<N>>(rng: &mut R) -> (f64, f64) {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

fn uniform_unit_ball<const N: usize, R: DekeRng<N>>(rng: &mut R) -> [f64; N] {
    let mut point = [0.0; N];

    let mut i = 0;
    while i + 1 < N {
        let (n1, n2) = box_muller(rng);
        point[i] = n1;
        point[i + 1] = n2;
        i += 2;
    }
    if i < N {
        let (n1, _) = box_muller(rng);
        point[i] = n1;
    }

    let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-30 {
        return point;
    }
    for v in &mut point {
        *v /= norm;
    }

    let r = rng.next_f64().powf(1.0 / N as f64);
    for v in &mut point {
        *v *= r;
    }
    point
}

fn steer<const N: usize>(
    from: &SRobotQ<N, f64>,
    toward: &SRobotQ<N, f64>,
    range: f64,
    coeffs: &[f64; N],
) -> SRobotQ<N, f64> {
    let dist = weighted_distance(from, toward, coeffs);
    if dist <= range {
        *toward
    } else {
        *from + (*toward - *from) * (range / dist)
    }
}

pub(crate) fn solve<const N: usize, V: Validator<N, (), f64>, S: DekeRng<N>, A: DekeRng<N>>(
    start: &SRobotQ<N, f64>,
    goal: &SRobotQ<N, f64>,
    validator: &V,
    ctx: &V::Context<'_>,
    settings: &AorrtcSettings<N>,
    sample_rng: &mut S,
    aux_rng: &mut A,
) -> (DekeResult<SRobotPath<N, f64>>, RrtDiagnostic) {
    let timer = std::time::Instant::now();
    let dof_coeffs = {
        let mut c = [0.0f64; N];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = settings.dof_cost_weights.0[i];
            if settings.penalize_static_dof {
                let delta = (start.0[i] - goal.0[i]).abs();
                if delta < settings.static_dof_threshold {
                    *ci *= settings.static_dof_penalty;
                }
            }
        }
        c
    };

    let (initial_result, initial_diag) =
        rrtc_solve(start, goal, validator, ctx, &settings.rrtc, sample_rng);

    let initial_path = match initial_result {
        Ok(path) => path,
        Err(e) => {
            // Surface the failed initial RRTC stats verbatim, but mark it
            // as a no-initial-path failure so callers can distinguish "the
            // RRTC could not even seed AORRTC" from a stalled refinement.
            let mut diag = initial_diag;
            diag.termination = RrtTermination::NoInitialPath;
            diag.anytime = Some(AnytimeInfo {
                initial_cost: f64::INFINITY,
                initial_iterations: diag.iterations,
                improvements: 0,
                iters_since_last_improvement: 0,
                optimality_ratio: f64::INFINITY,
            });
            return (Err(e), diag);
        }
    };

    // Carry RRTC's stats forward so the final diagnostic reflects work done
    // in *both* phases.
    let mut stats = initial_diag.extension_stats;
    let mut closest_approach = initial_diag.closest_approach;
    let initial_iterations = initial_diag.iterations;

    let mut best_waypoints: Vec<SRobotQ<N, f64>> = initial_path.iter().copied().collect();
    let mut best_cost = path_cost(&best_waypoints, &dof_coeffs);
    let initial_cost = best_cost;

    let c_min = weighted_distance(start, goal, &dof_coeffs);
    let mut total_iterations = initial_iterations;
    let mut iters_since_improvement = 0usize;
    let mut improvements = 0usize;
    #[allow(unused_assignments)]
    let mut termination = RrtTermination::MaxIterationsExceeded;

    if best_cost <= c_min + 1e-8 {
        termination = RrtTermination::OptimalReached;
    } else {
        let mut phs = Phs::new(start, goal, &dof_coeffs);
        phs.set_cost(best_cost);

        'outer: loop {
            if total_iterations >= settings.max_iterations {
                termination = RrtTermination::MaxIterationsExceeded;
                break;
            }
            if iters_since_improvement >= settings.stall_iterations {
                termination = RrtTermination::Stalled;
                break;
            }

            let mut start_tree = RrtTree::with_capacity(&dof_coeffs, settings.max_samples / 2);
            let mut goal_tree = RrtTree::with_capacity(&dof_coeffs, settings.max_samples / 2);

            start_tree.add(*start, 0, settings.rrtc.radius, 0.0);
            goal_tree.add(*goal, 0, settings.rrtc.radius, 0.0);

            for inner in 0..settings.max_samples {
                if total_iterations >= settings.max_iterations {
                    termination = RrtTermination::MaxIterationsExceeded;
                    break 'outer;
                }
                if iters_since_improvement >= settings.stall_iterations {
                    termination = RrtTermination::Stalled;
                    break 'outer;
                }
                if start_tree.len() + goal_tree.len() >= settings.max_samples {
                    break;
                }
                total_iterations += 1;
                iters_since_improvement += 1;
                stats.extension_attempts += 1;

                let q_rand = if settings.use_phs {
                    phs.sample(
                        sample_rng,
                        aux_rng,
                        &settings.rrtc.joint_lower,
                        &settings.rrtc.joint_upper,
                    )
                } else {
                    sample_uniform(
                        sample_rng,
                        &settings.rrtc.joint_lower,
                        &settings.rrtc.joint_upper,
                    )
                };

                let extend_start = inner % 2 == 0;
                let (root_q, target_q) = if extend_start {
                    (start, goal)
                } else {
                    (goal, start)
                };
                let g_hat = weighted_distance(&q_rand, root_q, &dof_coeffs);
                let h_hat = weighted_distance(&q_rand, target_q, &dof_coeffs);
                let f_hat = g_hat + h_hat;
                let c_range = (best_cost - f_hat).max(0.0);
                let c_rand = aux_rng.next_f64() * c_range + g_hat;

                let (tree_a, tree_b) = if extend_start {
                    (&mut start_tree, &mut goal_tree)
                } else {
                    (&mut goal_tree, &mut start_tree)
                };

                let (near_idx, near_dist) = tree_a.find_nearest_ao(&q_rand, c_rand);

                if settings.rrtc.dynamic_domain && near_dist > tree_a.radius(near_idx) {
                    let r = tree_a.radius(near_idx);
                    tree_a.set_radius(
                        near_idx,
                        (r * (1.0 - settings.rrtc.alpha)).max(settings.rrtc.min_radius),
                    );
                    stats.dynamic_domain_rejections += 1;
                    continue;
                }

                let q_near = *tree_a.node(near_idx);
                let q_new = steer(&q_near, &q_rand, settings.rrtc.range, &dof_coeffs);

                if validate_edge_stats(
                    &q_near,
                    &q_new,
                    settings.rrtc.resolution,
                    validator,
                    ctx,
                    &mut stats,
                )
                .is_err()
                {
                    if settings.rrtc.dynamic_domain {
                        let r = tree_a.radius(near_idx);
                        tree_a.set_radius(
                            near_idx,
                            (r * (1.0 - settings.rrtc.alpha)).max(settings.rrtc.min_radius),
                        );
                    }
                    continue;
                }

                let mut best_parent = near_idx;
                let mut new_cost =
                    tree_a.cost(near_idx) + weighted_distance(&q_near, &q_new, &dof_coeffs);

                if settings.cost_bound_resamples > 0 {
                    let g_hat_new = weighted_distance(&q_new, root_q, &dof_coeffs);
                    for _ in 0..settings.cost_bound_resamples {
                        let c_range = (new_cost - g_hat_new).max(0.0);
                        if c_range == 0.0 {
                            break;
                        }
                        let c_rand = aux_rng.next_f64() * c_range + g_hat_new;
                        let (cand_idx, cand_dist) = tree_a.find_nearest_ao(&q_new, c_rand);
                        let cand_cost = tree_a.cost(cand_idx) + cand_dist;
                        if cand_idx == best_parent || cand_cost >= new_cost {
                            break;
                        }
                        let q_cand = *tree_a.node(cand_idx);
                        if validate_edge_stats(
                            &q_cand,
                            &q_new,
                            settings.rrtc.resolution,
                            validator,
                            ctx,
                            &mut stats,
                        )
                        .is_ok()
                        {
                            best_parent = cand_idx;
                            new_cost = cand_cost;
                        } else {
                            break;
                        }
                    }
                }

                let new_idx = tree_a.add(q_new, best_parent, settings.rrtc.radius, new_cost);
                stats.successful_extensions += 1;

                if settings.rrtc.dynamic_domain {
                    let r = tree_a.radius(near_idx);
                    tree_a.set_radius(near_idx, r * (1.0 + settings.rrtc.alpha));
                }

                let (connect_near, connect_dist) =
                    tree_b.find_nearest_ao(&q_new, best_cost - new_cost);
                let connect_cost = tree_b.cost(connect_near);

                if new_cost + connect_dist + connect_cost >= best_cost {
                    continue;
                }

                stats.connect_attempts += 1;
                let mut cur_idx = connect_near;

                let mut connected = false;
                loop {
                    let q_cur = *tree_b.node(cur_idx);
                    let dist = weighted_distance(&q_cur, &q_new, &dof_coeffs);

                    if dist < 1e-6 {
                        connected = true;
                        break;
                    }

                    let q_step = steer(&q_cur, &q_new, settings.rrtc.range, &dof_coeffs);

                    if validate_edge_stats(
                        &q_cur,
                        &q_step,
                        settings.rrtc.resolution,
                        validator,
                        ctx,
                        &mut stats,
                    )
                    .is_err()
                    {
                        break;
                    }

                    let step_cost =
                        tree_b.cost(cur_idx) + weighted_distance(&q_cur, &q_step, &dof_coeffs);
                    let reached = weighted_distance(&q_step, &q_new, &dof_coeffs) < 1e-6;
                    let added_q = if reached { q_new } else { q_step };
                    cur_idx = tree_b.add(added_q, cur_idx, settings.rrtc.radius, step_cost);

                    if reached {
                        connected = true;
                        break;
                    }
                }

                if connected {
                    stats.connect_successes += 1;
                    let (ta, tb) = if extend_start {
                        (&start_tree as &RrtTree<N>, &goal_tree as &RrtTree<N>)
                    } else {
                        (&goal_tree as &RrtTree<N>, &start_tree as &RrtTree<N>)
                    };
                    let waypoints = reconstruct(ta, new_idx, tb, cur_idx, extend_start);
                    let actual_cost = path_cost(&waypoints, &dof_coeffs);
                    if actual_cost < best_cost {
                        best_cost = actual_cost;
                        best_waypoints = waypoints;
                        phs.set_cost(best_cost);
                        iters_since_improvement = 0;
                        improvements += 1;
                        closest_approach = 0.0;
                        break;
                    }
                }
            }

            if best_cost <= c_min + 1e-8 {
                termination = RrtTermination::OptimalReached;
                break;
            }
        }
    }

    if settings.simplify_shortcut {
        shortcut(&mut best_waypoints, validator, ctx, settings.rrtc.resolution);
    }

    if settings.simplify_bspline_steps > 0 {
        smooth_bspline(
            &mut best_waypoints,
            validator,
            ctx,
            settings.rrtc.resolution,
            settings.simplify_bspline_steps,
            settings.simplify_bspline_midpoint_interpolation,
            settings.simplify_bspline_min_change,
        );

        if settings.simplify_shortcut {
            shortcut(&mut best_waypoints, validator, ctx, settings.rrtc.resolution);
        }
    }

    if settings.simplify_reduce_max_steps > 0 {
        reduce(
            &mut best_waypoints,
            validator,
            ctx,
            settings.rrtc.resolution,
            &dof_coeffs,
            settings.simplify_reduce_max_steps,
            settings.simplify_reduce_range_ratio,
        );
    }

    best_cost = path_cost(&best_waypoints, &dof_coeffs);
    let final_path = SRobotPath::try_new(best_waypoints);

    let optimality_ratio = if c_min > 1e-12 {
        best_cost / c_min
    } else if best_cost.abs() < 1e-12 {
        1.0
    } else {
        f64::INFINITY
    };

    // The final phase here is "we found *some* path." Override the inner
    // termination to `Solved` only if we ran out of budget without ever
    // entering the refinement loop (degenerate / direct-connection cases
    // were already short-circuited above).
    let outcome = match termination {
        RrtTermination::MaxIterationsExceeded
            if best_cost <= initial_cost && total_iterations == initial_iterations =>
        {
            // We never entered the refinement loop; the initial RRTC's own
            // termination is more informative.
            initial_diag.termination
        }
        other => other,
    };

    (
        final_path,
        RrtDiagnostic {
            iterations: total_iterations,
            // Refinement uses ephemeral trees — surfacing tree sizes here
            // would be misleading. Report the initial-phase tree sizes which
            // describe the actual seed solution.
            start_tree_size: initial_diag.start_tree_size,
            goal_tree_size: initial_diag.goal_tree_size,
            path_cost: best_cost,
            elapsed_ns: timer.elapsed().as_nanos(),
            termination: outcome,
            extension_stats: stats,
            c_min,
            closest_approach,
            anytime: Some(AnytimeInfo {
                initial_cost,
                initial_iterations,
                improvements,
                iters_since_last_improvement: iters_since_improvement,
                optimality_ratio,
            }),
        },
    )
}

fn reconstruct<const N: usize>(
    tree_a: &RrtTree<N>,
    idx_a: usize,
    tree_b: &RrtTree<N>,
    idx_b: usize,
    a_is_start: bool,
) -> Vec<SRobotQ<N, f64>> {
    let backtrack = |tree: &RrtTree<N>, idx: usize| -> Vec<SRobotQ<N, f64>> {
        let mut path = Vec::new();
        let mut current = idx;
        loop {
            path.push(*tree.node(current));
            let parent = tree.parent(current);
            if parent == current {
                break;
            }
            current = parent;
        }
        path.reverse();
        path
    };

    let path_a = backtrack(tree_a, idx_a);
    let path_b = backtrack(tree_b, idx_b);

    if a_is_start {
        let mut full = path_a;
        full.extend(path_b.into_iter().rev().skip(1));
        full
    } else {
        let mut full = path_b;
        full.extend(path_a.into_iter().rev().skip(1));
        full
    }
}
