use deke_types::{DekeResult, RobotPath, SRobotQ, Validator};
use tinyrand::Rand;

use crate::RrtDiagnostic;
use crate::rrtc::{
    RrtcSettings, path_cost, rand_f64, reduce, sample_uniform, shortcut, smooth_bspline,
    solve as rrtc_solve, validate_edge, weighted_distance,
};
use crate::tree::RrtTree;

#[derive(Debug, Clone, Copy)]
pub struct AorrtcSettings<const N: usize> {
    pub rrtc: RrtcSettings<N>,
    pub max_iterations: usize,
    pub max_samples: usize,
    pub use_phs: bool,
    pub cost_bound_resamples: usize,
    pub stall_iterations: usize,
    pub dof_cost_weights: SRobotQ<N>,
    pub penalize_static_dof: bool,
    pub static_dof_penalty: f32,
    pub static_dof_threshold: f32,
    pub simplify_shortcut: bool,
    pub simplify_bspline_steps: usize,
    pub simplify_bspline_midpoint_interpolation: f32,
    pub simplify_bspline_min_change: f64,
    pub simplify_reduce_max_steps: usize,
    pub simplify_reduce_range_ratio: f64,
}

impl<const N: usize> AorrtcSettings<N> {
    pub fn new(lower: SRobotQ<N>, upper: SRobotQ<N>) -> Self {
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
    fn new(start: &SRobotQ<N>, goal: &SRobotQ<N>, coeffs: &[f64; N]) -> Self {
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
            scaled_start[i] = start.0[i] as f64 * sqrt_coeffs[i];
            scaled_goal[i] = goal.0[i] as f64 * sqrt_coeffs[i];
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

    fn sample(&self, rng: &mut impl Rand, lower: &SRobotQ<N>, upper: &SRobotQ<N>) -> SRobotQ<N> {
        if self.c_best >= f64::INFINITY || self.c_best <= self.c_min + 1e-10 {
            return sample_uniform(rng, lower, upper);
        }

        let r_major = self.c_best * 0.5;
        let r_minor = (self.c_best * self.c_best - self.c_min * self.c_min).sqrt() * 0.5;

        const MAX_REJECTIONS: u32 = 1000;
        for _ in 0..MAX_REJECTIONS {
            let ball = uniform_unit_ball::<N>(rng);

            let mut scaled_ball = [0.0; N];
            scaled_ball[0] = ball[0] * r_major;
            for i in 1..N {
                scaled_ball[i] = ball[i] * r_minor;
            }

            let mut point = [0.0; N];
            for i in 0..N {
                for j in 0..N {
                    point[i] += self.basis[j][i] * scaled_ball[j];
                }
                point[i] += self.center[i];
            }

            let mut q_f32 = [0.0f32; N];
            let mut in_bounds = true;
            for i in 0..N {
                let qi = point[i] * self.inv_sqrt_coeffs[i];
                q_f32[i] = qi as f32;
                if qi < lower[i] as f64 || qi > upper[i] as f64 {
                    in_bounds = false;
                    break;
                }
            }

            if in_bounds {
                return SRobotQ::from_array(q_f32);
            }
        }

        sample_uniform(rng, lower, upper)
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

fn box_muller(rng: &mut impl Rand) -> (f64, f64) {
    let u1 = rand_f64(rng).max(1e-300);
    let u2 = rand_f64(rng);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

fn uniform_unit_ball<const N: usize>(rng: &mut impl Rand) -> [f64; N] {
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

    let r = rand_f64(rng).powf(1.0 / N as f64);
    for v in &mut point {
        *v *= r;
    }
    point
}

fn steer<const N: usize>(
    from: &SRobotQ<N>,
    toward: &SRobotQ<N>,
    range: f64,
    coeffs: &[f64; N],
) -> SRobotQ<N> {
    let dist = weighted_distance(from, toward, coeffs);
    if dist <= range {
        *toward
    } else {
        *from + (*toward - *from) * (range / dist) as f32
    }
}

pub(crate) fn solve<const N: usize>(
    start: &SRobotQ<N>,
    goal: &SRobotQ<N>,
    validator: &mut impl Validator<N>,
    settings: &AorrtcSettings<N>,
    rng: &mut impl Rand,
) -> (DekeResult<RobotPath>, RrtDiagnostic) {
    let timer = std::time::Instant::now();
    let dof_coeffs = {
        let mut c = [0.0f64; N];
        for i in 0..N {
            c[i] = settings.dof_cost_weights.0[i] as f64;
            if settings.penalize_static_dof {
                let delta = (start.0[i] - goal.0[i]).abs();
                if delta < settings.static_dof_threshold {
                    c[i] *= settings.static_dof_penalty as f64;
                }
            }
        }
        c
    };

    let (initial_result, initial_diag) = rrtc_solve(start, goal, validator, &settings.rrtc, rng);

    let mut best_path = match initial_result {
        Ok(path) => path,
        Err(e) => return (Err(e), initial_diag),
    };

    let mut best_waypoints: Vec<SRobotQ<N>> = best_path
        .iter()
        .map(|q| SRobotQ::force_from_robotq(q))
        .collect();
    let mut best_cost = path_cost(&best_waypoints, &dof_coeffs);

    let c_min = weighted_distance(start, goal, &dof_coeffs);
    if best_cost <= c_min + 1e-8 {
        return (
            Ok(best_path),
            RrtDiagnostic {
                iterations: initial_diag.iterations,
                start_tree_size: initial_diag.start_tree_size,
                goal_tree_size: initial_diag.goal_tree_size,
                path_cost: best_cost,
                elapsed_ns: timer.elapsed().as_nanos(),
            },
        );
    }

    let mut phs = Phs::new(start, goal, &dof_coeffs);
    phs.set_cost(best_cost);
    let mut total_iterations = initial_diag.iterations;
    let mut iters_since_improvement = 0usize;

    loop {
        if total_iterations >= settings.max_iterations {
            break;
        }
        if iters_since_improvement >= settings.stall_iterations {
            break;
        }

        let mut start_tree = RrtTree::with_capacity(&dof_coeffs, settings.max_samples / 2);
        let mut goal_tree = RrtTree::with_capacity(&dof_coeffs, settings.max_samples / 2);

        start_tree.add(*start, 0, settings.rrtc.radius, 0.0);
        goal_tree.add(*goal, 0, settings.rrtc.radius, 0.0);

        for inner in 0..settings.max_samples {
            if total_iterations >= settings.max_iterations {
                break;
            }
            if iters_since_improvement >= settings.stall_iterations {
                break;
            }
            if start_tree.len() + goal_tree.len() >= settings.max_samples {
                break;
            }
            total_iterations += 1;
            iters_since_improvement += 1;

            let q_rand = if settings.use_phs {
                phs.sample(rng, &settings.rrtc.joint_lower, &settings.rrtc.joint_upper)
            } else {
                sample_uniform(rng, &settings.rrtc.joint_lower, &settings.rrtc.joint_upper)
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
            let c_rand = rand_f64(rng) * c_range + g_hat;

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
                continue;
            }

            let q_near = *tree_a.node(near_idx);
            let q_new = steer(&q_near, &q_rand, settings.rrtc.range, &dof_coeffs);

            if validate_edge(&q_near, &q_new, settings.rrtc.resolution, validator).is_err() {
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
            let mut new_cost = tree_a.cost(near_idx) + weighted_distance(&q_near, &q_new, &dof_coeffs);

            if settings.cost_bound_resamples > 0 {
                let g_hat_new = weighted_distance(&q_new, root_q, &dof_coeffs);
                for _ in 0..settings.cost_bound_resamples {
                    let c_range = (new_cost - g_hat_new).max(0.0);
                    if c_range == 0.0 {
                        break;
                    }
                    let c_rand = rand_f64(rng) * c_range + g_hat_new;
                    let (cand_idx, cand_dist) = tree_a.find_nearest_ao(&q_new, c_rand);
                    let cand_cost = tree_a.cost(cand_idx) + cand_dist;
                    if cand_idx == best_parent || cand_cost >= new_cost {
                        break;
                    }
                    let q_cand = *tree_a.node(cand_idx);
                    if validate_edge(&q_cand, &q_new, settings.rrtc.resolution, validator)
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

            if settings.rrtc.dynamic_domain {
                let r = tree_a.radius(near_idx);
                tree_a.set_radius(near_idx, r * (1.0 + settings.rrtc.alpha));
            }

            let (connect_near, connect_dist) = tree_b.find_nearest_ao(&q_new, best_cost - new_cost);
            let connect_cost = tree_b.cost(connect_near);

            if new_cost + connect_dist + connect_cost >= best_cost {
                continue;
            }

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

                if validate_edge(&q_cur, &q_step, settings.rrtc.resolution, validator).is_err() {
                    break;
                }

                let step_cost = tree_b.cost(cur_idx) + weighted_distance(&q_cur, &q_step, &dof_coeffs);
                let reached = weighted_distance(&q_step, &q_new, &dof_coeffs) < 1e-6;
                let added_q = if reached { q_new } else { q_step };
                cur_idx = tree_b.add(added_q, cur_idx, settings.rrtc.radius, step_cost);

                if reached {
                    connected = true;
                    break;
                }
            }

            if connected {
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
                    break;
                }
            }
        }

        if best_cost <= c_min + 1e-8 {
            break;
        }
    }

    if settings.simplify_shortcut {
        shortcut(&mut best_waypoints, validator, settings.rrtc.resolution);
    }

    if settings.simplify_bspline_steps > 0 {
        smooth_bspline(
            &mut best_waypoints,
            validator,
            settings.rrtc.resolution,
            settings.simplify_bspline_steps,
            settings.simplify_bspline_midpoint_interpolation,
            settings.simplify_bspline_min_change,
        );

        if settings.simplify_shortcut {
            shortcut(&mut best_waypoints, validator, settings.rrtc.resolution);
        }
    }

    if settings.simplify_reduce_max_steps > 0 {
        reduce(
            &mut best_waypoints,
            validator,
            settings.rrtc.resolution,
            &dof_coeffs,
            settings.simplify_reduce_max_steps,
            settings.simplify_reduce_range_ratio,
        );
    }

    best_cost = path_cost(&best_waypoints, &dof_coeffs);
    best_path = best_waypoints.iter().map(|q| (*q).into()).collect();

    (
        Ok(best_path),
        RrtDiagnostic {
            iterations: total_iterations,
            start_tree_size: 0,
            goal_tree_size: 0,
            path_cost: best_cost,
            elapsed_ns: timer.elapsed().as_nanos(),
        },
    )
}

fn reconstruct<const N: usize>(
    tree_a: &RrtTree<N>,
    idx_a: usize,
    tree_b: &RrtTree<N>,
    idx_b: usize,
    a_is_start: bool,
) -> Vec<SRobotQ<N>> {
    let backtrack = |tree: &RrtTree<N>, idx: usize| -> Vec<SRobotQ<N>> {
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
