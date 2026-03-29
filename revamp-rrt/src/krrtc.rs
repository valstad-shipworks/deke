use revamp_types::{RevampError, RevampResult, RobotPath, SRobotQ, Token, Validator};
use tinyrand::Rand;

use crate::rrtc::{rand_f64, sample_uniform, validate_edge};
use crate::scurve::{
    direction_cosine, kinematic_interpolate, kinematic_path_cost, time_optimal_cost,
    KinematicLimits,
};
use crate::tree::RrtTree;
use crate::RrtDiagnostic;

#[derive(Debug, Clone)]
pub struct KrrtcSettings<const N: usize> {
    pub range: f64,
    pub max_iterations: usize,
    pub max_samples: usize,
    pub joint_lower: SRobotQ<N>,
    pub joint_upper: SRobotQ<N>,
    pub kin_limits: KinematicLimits<N>,
    pub resolution: f64,
    pub dynamic_domain: bool,
    pub radius: f64,
    pub alpha: f64,
    pub min_radius: f64,
    pub balance: bool,
    pub tree_ratio: f64,
    pub seed: u64,
    pub shortcut_iterations: usize,
    pub smoothing_iterations: usize,
}

impl<const N: usize> KrrtcSettings<N> {
    pub fn new(lower: SRobotQ<N>, upper: SRobotQ<N>, kin_limits: KinematicLimits<N>) -> Self {
        Self {
            range: 0.5,
            max_iterations: 100_000,
            max_samples: 100_000,
            joint_lower: lower,
            joint_upper: upper,
            kin_limits,
            resolution: 0.05,
            dynamic_domain: true,
            radius: 4.0,
            alpha: 0.0001,
            min_radius: 1.0,
            balance: true,
            tree_ratio: 1.0,
            seed: 42,
            shortcut_iterations: 200,
            smoothing_iterations: 100,
        }
    }
}

fn kinematic_steer<const N: usize>(
    from: &SRobotQ<N>,
    toward: &SRobotQ<N>,
    range: f64,
    limits: &KinematicLimits<N>,
) -> SRobotQ<N> {
    let cost = time_optimal_cost(from, toward, limits);
    if cost <= range {
        *toward
    } else {
        kinematic_interpolate(from, toward, range / cost)
    }
}


/// Iteratively smooth sharp corners by perturbing interior waypoints.
///
/// For each iteration, finds the sharpest corner (measured in velocity-scaled
/// space) and attempts two strategies:
///
/// 1. **Pull**: move the waypoint toward the chord midpoint of its neighbors,
///    straightening the path. Tries progressively smaller pull fractions.
///
/// 2. **Arc split**: replace the sharp corner with two waypoints placed along
///    the incoming and outgoing segments, creating a rounded arc. This avoids
///    the hard direction change while staying close to the original path.
///
/// Every candidate modification is collision-validated on both affected edges
/// and only accepted if the local time-optimal cost decreases.
fn round_corners<const N: usize, TKN: Token>(
    path: &mut Vec<SRobotQ<N>>,
    validator: &mut impl Validator<N, TKN>,
    limits: &KinematicLimits<N>,
    resolution: f64,
    iterations: usize,
) {
    if path.len() < 3 {
        return;
    }

    for _ in 0..iterations {
        if path.len() < 3 {
            break;
        }

        let mut worst_idx = 0;
        let mut worst_cos = 2.0f64;

        for i in 1..path.len() - 1 {
            let cos = direction_cosine(&path[i - 1], &path[i], &path[i + 1], limits);
            if cos < worst_cos {
                worst_cos = cos;
                worst_idx = i;
            }
        }

        if worst_cos > 0.5 {
            break;
        }

        let i = worst_idx;
        let prev = path[i - 1];
        let curr = path[i];
        let next = path[i + 1];

        let old_cost = time_optimal_cost(&prev, &curr, limits)
            + time_optimal_cost(&curr, &next, limits);

        let midpoint = (prev + next) * 0.5;
        let pull_dir = midpoint - curr;

        let mut best_cost = old_cost;
        let mut best_pull = 0.0f32;

        for &alpha in &[0.5f32, 0.4, 0.3, 0.2, 0.1, 0.05] {
            let candidate = curr + pull_dir * alpha;

            if validate_edge(&prev, &candidate, resolution, validator).is_err() {
                continue;
            }
            if validate_edge(&candidate, &next, resolution, validator).is_err() {
                continue;
            }

            let new_cost = time_optimal_cost(&prev, &candidate, limits)
                + time_optimal_cost(&candidate, &next, limits);

            if new_cost < best_cost {
                best_cost = new_cost;
                best_pull = alpha;
            }
        }

        if best_pull > 0.0 {
            path[i] = curr + pull_dir * best_pull;
            continue;
        }

        // Pull didn't help — try arc split: replace the corner with two
        // waypoints that round it, placed along the incoming/outgoing edges.
        let mut best_arc_cost = old_cost;
        let mut best_arc: Option<(SRobotQ<N>, SRobotQ<N>)> = None;

        for &frac in &[0.25f32, 0.33, 0.4] {
            let p1 = prev + (curr - prev) * (1.0 - frac);
            let p2 = curr + (next - curr) * frac;

            if validate_edge(&prev, &p1, resolution, validator).is_err() {
                continue;
            }
            if validate_edge(&p1, &p2, resolution, validator).is_err() {
                continue;
            }
            if validate_edge(&p2, &next, resolution, validator).is_err() {
                continue;
            }

            let arc_cost = time_optimal_cost(&prev, &p1, limits)
                + time_optimal_cost(&p1, &p2, limits)
                + time_optimal_cost(&p2, &next, limits);

            if arc_cost < best_arc_cost {
                best_arc_cost = arc_cost;
                best_arc = Some((p1, p2));
            }
        }

        if let Some((p1, p2)) = best_arc {
            path[i] = p1;
            path.insert(i + 1, p2);
        }
    }
}

/// Random shortcutting: pick two random waypoints, if the direct edge is
/// collision-free, remove everything between them. Biases toward pairs
/// with high time-optimal cost to maximize improvement per shortcut.
fn shortcut<const N: usize, TKN: Token>(
    path: &mut Vec<SRobotQ<N>>,
    validator: &mut impl Validator<N, TKN>,
    limits: &KinematicLimits<N>,
    resolution: f64,
    iterations: usize,
    rng: &mut impl Rand,
) {
    if path.len() <= 2 {
        return;
    }

    let mut segment_costs: Vec<f64> = path
        .windows(2)
        .map(|w| time_optimal_cost(&w[0], &w[1], limits))
        .collect();
    let mut cumulative: Vec<f64> = Vec::with_capacity(segment_costs.len());
    let mut acc = 0.0;
    for &c in &segment_costs {
        acc += c;
        cumulative.push(acc);
    }

    for _ in 0..iterations {
        if path.len() <= 2 {
            break;
        }

        let total_cost = *cumulative.last().unwrap();
        if total_cost < 1e-10 {
            break;
        }

        // Pick first index weighted by segment cost (favor expensive segments)
        let r1 = rand_f64(rng) * total_cost;
        let i = cumulative.partition_point(|&c| c < r1).min(segment_costs.len() - 1);

        // Pick second index at least 2 away
        let remaining = path.len() - i - 2;
        if remaining == 0 {
            continue;
        }
        let j = i + 2 + (rand_f64(rng) * remaining as f64) as usize;
        let j = j.min(path.len() - 1);

        if validate_edge(&path[i], &path[j], resolution, validator).is_ok() {
            let old_cost: f64 = (i..j).map(|k| segment_costs[k]).sum();
            let new_cost = time_optimal_cost(&path[i], &path[j], limits);

            if new_cost < old_cost {
                path.drain(i + 1..j);

                segment_costs.drain(i..j.min(segment_costs.len()));
                segment_costs.insert(i, new_cost);

                cumulative.clear();
                acc = 0.0;
                for &c in &segment_costs {
                    acc += c;
                    cumulative.push(acc);
                }
            }
        }
    }
}

fn backtrack_path<const N: usize>(tree: &RrtTree<N>, idx: usize) -> Vec<SRobotQ<N>> {
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
}

fn reconstruct<const N: usize>(
    tree_a: &RrtTree<N>,
    idx_a: usize,
    tree_b: &RrtTree<N>,
    idx_b: usize,
    a_is_start: bool,
) -> Vec<SRobotQ<N>> {
    let path_a = backtrack_path(tree_a, idx_a);
    let path_b = backtrack_path(tree_b, idx_b);

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

pub(crate) fn solve<const N: usize, TKN: Token>(
    start: &SRobotQ<N>,
    goal: &SRobotQ<N>,
    validator: &mut impl Validator<N, TKN>,
    settings: &KrrtcSettings<N>,
    rng: &mut impl Rand,
) -> (RevampResult<RobotPath, TKN>, RrtDiagnostic) {
    let timer = std::time::Instant::now();
    let coeffs = settings.kin_limits.velocity_coeffs();

    let direct_cost = time_optimal_cost(start, goal, &settings.kin_limits);
    if direct_cost < 1e-10 {
        let path: RobotPath = vec![(*start).into()].into_iter().collect();
        return (
            Ok(path),
            RrtDiagnostic {
                iterations: 0,
                start_tree_size: 1,
                goal_tree_size: 1,
                path_cost: 0.0,
                elapsed_ns: timer.elapsed().as_nanos(),
            },
        );
    }

    if validate_edge(start, goal, settings.resolution, validator).is_ok()
    {
        let path: RobotPath = vec![(*start).into(), (*goal).into()].into_iter().collect();
        return (
            Ok(path),
            RrtDiagnostic {
                iterations: 0,
                start_tree_size: 1,
                goal_tree_size: 1,
                path_cost: direct_cost,
                elapsed_ns: timer.elapsed().as_nanos(),
            },
        );
    }

    let mut start_tree = RrtTree::with_capacity(&coeffs, settings.max_samples / 2);
    let mut goal_tree = RrtTree::with_capacity(&coeffs, settings.max_samples / 2);

    start_tree.add(*start, 0, settings.radius, 0.0);
    goal_tree.add(*goal, 0, settings.radius, 0.0);

    let mut extend_start = true;

    for iteration in 0..settings.max_iterations {
        if start_tree.len() + goal_tree.len() >= settings.max_samples {
            break;
        }

        let q_rand = sample_uniform(rng, &settings.joint_lower, &settings.joint_upper);

        let result = if extend_start {
            extend_and_connect(
                &mut start_tree,
                &mut goal_tree,
                &q_rand,
                validator,
                settings,
            )
        } else {
            extend_and_connect(
                &mut goal_tree,
                &mut start_tree,
                &q_rand,
                validator,
                settings,
            )
        };

        if let Some((idx_a, idx_b)) = result {
            let (tree_a, tree_b) = if extend_start {
                (&start_tree, &goal_tree)
            } else {
                (&goal_tree, &start_tree)
            };
            let mut waypoints = reconstruct(tree_a, idx_a, tree_b, idx_b, extend_start);

            if settings.shortcut_iterations > 0 {
                shortcut(
                    &mut waypoints,
                    validator,
                    &settings.kin_limits,
                    settings.resolution,
                    settings.shortcut_iterations,
                    rng,
                );
            }

            if settings.smoothing_iterations > 0 {
                round_corners(
                    &mut waypoints,
                    validator,
                    &settings.kin_limits,
                    settings.resolution,
                    settings.smoothing_iterations,
                );
            }

            let cost = kinematic_path_cost(&waypoints, &settings.kin_limits);
            let path: RobotPath = waypoints.into_iter().map(|q| q.into()).collect();
            return (
                Ok(path),
                RrtDiagnostic {
                    iterations: iteration + 1,
                    start_tree_size: start_tree.len(),
                    goal_tree_size: goal_tree.len(),
                    path_cost: cost,
                    elapsed_ns: timer.elapsed().as_nanos(),
                },
            );
        }

        if settings.balance {
            let (a_len, b_len) = if extend_start {
                (start_tree.len(), goal_tree.len())
            } else {
                (goal_tree.len(), start_tree.len())
            };
            if a_len as f64 / b_len.max(1) as f64 > settings.tree_ratio {
                extend_start = !extend_start;
            }
        } else {
            extend_start = !extend_start;
        }
    }

    (
        Err(RevampError::OutOfIterations),
        RrtDiagnostic {
            iterations: settings.max_iterations,
            start_tree_size: start_tree.len(),
            goal_tree_size: goal_tree.len(),
            path_cost: f64::INFINITY,
            elapsed_ns: timer.elapsed().as_nanos(),
        },
    )
}

fn extend_and_connect<const N: usize, TKN: Token>(
    tree_a: &mut RrtTree<N>,
    tree_b: &mut RrtTree<N>,
    q_rand: &SRobotQ<N>,
    validator: &mut impl Validator<N, TKN>,
    settings: &KrrtcSettings<N>,
) -> Option<(usize, usize)> {
    let (near_idx, near_dist) = tree_a.nearest(q_rand);

    if settings.dynamic_domain && near_dist > tree_a.radius(near_idx) {
        let r = tree_a.radius(near_idx);
        tree_a.set_radius(near_idx, (r * (1.0 - settings.alpha)).max(settings.min_radius));
        return None;
    }

    let q_near = *tree_a.node(near_idx);
    let q_new = kinematic_steer(&q_near, q_rand, settings.range, &settings.kin_limits);

    if validate_edge(&q_near, &q_new, settings.resolution, validator).is_err()
    {
        if settings.dynamic_domain {
            let r = tree_a.radius(near_idx);
            tree_a.set_radius(near_idx, (r * (1.0 - settings.alpha)).max(settings.min_radius));
        }
        return None;
    }

    let new_a_idx = tree_a.add(q_new, near_idx, settings.radius, 0.0);

    if settings.dynamic_domain {
        let r = tree_a.radius(near_idx);
        tree_a.set_radius(near_idx, r * (1.0 + settings.alpha));
    }

    let (mut connect_idx, _) = tree_b.nearest(&q_new);

    loop {
        let q_connect = *tree_b.node(connect_idx);
        let dist = time_optimal_cost(&q_connect, &q_new, &settings.kin_limits);

        if dist < 1e-6 {
            return Some((new_a_idx, connect_idx));
        }

        let q_step = kinematic_steer(&q_connect, &q_new, settings.range, &settings.kin_limits);

        if validate_edge(&q_connect, &q_step, settings.resolution, validator).is_err()
        {
            return None;
        }

        let reached = time_optimal_cost(&q_step, &q_new, &settings.kin_limits) < 1e-6;
        let step_idx = if reached {
            tree_b.add(q_new, connect_idx, settings.radius, 0.0)
        } else {
            tree_b.add(q_step, connect_idx, settings.radius, 0.0)
        };

        if reached {
            return Some((new_a_idx, step_idx));
        }

        connect_idx = step_idx;
    }
}
