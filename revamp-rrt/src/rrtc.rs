use revamp_types::{RevampError, RevampResult, RobotPath, SRobotQ, Token, Validator};
use tinyrand::Rand;

use crate::tree::RrtTree;
use crate::RrtDiagnostic;

#[derive(Debug, Clone)]
pub struct RrtcSettings<const N: usize> {
    pub range: f64,
    pub max_iterations: usize,
    pub max_samples: usize,
    pub joint_lower: SRobotQ<N>,
    pub joint_upper: SRobotQ<N>,
    pub joint_coefficients: SRobotQ<N>,
    pub resolution: f64,
    pub dynamic_domain: bool,
    pub radius: f64,
    pub alpha: f64,
    pub min_radius: f64,
    pub balance: bool,
    pub tree_ratio: f64,
    pub seed: u64,
}

impl<const N: usize> RrtcSettings<N> {
    pub fn new(lower: SRobotQ<N>, upper: SRobotQ<N>) -> Self {
        Self {
            range: 0.5,
            max_iterations: 100_000,
            max_samples: 100_000,
            joint_lower: lower,
            joint_upper: upper,
            joint_coefficients: SRobotQ::splat(1.0),
            resolution: 0.05,
            dynamic_domain: true,
            radius: 4.0,
            alpha: 0.0001,
            min_radius: 1.0,
            balance: true,
            tree_ratio: 1.0,
            seed: 42,
        }
    }
}

#[inline]
pub(crate) fn rand_f64(rng: &mut impl Rand) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

pub(crate) fn sample_uniform<const N: usize>(
    rng: &mut impl Rand,
    lower: &SRobotQ<N>,
    upper: &SRobotQ<N>,
) -> SRobotQ<N> {
    let mut q = [0.0f32; N];
    for i in 0..N {
        q[i] = (lower[i] as f64 + rand_f64(rng) * (upper[i] as f64 - lower[i] as f64)) as f32;
    }
    SRobotQ::from_array(q)
}

pub(crate) fn weighted_distance<const N: usize>(
    a: &SRobotQ<N>,
    b: &SRobotQ<N>,
    coeffs: &[f64; N],
) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        let d = a.0[i] as f64 - b.0[i] as f64;
        sum += coeffs[i] * d * d;
    }
    sum.sqrt()
}

pub(crate) fn path_cost<const N: usize>(path: &[SRobotQ<N>], coeffs: &[f64; N]) -> f64 {
    path.windows(2)
        .map(|w| weighted_distance(&w[0], &w[1], coeffs))
        .sum()
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

pub(crate) fn validate_edge<const N: usize, TKN: Token>(
    from: &SRobotQ<N>,
    to: &SRobotQ<N>,
    resolution: f64,
    validator: &mut impl Validator<N, TKN>,
) -> RevampResult<(), TKN> {
    let dist = from.distance(to) as f64;
    let steps = ((dist / resolution).ceil() as usize).max(1);
    let mut points = Vec::with_capacity(steps);
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        points.push(*from + (*to - *from) * t);
    }
    validator.validate_motion(&points)
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
    settings: &RrtcSettings<N>,
    rng: &mut impl Rand,
) -> (RevampResult<RobotPath, TKN>, RrtDiagnostic) {
    let timer = std::time::Instant::now();
    let coeffs = {
        let mut c = [0.0f64; N];
        for i in 0..N {
            c[i] = settings.joint_coefficients.0[i] as f64;
        }
        c
    };

    let direct_dist = weighted_distance(start, goal, &coeffs);
    if direct_dist < 1e-10 {
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

    if validate_edge(start, goal, settings.resolution, validator).is_ok() {
        let path: RobotPath = vec![(*start).into(), (*goal).into()].into_iter().collect();
        return (
            Ok(path),
            RrtDiagnostic {
                iterations: 0,
                start_tree_size: 1,
                goal_tree_size: 1,
                path_cost: direct_dist,
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
                &coeffs,
            )
        } else {
            extend_and_connect(
                &mut goal_tree,
                &mut start_tree,
                &q_rand,
                validator,
                settings,
                &coeffs,
            )
        };

        if let Some((idx_a, idx_b)) = result {
            let (tree_a, tree_b) = if extend_start {
                (&start_tree, &goal_tree)
            } else {
                (&goal_tree, &start_tree)
            };
            let waypoints = reconstruct(tree_a, idx_a, tree_b, idx_b, extend_start);
            let cost = path_cost(&waypoints, &coeffs);
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
    settings: &RrtcSettings<N>,
    coeffs: &[f64; N],
) -> Option<(usize, usize)> {
    let (near_idx, near_dist) = tree_a.nearest(q_rand);

    if settings.dynamic_domain && near_dist > tree_a.radius(near_idx) {
        let r = tree_a.radius(near_idx);
        tree_a.set_radius(near_idx, (r * (1.0 - settings.alpha)).max(settings.min_radius));
        return None;
    }

    let q_near = *tree_a.node(near_idx);
    let q_new = steer(&q_near, q_rand, settings.range, coeffs);

    if validate_edge(&q_near, &q_new, settings.resolution, validator).is_err() {
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
        let dist = weighted_distance(&q_connect, &q_new, coeffs);

        if dist < 1e-6 {
            return Some((new_a_idx, connect_idx));
        }

        let q_step = steer(&q_connect, &q_new, settings.range, coeffs);

        if validate_edge(&q_connect, &q_step, settings.resolution, validator).is_err() {
            return None;
        }

        let reached = weighted_distance(&q_step, &q_new, coeffs) < 1e-6;
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
