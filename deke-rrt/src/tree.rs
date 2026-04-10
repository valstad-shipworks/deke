use kiddo::KdTree;
use kiddo::SquaredEuclidean;
use deke_types::SRobotQ;

pub(crate) struct RrtTree<const N: usize> {
    nodes: Vec<SRobotQ<N>>,
    parents: Vec<usize>,
    radii: Vec<f64>,
    costs: Vec<f64>,
    kdtree: KdTree<f64, N>,
    sqrt_coeffs: [f64; N],
}

impl<const N: usize> RrtTree<N> {
    pub fn with_capacity(coefficients: &[f64; N], cap: usize) -> Self {
        let mut sqrt_coeffs = [0.0; N];
        for i in 0..N {
            sqrt_coeffs[i] = coefficients[i].sqrt();
        }
        Self {
            nodes: Vec::with_capacity(cap),
            parents: Vec::with_capacity(cap),
            radii: Vec::with_capacity(cap),
            costs: Vec::with_capacity(cap),
            kdtree: KdTree::with_capacity(cap),
            sqrt_coeffs,
        }
    }

    fn scale(&self, q: &SRobotQ<N>) -> [f64; N] {
        let mut scaled = [0.0; N];
        for i in 0..N {
            scaled[i] = q.0[i] as f64 * self.sqrt_coeffs[i];
        }
        scaled
    }

    pub fn add(&mut self, q: SRobotQ<N>, parent: usize, radius: f64, cost: f64) -> usize {
        let idx = self.nodes.len();
        let scaled = self.scale(&q);
        self.kdtree.add(&scaled, idx as u64);
        self.nodes.push(q);
        self.parents.push(parent);
        self.radii.push(radius);
        self.costs.push(cost);
        idx
    }

    pub fn nearest(&self, q: &SRobotQ<N>) -> (usize, f64) {
        let scaled = self.scale(q);
        let nn = self.kdtree.nearest_one::<SquaredEuclidean>(&scaled);
        (nn.item as usize, nn.distance.sqrt())
    }

    // pub fn within(&self, q: &SRobotQ<N>, radius: f64) -> Vec<(usize, f64)> {
    //     let scaled = self.scale(q);
    //     self.kdtree
    //         .within::<SquaredEuclidean>(&scaled, radius * radius)
    //         .into_iter()
    //         .map(|nn| (nn.item as usize, nn.distance.sqrt()))
    //         .collect()
    // }

    #[inline]
    pub fn node(&self, idx: usize) -> &SRobotQ<N> {
        &self.nodes[idx]
    }

    #[inline]
    pub fn parent(&self, idx: usize) -> usize {
        self.parents[idx]
    }

    #[inline]
    pub fn radius(&self, idx: usize) -> f64 {
        self.radii[idx]
    }

    #[inline]
    pub fn set_radius(&mut self, idx: usize, r: f64) {
        self.radii[idx] = r;
    }

    #[inline]
    pub fn cost(&self, idx: usize) -> f64 {
        self.costs[idx]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    // pub fn dist_to_node(&self, q: &SRobotQ<N>, idx: usize) -> f64 {
    //     let mut sum = 0.0;
    //     for i in 0..N {
    //         let d = (q.0[i] - self.nodes[idx].0[i]) as f64 * self.sqrt_coeffs[i];
    //         sum += d * d;
    //     }
    //     sum.sqrt()
    // }

    /// Finds the nearest node satisfying `node.cost + geo_dist(q, node) <= cost_bound`.
    /// Returns (index, weighted_geometric_distance).
    /// Falls back to root (index 0) if no node satisfies the constraint.
    pub fn find_nearest_ao(&self, q: &SRobotQ<N>, cost_bound: f64) -> (usize, f64) {
        let mut scaled = [0.0f64; N];
        for i in 0..N {
            scaled[i] = q.0[i] as f64 * self.sqrt_coeffs[i];
        }

        let mut best_idx = 0;
        let mut best_total = f64::INFINITY;
        let mut best_geo_dist = 0.0;

        for i in 0..self.nodes.len() {
            let cost_i = self.costs[i];
            if cost_i >= cost_bound {
                continue;
            }
            let remaining = cost_bound - cost_i;
            let remaining_sq = remaining * remaining;

            let mut dist_sq = 0.0;
            let node = &self.nodes[i];
            for j in 0..N {
                let d = scaled[j] - node.0[j] as f64 * self.sqrt_coeffs[j];
                dist_sq += d * d;
                if dist_sq > remaining_sq {
                    break;
                }
            }

            if dist_sq <= remaining_sq {
                let geo_dist = dist_sq.sqrt();
                let total = cost_i + geo_dist;
                if total < best_total {
                    best_idx = i;
                    best_total = total;
                    best_geo_dist = geo_dist;
                }
            }
        }

        (best_idx, best_geo_dist)
    }
}
