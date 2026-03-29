use kiddo::KdTree;
use kiddo::SquaredEuclidean;
use revamp_types::SRobotQ;

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

    pub fn within(&self, q: &SRobotQ<N>, radius: f64) -> Vec<(usize, f64)> {
        let scaled = self.scale(q);
        self.kdtree
            .within::<SquaredEuclidean>(&scaled, radius * radius)
            .into_iter()
            .map(|nn| (nn.item as usize, nn.distance.sqrt()))
            .collect()
    }

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
}
