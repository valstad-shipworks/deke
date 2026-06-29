//! General (non-decomposable) 5R IK exercised through the public [`Kinematics`]
//! API. A random non-DH 5R chain has no closed-form decomposition, so it takes
//! the generic eigenvalue fallback (lifted to a generic 6R). Each chain is
//! FK-roundtripped: the planted configuration must be recovered and every
//! returned solution must reproduce the target pose. (Valid 4R chains always
//! decompose analytically; the 4R padding fallback is covered by the unit test
//! `generic_padded_recovers_planted` in `lib.rs`.)

use deke_kin::deke_types::{FKChain, IkOutcome, IkSolver, JointSpec, KinSpec, SRobotQ};
use deke_kin::glam::{DAffine3, DVec3};
use deke_kin::{JointLimits, Kinematics};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
    fn axis(&mut self) -> DVec3 {
        DVec3::new(
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
        )
        .normalize()
    }
    fn affine(&mut self) -> DAffine3 {
        DAffine3::from_axis_angle(self.axis(), self.range(-2.5, 2.5))
            * DAffine3::from_translation(DVec3::new(
                self.range(-0.3, 0.3),
                self.range(-0.3, 0.3),
                self.range(0.05, 0.3),
            ))
    }
}

/// A fully generic revolute chain (arbitrary joint frames and axes), built from
/// a [`KinSpec`] so it carries no DH regularity — defeating the analytic class
/// matchers and forcing the generic eigenvalue fallback.
fn generic_chain<const N: usize>(rng: &mut Rng) -> Kinematics<N, f64> {
    let joints: [(DAffine3, JointSpec<f64>); N] = std::array::from_fn(|_| {
        (
            rng.affine(),
            JointSpec::Revolute {
                axis_local: rng.axis(),
            },
        )
    });
    let spec = KinSpec::new(DAffine3::IDENTITY, joints, rng.affine());
    Kinematics::<N, f64>::from_kinspec(spec, JointLimits::symmetric(7.0), &[])
}

fn roundtrip<const N: usize>(seed: u64, rounds: usize) {
    let mut rng = Rng(seed);
    let mut generic_chains = 0usize;
    let mut total_solutions = 0usize;

    for _ in 0..rounds {
        let chain = generic_chain::<N>(&mut rng);

        // Only assert on chains that actually take the generic path; skip the
        // rare random chain that lands in a recognised analytic class.
        if chain.ik_diagnostic().is_analytic() {
            continue;
        }
        generic_chains += 1;

        let q_arr: [f64; N] = std::array::from_fn(|_| rng.range(-2.6, 2.6));
        let q = SRobotQ::<N, f64>::from_array(q_arr);
        let target = chain.fk_end(&q).unwrap();
        let target_cols = target.to_cols_array();

        let sols = match chain.ik(target).unwrap() {
            IkOutcome::Solved(s) => s,
            _ => panic!("{N}R generic chain returned no solutions\n q={q_arr:?}"),
        };
        total_solutions += sols.len();

        // Every returned solution must reproduce the target pose.
        for s in &sols {
            let got = chain.fk_end(s).unwrap().to_cols_array();
            assert!(
                got.iter()
                    .zip(target_cols.iter())
                    .all(|(p, q)| (p - q).abs() < 1e-6),
                "{N}R solution does not reach target\n sol={:?}",
                s.as_slice()
            );
        }

        // The planted configuration must be among the solutions.
        let recovered = sols.iter().any(|s| {
            s.as_slice()
                .iter()
                .zip(q_arr.iter())
                .all(|(a, b)| {
                    let d = (a - b).rem_euclid(std::f64::consts::TAU);
                    d < 1e-5 || (std::f64::consts::TAU - d) < 1e-5
                })
        });
        assert!(
            recovered,
            "{N}R planted configuration not recovered\n q={q_arr:?}\n got {} solutions",
            sols.len()
        );
    }

    eprintln!("{N}R: {generic_chains}/{rounds} generic chains, {total_solutions} solutions, all roundtrip");
    assert!(
        generic_chains >= 20,
        "too few non-decomposable {N}R chains exercised the generic path: {generic_chains}/{rounds}"
    );
}

#[test]
fn general_5r_roundtrips() {
    roundtrip::<5>(0x0fed_cba9_8765_4321, 400);
}
