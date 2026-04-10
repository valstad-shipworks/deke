use deke_types::SRobotQ;

#[derive(Debug, Clone, Copy)]
pub struct JointKinLimits {
    pub v_max: f64,
    pub a_max: f64,
    pub j_max: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct KinematicLimits<const N: usize> {
    pub joints: [JointKinLimits; N],
}

impl<const N: usize> KinematicLimits<N> {
    /// Derives velocity-scaled coefficients for KDTree nearest-neighbor queries.
    /// coeff[i] = 1/v_max[i]^2 so that the Euclidean distance in scaled space
    /// approximates the time-optimal cost (lower bound).
    pub fn velocity_coeffs(&self) -> [f64; N] {
        let mut c = [0.0; N];
        for i in 0..N {
            let v = self.joints[i].v_max;
            c[i] = 1.0 / (v * v);
        }
        c
    }
}

/// Computes the minimum time for a rest-to-rest 1D motion of displacement
/// `delta_q` under jerk-limited (S-curve) constraints.
///
/// The S-curve profile has up to 7 phases:
///   jerk+ | const accel | jerk- | coast | jerk- | const decel | jerk+
///
/// Three regimes depending on whether a_max and v_max are reached.
pub fn min_time_1d(delta_q: f64, limits: &JointKinLimits) -> f64 {
    let d = delta_q.abs();
    if d < 1e-12 {
        return 0.0;
    }

    let v = limits.v_max;
    let a = limits.a_max;
    let j = limits.j_max;

    // Time for jerk phase to reach a_max
    let t_j = a / j;

    // Displacement consumed by a full accel+decel ramp (triangular accel profile only, no coast)
    // when a_max is just barely reached (t_a = 0):
    // Each side: jerk-up for t_j, jerk-down for t_j → peak accel = a, peak vel = a*t_j
    // Displacement per side = a*t_j^2 = a^3/j^2
    // Both sides: 2 * a*t_j^2
    let d_min_reach_a = 2.0 * a * t_j * t_j;

    if d < d_min_reach_a {
        // Case 1: Can't reach a_max — triangular jerk profile
        // Each side has one jerk phase of duration t_j'
        // Peak accel = j * t_j', peak vel = j * t_j'^2 / 2
        // Displacement = 2 * (j * t_j'^3 / 6 + j * t_j'^2 / 2 * t_j' - j * t_j'^3 / 6)
        // Simplified: d = j * t_j'^3
        let tj_prime = (d / j).cbrt();
        return 4.0 * tj_prime;
    }

    // Velocity reached with full jerk phases and no constant-accel cruise:
    // v_jerk = a * t_j = a^2/j
    let v_jerk = a * t_j;

    // Displacement if we reach a_max and v_max with constant-accel phase:
    // t_a chosen so that v_max = a*(t_j + t_a) - but we need to check
    // v_max reachable? Need t_a >= 0 → v_max >= v_jerk = a^2/j
    if v >= v_jerk {
        // Can potentially reach v_max. Check displacement needed.
        let t_a = (v - v_jerk) / a;

        // Displacement for accel ramp (jerk-up + const-accel + jerk-down):
        let d_no_coast = v * (2.0 * t_j + t_a);

        if d < d_no_coast {
            // Case 2: Reaches a_max but not v_max
            // Solve for the actual peak velocity v_peak < v_max
            // d = v_peak * (2*t_j + t_a') where t_a' = (v_peak - v_jerk)/a
            // d = v_peak * (2*t_j + (v_peak - a*t_j)/a)
            // d = v_peak * (2*t_j + v_peak/a - t_j)
            // d = v_peak * (t_j + v_peak/a)
            // d = v_peak * t_j + v_peak^2 / a
            // v_peak^2/a + v_peak*t_j - d = 0
            // v_peak = (-t_j + sqrt(t_j^2 + 4*d/a)) / 2 * a ... quadratic
            // v_peak^2 + a*t_j*v_peak - a*d = 0
            let discriminant = a * t_j * a * t_j + 4.0 * a * d;
            let v_peak = (-a * t_j + discriminant.sqrt()) / 2.0;
            let t_a_actual = (v_peak - v_jerk) / a;

            if t_a_actual < 0.0 {
                // Falls back to case 1 territory (shouldn't happen given our checks, but guard)
                let tj_prime = (d / j).cbrt();
                return 4.0 * tj_prime;
            }

            return 2.0 * (2.0 * t_j + t_a_actual);
        }

        // Case 3: Full 7-phase S-curve — reaches both a_max and v_max
        let t_coast = (d - d_no_coast) / v;
        return 2.0 * t_j + t_a + t_coast + 2.0 * t_j + t_a;
        // = 4*t_j + 2*t_a + t_coast
    }

    // v_max < a^2/j: velocity limit reached before acceleration limit
    // The profile never reaches a_max. Only jerk phases + coast.
    // Accel ramp: jerk+ for t_j', jerk- for t_j' where j*t_j' = v/t_j' → t_j' = sqrt(v/j)
    let tj_prime = (v / j).sqrt();

    // Displacement for accel+decel (no coast): d_no_coast = 2 * v * tj_prime
    let d_no_coast = 2.0 * v * tj_prime;

    if d < d_no_coast {
        // Can't reach v_max either — pure triangular (same as case 1)
        let tj_solve = (d / j).cbrt();
        return 4.0 * tj_solve;
    }

    // Coast phase
    let t_coast = (d - d_no_coast) / v;
    4.0 * tj_prime + t_coast
}

/// Computes the time-optimal cost between two configurations.
/// The bottleneck joint (longest single-joint time) determines the cost.
pub fn time_optimal_cost<const N: usize>(
    a: &SRobotQ<N>,
    b: &SRobotQ<N>,
    limits: &KinematicLimits<N>,
) -> f64 {
    let mut max_time = 0.0f64;
    for i in 0..N {
        let delta = (b.0[i] as f64 - a.0[i] as f64).abs();
        let t = min_time_1d(delta, &limits.joints[i]);
        max_time = max_time.max(t);
    }
    max_time
}

/// Quintic Hermite interpolation: zero velocity and acceleration at endpoints.
/// Maps t in [0,1] → s in [0,1] with s'(0)=s'(1)=0, s''(0)=s''(1)=0.
#[inline]
pub fn quintic_interp(t: f64) -> f64 {
    let t3 = t * t * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    6.0 * t5 - 15.0 * t4 + 10.0 * t3
}

/// Interpolates between two configurations using quintic blending.
/// t_normalized in [0, 1].
pub fn kinematic_interpolate<const N: usize>(
    from: &SRobotQ<N>,
    to: &SRobotQ<N>,
    t_normalized: f64,
) -> SRobotQ<N> {
    let s = quintic_interp(t_normalized) as f32;
    *from + (*to - *from) * s
}

/// Computes the time-optimal path cost (sum of segment times).
pub fn kinematic_path_cost<const N: usize>(
    path: &[SRobotQ<N>],
    limits: &KinematicLimits<N>,
) -> f64 {
    path.windows(2)
        .map(|w| time_optimal_cost(&w[0], &w[1], limits))
        .sum()
}

/// Computes the cosine of the direction change at waypoint `b` between
/// segments a→b and b→c, measured in velocity-scaled joint space.
/// Returns 1.0 for a straight path, -1.0 for a full reversal.
pub fn direction_cosine<const N: usize>(
    a: &SRobotQ<N>,
    b: &SRobotQ<N>,
    c: &SRobotQ<N>,
    limits: &KinematicLimits<N>,
) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_ab = 0.0f64;
    let mut norm_bc = 0.0f64;
    for i in 0..N {
        let s = 1.0 / limits.joints[i].v_max;
        let ab = (b.0[i] as f64 - a.0[i] as f64) * s;
        let bc = (c.0[i] as f64 - b.0[i] as f64) * s;
        dot += ab * bc;
        norm_ab += ab * ab;
        norm_bc += bc * bc;
    }
    let denom = (norm_ab * norm_bc).sqrt();
    if denom < 1e-12 {
        return 1.0;
    }
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical_limits() -> JointKinLimits {
        JointKinLimits {
            v_max: 3.0,
            a_max: 10.0,
            j_max: 50.0,
        }
    }

    #[test]
    fn zero_displacement() {
        assert_eq!(min_time_1d(0.0, &typical_limits()), 0.0);
    }

    #[test]
    fn negative_displacement_same_as_positive() {
        let lim = typical_limits();
        let t_pos = min_time_1d(1.0, &lim);
        let t_neg = min_time_1d(-1.0, &lim);
        assert!((t_pos - t_neg).abs() < 1e-12);
    }

    #[test]
    fn monotonically_increasing_with_displacement() {
        let lim = typical_limits();
        let mut prev = 0.0;
        for d in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let t = min_time_1d(d, &lim);
            assert!(t > prev, "time should increase: d={d}, t={t}, prev={prev}");
            prev = t;
        }
    }

    #[test]
    fn time_positive_for_nonzero_displacement() {
        let lim = typical_limits();
        for d in [0.001, 0.1, 1.0, 10.0] {
            let t = min_time_1d(d, &lim);
            assert!(t > 0.0, "d={d} should give positive time, got {t}");
        }
    }

    #[test]
    fn very_small_displacement_uses_triangular_jerk() {
        let lim = JointKinLimits {
            v_max: 10.0,
            a_max: 100.0,
            j_max: 1000.0,
        };
        // Very small displacement: d = j * t_j'^3 → t_j' = (d/j)^(1/3)
        let d = 0.001;
        let t = min_time_1d(d, &lim);
        let expected = 4.0 * (d / lim.j_max).cbrt();
        assert!((t - expected).abs() < 1e-10, "expected {expected}, got {t}");
    }

    #[test]
    fn large_displacement_has_coast_phase() {
        let lim = JointKinLimits {
            v_max: 1.0,
            a_max: 5.0,
            j_max: 25.0,
        };
        // Large enough displacement that coast phase appears
        let d = 100.0;
        let t = min_time_1d(d, &lim);
        // Coast time dominates: t ≈ d/v_max for very large d
        assert!(t > d / lim.v_max, "should be at least d/v_max");
        // But not too much more
        assert!(t < d / lim.v_max + 10.0, "coast should dominate");
    }

    #[test]
    fn quintic_endpoints() {
        assert!((quintic_interp(0.0)).abs() < 1e-15);
        assert!((quintic_interp(1.0) - 1.0).abs() < 1e-15);
        assert!((quintic_interp(0.5) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn time_optimal_cost_bottleneck() {
        let limits = KinematicLimits {
            joints: [
                JointKinLimits {
                    v_max: 10.0,
                    a_max: 50.0,
                    j_max: 200.0,
                },
                JointKinLimits {
                    v_max: 1.0,
                    a_max: 5.0,
                    j_max: 20.0,
                },
            ],
        };
        let a = SRobotQ([0.0, 0.0]);
        let b = SRobotQ([1.0, 1.0]);
        let cost = time_optimal_cost(&a, &b, &limits);
        // Joint 1 is fast, joint 0 is slow for the same displacement
        let t0 = min_time_1d(1.0, &limits.joints[0]);
        let t1 = min_time_1d(1.0, &limits.joints[1]);
        assert_eq!(cost, t0.max(t1));
        assert!(t1 > t0, "joint 1 should be the bottleneck");
    }
}
