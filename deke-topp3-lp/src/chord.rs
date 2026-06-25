//! Joint-space path conditioning: turn the raw polyline into the knots the σ-LP
//! times, parameterised by joint-space arc length. No FK is involved — the chord
//! is the joint-space polyline itself, so every conditioned knot lies on it.

use deke_types::SRobotQ;

use crate::constraints::Conditioning;

/// Euclidean joint-space distance between two configurations.
pub(crate) fn joint_distance<const N: usize>(a: &SRobotQ<N, f64>, b: &SRobotQ<N, f64>) -> f64 {
    (0..N)
        .map(|j| {
            let d = a.0[j] - b.0[j];
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Condition the raw joint path into knots and their cumulative joint-space arc
/// length. Coincident knots are dropped first (a zero-length chord makes the
/// secant slope blow up). `Collinear` densification inserts on-chord knots, so
/// the result is always a refinement of the input polyline — zero deviation.
pub(crate) fn condition<const N: usize>(
    raw: &[SRobotQ<N, f64>],
    conditioning: Conditioning,
) -> (Vec<SRobotQ<N, f64>>, Vec<f64>) {
    let mut qd: Vec<SRobotQ<N, f64>> = Vec::with_capacity(raw.len());
    let mut sd: Vec<f64> = Vec::with_capacity(raw.len());
    for &q in raw {
        match qd.last() {
            Some(prev) => {
                let d = joint_distance(prev, &q);
                if d > 1e-9 {
                    sd.push(sd[sd.len() - 1] + d);
                    qd.push(q);
                }
            }
            None => {
                qd.push(q);
                sd.push(0.0);
            }
        }
    }
    let n = qd.len();

    match conditioning {
        Conditioning::Collinear(res) if res > 0.0 && n >= 2 => {
            let mut out = vec![qd[0]];
            let mut s_out = vec![0.0f64];
            for i in 0..n - 1 {
                let h = (sd[i + 1] - sd[i]).max(1e-12);
                let k = ((h / res).ceil() as usize).max(1);
                for ss in 1..=k {
                    let b = ss as f64 / k as f64;
                    let a = 1.0 - b;
                    out.push(SRobotQ(std::array::from_fn(|j| {
                        qd[i].0[j] * a + qd[i + 1].0[j] * b
                    })));
                    s_out.push(sd[i] + h * b);
                }
            }
            (out, s_out)
        }
        _ => (qd, sd),
    }
}

/// Drop consecutive coincident waypoints (joint distance ≤ `1e-9`).
pub(crate) fn dedup<const N: usize>(raw: &[SRobotQ<N, f64>]) -> Vec<SRobotQ<N, f64>> {
    let mut out: Vec<SRobotQ<N, f64>> = Vec::with_capacity(raw.len());
    for &q in raw {
        match out.last() {
            Some(p) if joint_distance(p, &q) <= 1e-9 => {}
            _ => out.push(q),
        }
    }
    out
}

/// Split the polyline at vertices whose joint-space turn angle exceeds `angle`,
/// into runs that each start and stop at rest. Consecutive runs share the corner
/// vertex, so concatenating their (rest-to-rest) timings stops exactly on the
/// kink. Returns the whole path as one run when it has no sharp vertex.
pub(crate) fn split_sharp<const N: usize>(
    wps: &[SRobotQ<N, f64>],
    angle: f64,
) -> Vec<Vec<SRobotQ<N, f64>>> {
    if wps.len() < 3 {
        return vec![wps.to_vec()];
    }
    let dir = |a: &SRobotQ<N, f64>, b: &SRobotQ<N, f64>| -> [f64; N] {
        let d = joint_distance(a, b).max(1e-12);
        std::array::from_fn(|j| (b.0[j] - a.0[j]) / d)
    };
    let cos_thresh = angle.cos();
    let mut runs: Vec<Vec<SRobotQ<N, f64>>> = Vec::new();
    let mut start = 0;
    for i in 1..wps.len() - 1 {
        let d0 = dir(&wps[i - 1], &wps[i]);
        let d1 = dir(&wps[i], &wps[i + 1]);
        let dot: f64 = (0..N).map(|j| d0[j] * d1[j]).sum::<f64>().clamp(-1.0, 1.0);
        if dot < cos_thresh {
            runs.push(wps[start..=i].to_vec());
            start = i;
        }
    }
    runs.push(wps[start..].to_vec());
    runs
}

/// Per-segment secant slope `dq/ds` of the chord-linear path. The joint v/a/j of
/// the output are exactly `secantᵦ · Δᵐσ / dtᵐ` within a segment.
pub(crate) fn secants<const N: usize>(knots: &[SRobotQ<N, f64>], s: &[f64]) -> Vec<[f64; N]> {
    (0..knots.len().saturating_sub(1))
        .map(|b| {
            let ds = (s[b + 1] - s[b]).max(1e-12);
            std::array::from_fn(|j| (knots[b + 1].0[j] - knots[b].0[j]) / ds)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q<const N: usize>(a: [f64; N]) -> SRobotQ<N, f64> {
        SRobotQ(a)
    }

    #[test]
    fn collinear_keeps_every_knot_on_the_chord() {
        let raw = [q([0.0, 0.0]), q([1.0, 0.0]), q([1.0, 1.0])];
        let (knots, s) = condition(&raw, Conditioning::Collinear(0.2));
        assert!(knots.len() >= 8, "corner densified");
        assert!((s[s.len() - 1] - 2.0).abs() < 1e-9);
        for w in s.windows(2) {
            assert!(w[1] >= w[0]);
        }
        for k in &knots {
            let on_a = k.0[1].abs() < 1e-9;
            let on_b = (k.0[0] - 1.0).abs() < 1e-9;
            assert!(on_a || on_b, "knot off both legs: {:?}", k.0);
        }
    }

    #[test]
    fn raw_keeps_waypoints_and_arc() {
        let raw = [q([0.0, 0.0]), q([3.0, 4.0])];
        let (knots, s) = condition(&raw, Conditioning::Raw);
        assert_eq!(knots.len(), 2);
        assert!((s[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn dedup_drops_coincident_knots() {
        let raw = [q([0.0]), q([0.0]), q([1.0])];
        let (knots, _) = condition(&raw, Conditioning::Raw);
        assert_eq!(knots.len(), 2);
    }
}
