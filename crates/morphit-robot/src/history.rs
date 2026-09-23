//! Loss-history thinning for the `X-Morphit-Loss` header and `loss_history`
//! fields, a port of the API's `_subsample_history`.

/// Default cap on the number of points.
pub const MAX_POINTS: usize = 150;

/// Keep at most `max_points` `(iteration, loss)` pairs, picked like
/// `np.unique(np.linspace(0, n - 1, max_points).astype(int))` (so the first
/// and last points always survive and fewer than `max_points` may remain),
/// and round every loss to 5 significant digits (`float(f"{v:.5g}")`).
pub fn subsample_history(history: &[(usize, f64)], max_points: usize) -> Vec<(usize, f64)> {
    let n = history.len();
    let picked: Vec<(usize, f64)> = if n > max_points && max_points > 0 {
        let mut idx = Vec::with_capacity(max_points);
        let stop = (n - 1) as f64;
        let step = if max_points > 1 { stop / (max_points - 1) as f64 } else { 0.0 };
        for k in 0..max_points {
            // numpy: arange * step + start, with the endpoint pinned to stop.
            let x = if k + 1 == max_points && max_points > 1 { stop } else { k as f64 * step };
            idx.push(x as usize);
        }
        idx.dedup(); // monotone, so dedup == np.unique
        idx.into_iter().map(|i| history[i]).collect()
    } else {
        history.to_vec()
    };
    picked.into_iter().map(|(i, v)| (i, round_sig5(v))).collect()
}

/// `float(f"{v:.5g}")`.
pub fn round_sig5(v: f64) -> f64 {
    if !v.is_finite() || v == 0.0 {
        return v;
    }
    format!("{v:.4e}").parse().unwrap_or(v)
}

/// Compact JSON `[[iter,loss],...]`, as `json.dumps(..., separators=(",",":"))`.
pub fn history_json(points: &[(usize, f64)]) -> String {
    serde_json::to_string(&points.iter().map(|&(i, v)| (i, v)).collect::<Vec<_>>()).expect("pairs serialize")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist(n: usize) -> Vec<(usize, f64)> {
        (0..n).map(|i| (i, 1.0 / (i as f64 + 1.0))).collect()
    }

    #[test]
    fn short_histories_are_only_rounded() {
        let h = subsample_history(&hist(150), MAX_POINTS);
        assert_eq!(h.len(), 150);
        assert_eq!(h[2], (2, 0.33333));
    }

    #[test]
    fn long_histories_match_numpy() {
        // np.unique(np.linspace(0, 150, 150).astype(int)) has 150 entries,
        // skipping exactly one index; np.linspace(0, 999, 150) gives 150.
        let h = subsample_history(&hist(151), MAX_POINTS);
        assert_eq!(h.len(), 150);
        assert_eq!(h.first().unwrap().0, 0);
        assert_eq!(h.last().unwrap().0, 150);
        let h = subsample_history(&hist(1000), MAX_POINTS);
        assert_eq!(h.len(), 150);
        assert_eq!((h[0].0, h[1].0, h[2].0, h[149].0), (0, 6, 13, 999));
        // Python: [int(x) for x in np.linspace(0, 999, 150)][:5] == [0, 6, 13, 20, 26]
        assert_eq!(h[3].0, 20);
        assert_eq!(h[4].0, 26);
        // (n, sum of kept indices, count) from numpy.
        for (n, sum, len) in [
            (151, 11176, 150),
            (153, 11326, 150),
            (197, 14626, 150),
            (200, 14851, 150),
            (299, 22350, 150),
            (300, 22351, 150),
            (333, 24826, 150),
            (500, 37351, 150),
            (777, 58126, 150),
            (1000, 74851, 150),
        ] {
            let h = subsample_history(&hist(n), MAX_POINTS);
            assert_eq!((h.iter().map(|p| p.0).sum::<usize>(), h.len()), (sum, len), "n = {n}");
        }
    }

    #[test]
    fn five_significant_digits() {
        assert_eq!(round_sig5(12.345678), 12.346);
        assert_eq!(round_sig5(0.000123456), 0.00012346);
        assert_eq!(round_sig5(123456789.0), 123460000.0);
        assert_eq!(round_sig5(-2.5), -2.5);
        assert_eq!(history_json(&[(0, 1.5), (1, 1e-5)]), "[[0,1.5],[1,0.00001]]");
    }
}
