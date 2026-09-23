//! Scalar helpers that reproduce PyTorch semantics exactly where it matters.

/// PyTorch's softplus threshold (beta = 1): above it softplus is the identity.
pub const SOFTPLUS_THRESHOLD: f64 = 20.0;

/// `torch.nn.functional.softplus(x)` with beta = 1, threshold = 20.
#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > SOFTPLUS_THRESHOLD { x } else { x.exp().ln_1p() }
}

/// Derivative of [`softplus`] as computed by PyTorch's backward (sigmoid, or 1 above the threshold).
#[inline]
pub fn softplus_grad(x: f64) -> f64 {
    if x > SOFTPLUS_THRESHOLD {
        1.0
    } else {
        let z = x.exp();
        z / (z + 1.0)
    }
}

/// Inverse of softplus: `x + ln(-expm1(-x))`, as in `morphit.py::_inverse_softplus`.
#[inline]
pub fn inv_softplus(y: f64) -> f64 {
    y + (-(-y).exp_m1()).ln()
}

/// `relu(x)`.
#[inline]
pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

/// Sign with `sign(0) = 0`, matching the gradient of `torch.abs`.
#[inline]
pub fn sgn(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `torch.median` of a 1-D tensor: the lower of the two middle values for even lengths.
/// Returns `None` for an empty slice.
pub fn torch_median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[(v.len() - 1) / 2])
}

/// Index and value of the smallest element; ties resolve to the first index.
/// Panics on an empty slice.
#[inline]
pub fn argmin_first(values: &[f64]) -> (usize, f64) {
    let mut best = 0;
    let mut best_v = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v < best_v {
            best = i;
            best_v = v;
        }
    }
    (best, best_v)
}

/// Index and value of the largest element; ties resolve to the first index (like `torch.argmax`).
/// Panics on an empty slice.
#[inline]
pub fn argmax_first(values: &[f64]) -> (usize, f64) {
    let mut best = 0;
    let mut best_v = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best_v {
            best = i;
            best_v = v;
        }
    }
    (best, best_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_round_trip() {
        let mut x = -30.0;
        while x <= 30.0 {
            let y = softplus(x);
            assert!(y > 0.0);
            let back = inv_softplus(y);
            let tol = 1e-9 * x.abs().max(1.0);
            // Very negative x: softplus(x) ~ e^x, the inverse is still accurate to ~1e-9 relative.
            assert!((back - x).abs() < tol.max(1e-6), "x={x} back={back}");
            x += 0.25;
        }
    }

    #[test]
    fn softplus_threshold_is_identity_with_unit_grad() {
        assert_eq!(softplus(20.5), 20.5);
        assert_eq!(softplus_grad(20.5), 1.0);
        assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-15);
        assert!((softplus_grad(0.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn softplus_grad_matches_finite_difference() {
        for &x in &[-5.0, -1.0, 0.0, 0.7, 3.0, 19.0] {
            let h = 1e-6;
            let fd = (softplus(x + h) - softplus(x - h)) / (2.0 * h);
            assert!((fd - softplus_grad(x)).abs() < 1e-8, "x={x}");
        }
    }

    #[test]
    fn median_is_lower_middle() {
        assert_eq!(torch_median(&[4.0, 1.0, 3.0, 2.0]), Some(2.0));
        assert_eq!(torch_median(&[5.0, 1.0, 3.0]), Some(3.0));
        assert_eq!(torch_median(&[]), None);
    }

    #[test]
    fn arg_extrema_take_first_on_ties() {
        assert_eq!(argmin_first(&[3.0, 1.0, 1.0, 2.0]), (1, 1.0));
        assert_eq!(argmax_first(&[3.0, 5.0, 5.0, 2.0]), (1, 5.0));
    }

    #[test]
    fn sign_of_zero_is_zero() {
        assert_eq!(sgn(0.0), 0.0);
        assert_eq!(sgn(-2.0), -1.0);
        assert_eq!(sgn(1e-300), 1.0);
    }
}
