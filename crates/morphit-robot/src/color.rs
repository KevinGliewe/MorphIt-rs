//! Sphere colors: `#rrggbb` parsing and the per-link hue spread of the robot
//! rewriter. `rgb_to_hls` / `hls_to_rgb` are literal ports of CPython's
//! `colorsys`, so shades match the Python output digit for digit.

/// Soft blue used when no (valid) color is given.
pub const DEFAULT_SPHERE_RGBA: [f64; 4] = [0.2, 0.6, 1.0, 1.0];

/// Largest hue shift (fraction of the color wheel) at `color_variation = 1`.
pub const COLOR_VARIATION_MAX_HUE_SWING: f64 = 0.15;

/// Parse `#rrggbb` or `rrggbb` into RGBA in `0..=1` with alpha 1.
pub fn hex_to_rgba(hex: &str) -> Option<[f64; 4]> {
    let h = hex.trim_start_matches('#');
    if h.chars().count() != 6 || !h.is_ascii() {
        return None;
    }
    let ch = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok().map(|v| v as f64 / 255.0);
    Some([ch(0)?, ch(2)?, ch(4)?, 1.0])
}

/// `hex_to_rgba`, falling back to [`DEFAULT_SPHERE_RGBA`] for empty or
/// invalid input (the API never fails on a bad color).
pub fn safe_color_rgba(hex: Option<&str>) -> [f64; 4] {
    match hex {
        Some(h) if !h.is_empty() => hex_to_rgba(h).unwrap_or(DEFAULT_SPHERE_RGBA),
        _ => DEFAULT_SPHERE_RGBA,
    }
}

/// Python's `%` on floats: the result has the sign of the divisor.
fn py_mod(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r != 0.0 && (r < 0.0) != (b < 0.0) { r + b } else { r }
}

/// CPython `colorsys.rgb_to_hls`.
pub fn rgb_to_hls(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let maxc = r.max(g).max(b);
    let minc = r.min(g).min(b);
    let sumc = maxc + minc;
    let rangec = maxc - minc;
    let l = sumc / 2.0;
    if minc == maxc {
        return (0.0, l, 0.0);
    }
    let s = if l <= 0.5 { rangec / sumc } else { rangec / (2.0 - maxc - minc) };
    let rc = (maxc - r) / rangec;
    let gc = (maxc - g) / rangec;
    let bc = (maxc - b) / rangec;
    let h = if r == maxc {
        bc - gc
    } else if g == maxc {
        2.0 + rc - bc
    } else {
        4.0 + gc - rc
    };
    (py_mod(h / 6.0, 1.0), l, s)
}

/// CPython `colorsys.hls_to_rgb`.
pub fn hls_to_rgb(h: f64, l: f64, s: f64) -> (f64, f64, f64) {
    if s == 0.0 {
        return (l, l, l);
    }
    let m2 = if l <= 0.5 { l * (1.0 + s) } else { l + s - (l * s) };
    let m1 = 2.0 * l - m2;
    (v(m1, m2, h + 1.0 / 3.0), v(m1, m2, h), v(m1, m2, h - 1.0 / 3.0))
}

fn v(m1: f64, m2: f64, hue: f64) -> f64 {
    let hue = py_mod(hue, 1.0);
    if hue < 1.0 / 6.0 {
        return m1 + (m2 - m1) * hue * 6.0;
    }
    if hue < 0.5 {
        return m2;
    }
    if hue < 2.0 / 3.0 {
        return m1 + (m2 - m1) * (2.0 / 3.0 - hue) * 6.0;
    }
    m1
}

/// Shift the hue of `base` by `t * variation * 0.15` (`t` in `-1..=1` is the
/// link's position across the packed links). `variation <= 0` keeps `base`.
pub fn vary_color(base: [f64; 4], t: f64, variation: f64) -> [f64; 4] {
    if variation <= 0.0 {
        return base;
    }
    let (h, l, s) = rgb_to_hls(base[0], base[1], base[2]);
    let new_h = py_mod(h + t * variation * COLOR_VARIATION_MAX_HUE_SWING, 1.0);
    let (r, g, b) = hls_to_rgb(new_h, l, s);
    [r, g, b, base[3]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_parsing() {
        assert_eq!(hex_to_rgba("#3399ff"), Some([0.2, 0.6, 1.0, 1.0]));
        assert_eq!(hex_to_rgba("3399FF"), Some([0.2, 0.6, 1.0, 1.0]));
        assert_eq!(hex_to_rgba("#39f"), None);
        assert_eq!(hex_to_rgba("#zz99ff"), None);
        assert_eq!(safe_color_rgba(Some("nope")), DEFAULT_SPHERE_RGBA);
        assert_eq!(safe_color_rgba(Some("")), DEFAULT_SPHERE_RGBA);
        assert_eq!(safe_color_rgba(None), DEFAULT_SPHERE_RGBA);
        assert_eq!(safe_color_rgba(Some("#000000")), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn colorsys_values() {
        // python: colorsys.rgb_to_hls(0.2, 0.6, 1.0)
        let (h, l, s) = rgb_to_hls(0.2, 0.6, 1.0);
        assert!((h - 0.5833333333333334).abs() < 1e-15);
        assert!((l - 0.6).abs() < 1e-15 && (s - 1.0).abs() < 1e-15);
        // python: colorsys.hls_to_rgb(0.9, 0.3, 0.5)
        let (r, g, b) = hls_to_rgb(0.9, 0.3, 0.5);
        assert!((r - 0.45).abs() < 1e-12 && (g - 0.15).abs() < 1e-12 && (b - 0.33).abs() < 1e-12);
        for i in 0..=10 {
            for j in 0..=10 {
                for k in 0..=10 {
                    let c = (i as f64 / 10.0, j as f64 / 10.0, k as f64 / 10.0);
                    let (h, l, s) = rgb_to_hls(c.0, c.1, c.2);
                    let back = hls_to_rgb(h, l, s);
                    assert!((back.0 - c.0).abs() < 1e-12, "{c:?}");
                    assert!((back.1 - c.1).abs() < 1e-12, "{c:?}");
                    assert!((back.2 - c.2).abs() < 1e-12, "{c:?}");
                }
            }
        }
    }

    #[test]
    fn variation() {
        let base = DEFAULT_SPHERE_RGBA;
        assert_eq!(vary_color(base, 1.0, 0.0), base);
        let shifted = vary_color(base, -1.0, 1.0);
        let (h, _, _) = rgb_to_hls(shifted[0], shifted[1], shifted[2]);
        assert!((h - (0.5833333333333334 - 0.15)).abs() < 1e-12);
        assert_eq!(shifted[3], 1.0);
        // Wraps around the wheel like Python's modulo.
        let red = vary_color([1.0, 0.0, 0.0, 0.5], -1.0, 1.0);
        let (h, _, _) = rgb_to_hls(red[0], red[1], red[2]);
        assert!((h - 0.85).abs() < 1e-12);
    }
}
