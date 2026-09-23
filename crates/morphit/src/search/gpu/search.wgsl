// Candidate search kernels. Everything here is f32 and only *proposes*: the
// CPU recomputes the chosen pairs in f64 and rescans rows whose margin is
// below the error bound (see gpu/session.rs).

struct Params {
    n_samples: u32,
    n_spheres: u32,
    want_pairs: u32,
    pair_capacity: u32,
    tol: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct RowOut {
    idx: u32,
    margin: f32,
};

// Appended (sample, sphere) index pairs behind an atomic counter; one buffer
// so the whole kernel needs only four storage buffers (downlevel limit).
struct PairBuf {
    count: atomic<u32>,
    _pad: u32,
    data: array<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// xyz = position (shifted by the sample bounding-box centre), w unused.
@group(0) @binding(1) var<storage, read> samples: array<vec4<f32>>;
// xyz = centre (same shift), w = radius.
@group(0) @binding(2) var<storage, read> spheres: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<RowOut>;
@group(0) @binding(4) var<storage, read_write> pairs: PairBuf;

// f32::MAX, spelled exactly (a decimal literal above it is rejected by Tint).
const BIG: f32 = 0x1.fffffep+127f;

fn distance3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

// One thread per sample: the sphere minimizing |p - c_j| - r_j, and the
// margin to the runner-up. Optionally appends every sphere with
// r_j - |p - c_j| > -tol to the pair list (boundary loss candidates).
@compute @workgroup_size(256)
fn row_min(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_samples) {
        return;
    }
    let p = samples[i].xyz;
    let tol = params.tol;
    let want = params.want_pairs != 0u;
    var g1 = BIG;
    var g2 = BIG;
    var j1 = 0u;
    for (var j = 0u; j < params.n_spheres; j++) {
        let s = spheres[j];
        let d = distance3(p, s.xyz);
        let gap = d - s.w;
        if (gap < g1) {
            g2 = g1;
            g1 = gap;
            j1 = j;
        } else if (gap < g2) {
            g2 = gap;
        }
        if (want && (s.w - d > -tol)) {
            let k = atomicAdd(&pairs.count, 1u);
            if (k < params.pair_capacity) {
                pairs.data[2u * k] = i;
                pairs.data[2u * k + 1u] = j;
            }
        }
    }
    out[i] = RowOut(j1, g2 - g1);
}

var<workgroup> wd1: array<f32, 256>;
var<workgroup> wi1: array<u32, 256>;
var<workgroup> wd2: array<f32, 256>;

// One workgroup per sphere: the nearest sample and the margin to the
// second-nearest. Threads stride over the samples, then tree-reduce.
@compute @workgroup_size(256)
fn nearest_sample(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = wg.x;
    let t = lid.x;
    let c = spheres[j].xyz;
    var d1 = BIG;
    var d2 = BIG;
    var i1 = 0u;
    for (var i = t; i < params.n_samples; i += 256u) {
        let d = distance3(samples[i].xyz, c);
        if (d < d1) {
            d2 = d1;
            d1 = d;
            i1 = i;
        } else if (d < d2) {
            d2 = d;
        }
    }
    wd1[t] = d1;
    wi1[t] = i1;
    wd2[t] = d2;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            let a1 = wd1[t];
            let a2 = wd2[t];
            let b1 = wd1[t + s];
            let b2 = wd2[t + s];
            if (b1 < a1) {
                wi1[t] = wi1[t + s];
                wd1[t] = b1;
                wd2[t] = min(a1, b2);
            } else {
                wd2[t] = min(a2, b1);
            }
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        out[j] = RowOut(wi1[0], wd2[0] - wd1[0]);
    }
}
