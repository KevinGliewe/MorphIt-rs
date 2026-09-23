// Minimal MorphIt page: pick a mesh, pack it in the browser (WebGPU when
// available), watch the spheres (top view) and the loss, download results.
import init, { Config, Mesh, Session, initGpu, initLogging, objectUrdf, objectMjcf } from "./pkg/morphit_wasm.js";

const $ = (id) => document.getElementById(id);
let mesh = null;
let meshName = "object";
let session = null;
let losses = [];

await init();
initLogging("warn");
for (const p of Config.presets()) $("preset").add(new Option(p, p, p === "MorphIt-B", p === "MorphIt-B"));
for (const g of await initGpu()) $("device").add(new Option(`gpu:${g.index} (${g.name})`, `gpu:${g.index}`));
$("device").add(new Option("auto", "auto"));
$("status").textContent = "Pick a mesh.";

$("file").onchange = async () => {
  const f = $("file").files[0];
  if (!f) return;
  try {
    const ext = f.name.split(".").pop();
    mesh = Mesh.fromBytes(new Uint8Array(await f.arrayBuffer()), ext, f.name);
    meshName = f.name.replace(/\.[^.]*$/, "");
    const i = mesh.info();
    $("status").textContent = `${f.name}: ${i.faces} faces, volume ${i.volume.toPrecision(4)}`;
    $("run").disabled = false;
    session = null;
    draw();
  } catch (e) {
    $("status").textContent = String(e);
  }
};

$("run").onclick = async () => {
  const c = Config.fromPreset($("preset").value);
  c.set("model.num_spheres", Number($("spheres").value));
  c.set("training.iterations", Number($("iters").value));
  c.set("model.device", $("device").value);
  $("run").disabled = true;
  try {
    session = new Session(c, mesh);
  } catch (e) {
    $("status").textContent = String(e);
    $("run").disabled = false;
    return;
  }
  losses = [];
  const t0 = performance.now();
  let drawn = 0;
  while (!session.isDone) {
    // ~12 ms of optimizer work, then yield so the page stays responsive.
    // (Yielding with a task, not requestAnimationFrame: a hidden or occluded
    // window throttles animation frames to about one per second.)
    const info = await session.stepMany({ budgetMs: 12 });
    if (info) losses.push(info.totalLoss);
    if (performance.now() - drawn > 30) {
      $("status").textContent = `${session.device}: iteration ${session.iteration}/${session.totalIterations}` +
        (info ? `, loss ${info.totalLoss.toPrecision(5)}` : "");
      draw();
      drawn = performance.now();
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  const pruned = session.finalize();
  draw();
  $("status").textContent = `${session.device}: done in ${((performance.now() - t0) / 1000).toFixed(1)} s, ` +
    `${session.iteration} iterations, ${session.radii().length} spheres (${pruned} pruned)`;
  for (const id of ["run", "json", "urdf", "mjcf"]) $(id).disabled = false;
};

// A file picked while the module was still loading.
if ($("file").files.length) $("file").onchange();

function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/octet-stream" }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
$("json").onclick = () => download(`${meshName}.spheres.json`, session.resultJson());
$("urdf").onclick = () => download(`${meshName}.urdf`, objectUrdf(session.centers(), session.radii(), { name: meshName }).text);
$("mjcf").onclick = () => download(`${meshName}.xml`, objectMjcf(session.centers(), session.radii(), { name: meshName }).text);

function draw() {
  const cv = $("view"), g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  if (!mesh) return;
  const { bounds } = mesh.info();
  const [lo, hi] = bounds;
  const s = 0.9 * Math.min(cv.width / (hi[0] - lo[0]), cv.height / (hi[1] - lo[1]));
  const px = (x) => cv.width / 2 + (x - (lo[0] + hi[0]) / 2) * s;
  const py = (y) => cv.height / 2 - (y - (lo[1] + hi[1]) / 2) * s;
  // Mesh outline: every triangle, faintly.
  const v = mesh.vertices(), f = mesh.faces();
  g.strokeStyle = "rgba(128,128,128,0.15)";
  g.beginPath();
  for (let t = 0; t < f.length; t += 3) {
    const [a, b, c] = [f[t] * 3, f[t + 1] * 3, f[t + 2] * 3];
    g.moveTo(px(v[a]), py(v[a + 1])); g.lineTo(px(v[b]), py(v[b + 1])); g.lineTo(px(v[c]), py(v[c + 1]));
  }
  g.stroke();
  if (session) {
    const c = session.centers(), r = session.radii();
    g.fillStyle = "rgba(51,153,255,0.35)"; g.strokeStyle = "rgba(51,153,255,0.9)";
    for (let i = 0; i < r.length; i++) {
      g.beginPath(); g.arc(px(c[3 * i]), py(c[3 * i + 1]), r[i] * s, 0, 2 * Math.PI); g.fill(); g.stroke();
    }
  }
  const lc = $("loss"), l = lc.getContext("2d");
  l.clearRect(0, 0, lc.width, lc.height);
  if (losses.length > 1) {
    const logs = losses.map((x) => Math.log10(Math.max(x, 1e-12)));
    const mn = Math.min(...logs), mx = Math.max(...logs);
    l.strokeStyle = "#3399ff"; l.beginPath();
    logs.forEach((y, i) => {
      const X = (i / (logs.length - 1)) * lc.width, Y = lc.height - 8 - ((y - mn) / (mx - mn || 1)) * (lc.height - 16);
      i ? l.lineTo(X, Y) : l.moveTo(X, Y);
    });
    l.stroke();
  }
}
