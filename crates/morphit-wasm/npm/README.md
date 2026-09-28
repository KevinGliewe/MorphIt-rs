# morphit-rs

Approximate any triangle mesh, or every collision mesh of a robot, with a fixed
budget of spheres: the [MorphIt](https://github.com/HIRO-group/MorphIt-1)
optimizer ([paper](https://arxiv.org/abs/2507.14061)) in Rust, compiled to
WebAssembly, with TypeScript types. It runs in browsers (with WebGPU when
available), bundlers and Node.js.

```sh
npm install morphit-rs
```

```ts
import init, { Config, Mesh, Session, initGpu, objectUrdf } from "morphit-rs";

await init();                              // loads the WebAssembly module (a no-op in Node)
await initGpu();                           // optional: WebGPU for the searches (browsers)

const mesh = Mesh.fromBytes(bytes, "obj", "bunny.obj");   // Uint8Array of an OBJ/STL/PLY/DAE file
const config = Config.fromPreset("MorphIt-B");
config.set("model.num_spheres", 64);
config.set("random_seed", 42);

const session = new Session(config, mesh);
while (!session.isDone) {
  await session.stepMany({ budgetMs: 12 });   // then yield, so a page stays responsive
  draw(session.centers(), session.radii());   // Float64Array (x y z per sphere), radii
}
session.finalize();
const result = JSON.parse(session.resultJson());          // the Python MorphIt JSON schema
const urdf = objectUrdf(session.centers(), session.radii(), { name: "bunny" }).text;
```

In Node.js, read the mesh with `fs.readFileSync(path)`; everything runs on the
calling thread on the CPU (WebGPU is browser-only).

The classes implement `Symbol.dispose` (`using mesh = Mesh.fromBytes(...)` frees
the memory at the end of the scope; otherwise call `free()` or let the garbage
collector do it). For TypeScript this needs 5.2+ with `esnext.disposable` in
`lib`, or `skipLibCheck`.

Robots:

```ts
import { RobotPackage } from "morphit-rs";

const pkg = RobotPackage.fromZip(zipBytes);        // or fromFiles({ "robot.urdf": bytes, ... })
const report = pkg.inspect();
for (const item of report.collisions.filter((c) => c.action === "pack")) {
  const s = pkg.packLink(item, { numSpheres: 20, iterations: 200, seed: 0 });
  while (!s.isDone) await s.stepMany({ budgetMs: 50 });
  s.finalize();
  pkg.setLinkResult(item.link_name, item.collision_index, s.resultJson());
}
const { urdf } = pkg.assemble(undefined, "#3399ff", 0.6);
```

Also: `Mesh.prepared` (merge overlapping bodies, convex hulls), `toObj`/`toStl`,
`objectMjcf`, `evaluatePacking` and more; see the bundled type declarations.

A seed makes a run reproducible, identical on the CPU and WebGPU; results may
differ in the last bits from native builds (WebAssembly has its own `exp`/`ln`).

Documentation, the Rust crate, Python, C/C++, C# and MorphIt Studio:
<https://github.com/KevinGliewe/MorphIt-rs>. License: MIT.
