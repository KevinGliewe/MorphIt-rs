# morphit-rs

Approximate any triangle mesh, or every collision mesh of a robot, with a fixed
budget of spheres. This is the [MorphIt](https://github.com/HIRO-group/MorphIt-1)
optimizer ([paper](https://arxiv.org/abs/2507.14061)) ported to Rust, for
Python: 15–30× faster than the original on the CPU, GPU searches on any GPU
(Vulkan, Metal, DirectX 12), and results that are bit-identical across thread
counts and devices.

```sh
pip install morphit-rs
```

Wheels are built for Windows x64, Linux x64/arm64 (glibc 2.28+) and macOS
(Apple silicon and Intel), for Python 3.9 and newer. NumPy is the only
dependency.

## Pack a mesh

```python
import morphit_rs as mi

mesh = mi.Mesh.load("bunny.obj")                 # OBJ, STL, PLY, DAE; or Mesh.from_arrays(v, f)
config = mi.Config.preset("MorphIt-B", num_spheres=64, seed=42)
config["training.center_lr"] = 0.001             # any dotted key of the original config

result = mi.pack(mesh, config)                   # the GIL is released while it runs
result.centers                                   # (64, 3) float64
result.radii                                     # (64,)
result.save("bunny.json")                        # the original's JSON schema

urdf, centroid = mi.object_urdf(result.centers, result.radii, name="bunny")
mjcf, _ = mi.object_mjcf(result.centers, result.radii, anchored=True)
print(mi.evaluate_packing(mesh, result))         # coverage, surface distance, mass errors
```

## Step by step

```python
session = mi.Session(mesh, config)
for step in session:                             # Ctrl+C stops between steps
    print(step.iteration, step.total_loss, step.weighted_losses["coverage_loss"])
session.finalize()
result = session.result()
```

`session.run(callback, every=10)` does the same with a callback (return
`False` to stop). Another thread can read `session.centers`, `session.radii`
and `session.iteration` while it runs, or call `session.cancel()`.

## Robots

```python
pkg = mi.RobotPackage.from_folder("franka_panda")    # or from_zip(...), from_files({...})
report = pkg.inspect()                               # which collisions to pack or remove
results = pkg.pack_all(num_spheres=20, iterations=200, seed=0)
urdf, stats = pkg.assemble(base_color="#3399ff", color_variation=0.6)
open("panda.spherical.urdf", "w").write(urdf)
```

## Coming from the Python MorphIt

`morphit_rs.compat` provides the original API. Change the imports and keep the
script:

```python
from morphit_rs.compat import get_config, update_config_from_dict, MorphIt, train_morphit

config = get_config("MorphIt-B")
config = update_config_from_dict(config, {"model.num_spheres": 20, "model.mesh_path": "link0.obj"})
model = MorphIt(config)
tracker = train_morphit(model, iteration_callback=lambda i, model, info: print(i, info["total_loss"]))
model.save_results()
```

`model.centers` and `model.radii` are torch tensors when torch is installed,
NumPy arrays otherwise. PyVista visualization is not included.

## More

- Documentation, the C library, CLI, HTTP server and WebAssembly build:
  <https://github.com/KevinGliewe/MorphIt-rs#readme>
- MorphIt Studio in the browser: <https://kevingliewe.github.io/MorphIt-rs/>
- License: MIT
