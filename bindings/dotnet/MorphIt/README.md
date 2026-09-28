# MorphIt for .NET

Approximate any triangle mesh, or every collision mesh of a robot, with a fixed
budget of spheres: the [MorphIt](https://github.com/HIRO-group/MorphIt-1)
optimizer ([paper](https://arxiv.org/abs/2507.14061)) in Rust, for .NET. The
package contains the native library for Windows x64, Linux x64/arm64 and macOS
(Apple silicon and Intel); it targets .NET Standard 2.0 and .NET 8.

```csharp
using MorphIt;

using var mesh = Mesh.Load("bunny.obj");                  // OBJ, STL, PLY, DAE
using var config = new Config("MorphIt-B")
    .Set("model.num_spheres", 64)
    .Set("random_seed", 42);

using var session = new Session(mesh, config);
session.Run(step =>
{
    if (step.Iteration % 50 == 0) Console.WriteLine($"{step.Iteration}: {step.TotalLoss:F4}");
    return true;                                           // false stops the run
});

double[,] centers = session.Centers;                      // n x 3
double[] radii = session.Radii;
session.Save("bunny.json");                               // the Python MorphIt JSON schema

var urdf = ObjectModel.Urdf(centers, radii, new ObjectOptions { Name = "bunny" });
File.WriteAllText("bunny.urdf", urdf.Text);
Console.WriteLine($"coverage {session.Evaluate().RIn:P1}");
```

`await session.RunAsync(progress, cancellationToken)` runs on a worker thread;
cancelling the token stops the run. While a session runs, other threads can
read `State`, `Centers` and `Radii` or call `Cancel()`.

Robots:

```csharp
using var robot = RobotPackage.FromFolder("franka_panda");   // or FromZip(bytes)
using var report = robot.Inspect();
foreach (var item in report.PackItems)
{
    using var s = robot.PackLink(report, item.Index, new PackParams { NumSpheres = 20, Seed = 0 });
    s.Run();
    robot.SetLinkResult(report, item.Index, s);
}
File.WriteAllText("panda.spherical.urdf", robot.Assemble(report, "#3399ff", 0.6).Urdf);
```

Errors throw `MorphItException` with a `Status`. Documentation, the C/C++
library, Python, TypeScript, the CLI and MorphIt Studio:
<https://github.com/KevinGliewe/MorphIt-rs>. License: MIT.
