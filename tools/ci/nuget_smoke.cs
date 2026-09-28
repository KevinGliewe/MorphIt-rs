// Program of the release workflow's NuGet smoke test: install the freshly
// packed MorphIt package into a new console app and pack a mesh.
using MorphIt;

using var mesh = Mesh.Load(args[0]);
using var config = new Config().Set("model.num_spheres", 8).Set("training.iterations", 20).Set("random_seed", 0);
using var session = new Session(mesh, config);
session.Run();
System.Console.WriteLine($"MorphIt {MorphItLibrary.Version}: {session.Radii.Length} spheres");
