using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace MorphIt.Tests
{
    public class MorphItTests
    {
        private static readonly string Repo = FindRepo();
        private static string Link0 => Path.Combine(Repo, "crates", "morphit", "tests", "fixtures", "link0.obj");
        private static string Kinova => Path.Combine(Repo, "web", "examples", "kinova_description");

        private static string FindRepo()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null && !File.Exists(Path.Combine(dir.FullName, "Cargo.lock"))) dir = dir.Parent;
            return dir?.FullName ?? throw new InvalidOperationException("repository root not found");
        }

        private static Config SmallConfig(int spheres, int iterations) =>
            new Config("MorphIt-B")
                .Set("model.num_spheres", spheres)
                .Set("training.iterations", iterations)
                .Set("random_seed", 7)
                .Set("model.num_inside_samples", 400)
                .Set("model.num_surface_samples", 400)
                .Set("model.device", "cpu");

        private static Mesh UnitBox() => Mesh.FromArrays(
            new double[,] { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 } },
            new uint[,] { { 0, 2, 1 }, { 0, 3, 2 }, { 4, 5, 6 }, { 4, 6, 7 }, { 0, 1, 5 }, { 0, 5, 4 }, { 2, 3, 7 }, { 2, 7, 6 }, { 0, 4, 7 }, { 0, 7, 3 }, { 1, 2, 6 }, { 1, 6, 5 } });

        [Fact]
        public void VersionAndDevices()
        {
            Assert.Matches(@"^\d+\.\d+\.\d+", MorphItLibrary.Version);
            Assert.NotNull(MorphItLibrary.Devices());
        }

        [Fact]
        public void ErrorsCarryStatusAndMessage()
        {
            var e = Assert.Throws<MorphItException>(() => Mesh.Load("definitely-missing.obj"));
            Assert.Equal(Status.Io, e.Status);
            e = Assert.Throws<MorphItException>(() => new Config("MorphIt-X"));
            Assert.Equal(Status.Config, e.Status);
            using var c = new Config();
            e = Assert.Throws<MorphItException>(() => c.Set("training.nope", 1.0));
            Assert.Contains("nope", e.Message);
        }

        [Fact]
        public void MeshAndConfig()
        {
            using var box = UnitBox();
            Assert.Equal(1.0, box.Info.Volume, 12);
            Assert.Equal(new[] { true, false }, box.Contains(new double[,] { { 0.3, 0.4, 0.45 }, { 2, 0.4, 0.45 } }));
            using var fromBytes = Mesh.FromBytes(File.ReadAllBytes(Link0), "obj", "link0.obj");
            Assert.Equal(200, fromBytes.Info.NumFaces);

            using var c = SmallConfig(5, 10);
            using var d = c.Clone().Set("model.num_spheres", 9);
            Assert.Equal(5, c.GetLong("model.num_spheres"));
            Assert.Equal(9, d.GetLong("model.num_spheres"));
            Assert.Contains("\"num_spheres\": 5", c.ToJson());
            c.SetJson("{\"training.center_lr\": 0.001}");
            Assert.Equal(0.001, c.GetDouble("training.center_lr"));
        }

        [Fact]
        public void RunEqualsStepping()
        {
            using var mesh = Mesh.Load(Link0);
            using var config = SmallConfig(6, 40);
            using var a = new Session(mesh, config);
            int calls = 0;
            Assert.NotEqual(RunOutcome.Cancelled, a.Run(_ => ++calls > 0));
            Assert.Equal(40, calls);
            Assert.Equal(SessionState.Finalized, a.State.State);

            using var b = new Session(mesh, config);
            int steps = 0;
            while (b.Step() != null) steps++;
            Assert.Equal(40, steps);
            b.FinalizeSpheres();
            Assert.Equal(a.Radii, b.Radii);
            Assert.Equal(a.Centers, b.Centers);
            Assert.Equal(a.ResultJson, b.ResultJson);
            Assert.Equal(6, a.Centers.GetLength(0));
            Assert.Equal(12, a.LastStep!.WeightedLosses.Count);
            Assert.Equal("cpu", a.Device);
        }

        [Fact]
        public void CallbackStopsAndExceptionsPropagate()
        {
            using var mesh = Mesh.Load(Link0);
            using var config = SmallConfig(6, 40);
            using var s = new Session(mesh, config);
            Assert.Equal(RunOutcome.Cancelled, s.Run(step => step.Iteration < 3));
            Assert.Equal(4, s.State.Iteration);
            Assert.NotEqual(RunOutcome.Cancelled, s.Run());

            using var t = new Session(mesh, config);
            var e = Assert.Throws<InvalidOperationException>(() => t.Run(_ => throw new InvalidOperationException("boom")));
            Assert.Equal("boom", e.Message);
        }

        [Fact]
        public async Task CancelFromAnotherThreadAndBusy()
        {
            using var mesh = Mesh.Load(Link0);
            using var config = SmallConfig(6, 100000);
            using var s = new Session(mesh, config);
            using var started = new ManualResetEventSlim();
            var run = Task.Run(() => s.Run(_ =>
            {
                started.Set();
                return true;
            }));
            started.Wait();
            Assert.True(s.State.Running);
            Assert.Equal(6, s.Radii.Length);
            Assert.Equal(Status.Busy, Assert.Throws<MorphItException>(() => s.Step()).Status);
            s.Cancel();
            Assert.Equal(RunOutcome.Cancelled, await run);

            using var cts = new CancellationTokenSource();
            var progress = new SynchronousProgress(() => cts.Cancel());
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => s.RunAsync(progress, cts.Token));
        }

        private sealed class SynchronousProgress : IProgress<StepInfo>
        {
            private readonly Action onReport;
            public SynchronousProgress(Action onReport) => this.onReport = onReport;
            public void Report(StepInfo value) => onReport();
        }

        [Fact]
        public void ObjectModelsMatchPython()
        {
            var fixture = Path.Combine(Repo, "crates", "morphit-robot", "tests", "fixtures", "py_object_model.json");
            using var doc = JsonDocument.Parse(File.ReadAllText(fixture));
            int checkedCases = 0;
            foreach (var c in doc.RootElement.EnumerateObject())
            {
                var centerRows = c.Value.GetProperty("centers").EnumerateArray().ToArray();
                var radii = c.Value.GetProperty("radii").EnumerateArray().Select(v => v.GetDouble()).ToArray();
                if (centerRows.Length != radii.Length) continue; // Python zips unequal lists; one count here
                var centers = new double[radii.Length, 3];
                for (int i = 0; i < radii.Length; i++)
                    for (int k = 0; k < 3; k++)
                        centers[i, k] = centerRows[i][k].GetDouble();
                var options = new ObjectOptions
                {
                    Name = c.Value.GetProperty("robot_name").GetString()!,
                    Rgba = c.Value.GetProperty("rgba").EnumerateArray().Select(v => v.GetDouble()).ToArray(),
                };
                foreach (var (anchored, key) in new[] { (false, ""), (true, "_anchored") })
                {
                    options.Anchored = anchored;
                    var urdf = ObjectModel.Urdf(centers, radii, options);
                    Assert.Equal(c.Value.GetProperty("urdf" + key).GetString(), urdf.Text);
                    Assert.Equal(c.Value.GetProperty("centroid").EnumerateArray().Select(v => v.GetDouble()), urdf.Centroid);
                    Assert.Equal(c.Value.GetProperty("mjcf" + key).GetString(), ObjectModel.Mjcf(centers, radii, options).Text);
                }
                checkedCases++;
            }
            Assert.True(checkedCases >= 1);
        }

        [Fact]
        public void QualityMetrics()
        {
            using var mesh = Mesh.Load(Link0);
            using var config = SmallConfig(8, 30);
            using var s = new Session(mesh, config);
            s.Run();
            var options = new QualityOptions { SurfaceSamples = 2000, VolumeSamples = 2000 };
            var q = s.Evaluate(options);
            Assert.Equal(s.Radii.Length, q.ActualN);
            Assert.InRange(q.RIn, 0.01, 1.0);
            using var prepared = s.GetMesh();
            var q2 = Quality.Evaluate(prepared, s.Centers, s.Radii, null, options);
            Assert.Equal(q.RIn, q2.RIn);
        }

        [Fact]
        public void RobotPipeline()
        {
            using var robot = RobotPackage.FromFolder(Kinova);
            using var report = robot.Inspect();
            var items = report.PackItems;
            Assert.True(items.Count >= 2);
            Assert.Contains("\"collisions\"", report.Json);
            var p = new PackParams { NumSpheres = 4, Iterations = 10, Seed = 0 };
            foreach (var item in items.Take(2))
            {
                using var s = robot.PackLink(report, item.Index, p, "cpu");
                s.Run();
                robot.SetLinkResult(report, item.Index, s);
            }
            var assembled = robot.Assemble(report, "#3399ff", 0.5);
            Assert.Contains("<sphere", assembled.Urdf);
            Assert.Equal(2, assembled.MeshCollisionsReplaced);

            var bad = Assert.Throws<MorphItException>(() => robot.PackLink(report, 0, new PackParams { NumSpheres = 0 }));
            Assert.Equal(Status.InvalidArg, bad.Status);
            Assert.Equal(Status.Io, Assert.Throws<MorphItException>(() => RobotPackage.FromFolder("/definitely/not/here")).Status);
        }

        [Fact]
        public void RobotFromZipMatchesFolder()
        {
            var zipPath = Path.Combine(Path.GetTempPath(), $"morphit-kinova-{Guid.NewGuid():N}.zip");
            try
            {
                ZipFile.CreateFromDirectory(Kinova, zipPath);
                using var fromZip = RobotPackage.FromZip(File.ReadAllBytes(zipPath));
                using var fromFolder = RobotPackage.FromFolder(Kinova);
                using var a = fromZip.Inspect();
                using var b = fromFolder.Inspect();
                Assert.Equal(b.PackItems.Select(i => i.ToString()), a.PackItems.Select(i => i.ToString()));
            }
            finally
            {
                File.Delete(zipPath);
            }
        }
    }
}
