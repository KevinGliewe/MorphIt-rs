using System;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using MorphIt.Native;

namespace MorphIt
{
    /// <summary>
    /// A packing run. The read members (<see cref="State"/>, <see cref="Centers"/>,
    /// <see cref="Radii"/>, ...) and <see cref="Cancel"/> may be used from any thread
    /// while <see cref="Run"/> is active on another; a second Run, Step or
    /// <see cref="FinalizeSpheres"/> meanwhile throws with <see cref="Status.Busy"/>.
    /// </summary>
    public sealed class Session : IDisposable
    {
        internal readonly SessionHandle Handle;

        internal Session(SessionHandle handle) => Handle = handle;

        /// <summary>Prepare the mesh, draw the samples and place the initial spheres.</summary>
        public Session(Mesh mesh, Config config)
        {
            if (mesh == null) throw new ArgumentNullException(nameof(mesh));
            if (config == null) throw new ArgumentNullException(nameof(config));
            Api.Check(NativeMethods.morphit_session_new(mesh.Handle, config.Handle, out Handle));
        }

        /// <summary>Run one iteration; null when no iterations are left.</summary>
        public StepInfo? Step()
        {
            var s = Api.Check(NativeMethods.morphit_step(Handle, out var info));
            return s == Status.Done ? null : new StepInfo(info);
        }

        /// <summary>
        /// Step until done, then remove escaped spheres. <paramref name="onStep"/>
        /// returning false stops the run (it can be continued later); an exception it
        /// throws stops the run and is rethrown. <see cref="Cancel"/> stops it too.
        /// </summary>
        public RunOutcome Run(Func<StepInfo, bool>? onStep = null)
        {
            ExceptionDispatchInfo? error = null;
            bool converged = false;
            ProgressFn progress = (info, _) =>
            {
                try
                {
                    var step = new StepInfo(Marshal.PtrToStructure<NativeStepInfo>(info));
                    converged = step.Converged;
                    return onStep == null || onStep(step) ? 0 : 1;
                }
                catch (Exception e)
                {
                    error = ExceptionDispatchInfo.Capture(e);
                    return 1;
                }
            };
            var status = NativeMethods.morphit_run(Handle, progress, IntPtr.Zero);
            GC.KeepAlive(progress);
            error?.Throw();
            if (status == Status.Cancelled) return RunOutcome.Cancelled;
            Api.Check(status);
            return converged ? RunOutcome.Converged : RunOutcome.Completed;
        }

        /// <summary>
        /// <see cref="Run"/> on a worker thread. <paramref name="progress"/> receives every
        /// step; cancelling <paramref name="cancellationToken"/> stops the run and throws
        /// <see cref="OperationCanceledException"/>.
        /// </summary>
        public Task<RunOutcome> RunAsync(IProgress<StepInfo>? progress = null, CancellationToken cancellationToken = default)
        {
            return Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                RunOutcome outcome;
                using (cancellationToken.Register(Cancel))
                {
                    outcome = Run(step =>
                    {
                        progress?.Report(step);
                        return true;
                    });
                }
                if (outcome == RunOutcome.Cancelled) cancellationToken.ThrowIfCancellationRequested();
                return outcome;
            }, cancellationToken);
        }

        /// <summary>Stop a running <see cref="Run"/> after the current iteration (any thread, never blocks).</summary>
        public void Cancel() => Api.Check(NativeMethods.morphit_cancel(Handle));

        /// <summary>Remove spheres whose centers ended outside the mesh and end the session; returns how many.</summary>
        public int FinalizeSpheres()
        {
            Api.Check(NativeMethods.morphit_finalize(Handle, out var pruned));
            return (int)pruned;
        }

        public StateInfo State
        {
            get
            {
                Api.Check(NativeMethods.morphit_state(Handle, out var st));
                return new StateInfo(st);
            }
        }

        /// <summary>Sphere centers, n x 3.</summary>
        public double[,] Centers
        {
            get
            {
                var flat = Read(NativeMethods.morphit_get_centers);
                return Api.Unflatten(flat, flat.Length / 3);
            }
        }

        public double[] Radii => Read(NativeMethods.morphit_get_radii);
        public double[] Masses => Read(NativeMethods.morphit_get_masses);

        /// <summary>The last iteration's info, or null before the first.</summary>
        public StepInfo? LastStep
        {
            get
            {
                var s = NativeMethods.morphit_get_last_step(Handle, out var info);
                if (s == Status.State) return null;
                Api.Check(s);
                return new StepInfo(info);
            }
        }

        /// <summary>The current spheres in the Python MorphIt JSON schema.</summary>
        public string ResultJson => Str(NativeMethods.morphit_result_json);
        public string ConfigJson => Str(NativeMethods.morphit_session_config_json);
        /// <summary>The compute device, e.g. "cpu" or "gpu:0 (...)".</summary>
        public string Device => Str(NativeMethods.morphit_session_device);
        public string MeshPrepJson => Str(NativeMethods.morphit_session_mesh_prep_json);
        /// <summary>The per-iteration history (waits for a running iteration).</summary>
        public string HistoryJson => Str(NativeMethods.morphit_history_json);

        /// <summary>Write the result JSON.</summary>
        public void Save(string path) => Api.Check(NativeMethods.morphit_result_save(Handle, Utf8.Encode(path)!));

        /// <summary>The mesh as packed (after mesh preparation).</summary>
        public Mesh GetMesh()
        {
            Api.Check(NativeMethods.morphit_session_mesh(Handle, out var h));
            return new Mesh(h);
        }

        /// <summary>Quality of the current spheres against the prepared mesh.</summary>
        public QualityMetrics Evaluate(QualityOptions? options = null)
        {
            var o = (options ?? new QualityOptions()).ToNative();
            Api.Check(NativeMethods.morphit_session_evaluate(Handle, ref o, out var q));
            return new QualityMetrics(q);
        }

        public void Dispose() => Handle.Dispose();

        private delegate Status ArrayOut(SessionHandle s, double[] buf, nuint capacity, out nuint written);
        private delegate Status SessionStringOut(SessionHandle s, byte[] buf, nuint capacity, out nuint needed);

        private double[] Read(ArrayOut f)
        {
            Api.Check(f(Handle, null!, 0, out var n));
            var buf = new double[(int)n];
            Api.Check(f(Handle, buf, (nuint)buf.Length, out n));
            if ((int)n != buf.Length) Array.Resize(ref buf, (int)n);
            return buf;
        }

        private string Str(SessionStringOut f) =>
            Api.ReadString((byte[]? b, nuint c, out nuint n) => f(Handle, b!, c, out n));
    }
}
