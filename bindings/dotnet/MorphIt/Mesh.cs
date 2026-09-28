using System;
using MorphIt.Native;

namespace MorphIt
{
    /// <summary>An immutable triangle mesh; may be shared by any number of sessions and threads.</summary>
    public sealed class Mesh : IDisposable
    {
        internal readonly MeshHandle Handle;

        internal Mesh(MeshHandle handle) => Handle = handle;

        /// <summary>Load an OBJ, STL, PLY or DAE file.</summary>
        public static Mesh Load(string path)
        {
            Api.Check(NativeMethods.morphit_mesh_load(Utf8.Encode(path)!, out var h));
            return new Mesh(h);
        }

        /// <summary>Load from file contents; <paramref name="ext"/> is "obj", "stl", "ply" or "dae".</summary>
        public static Mesh FromBytes(byte[] data, string ext, string? name = null)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            Api.Check(NativeMethods.morphit_mesh_from_bytes(data, (nuint)data.Length, Utf8.Encode(ext)!, Utf8.Encode(name), out var h));
            return new Mesh(h);
        }

        /// <summary>Build from vertices (n x 3) and counter-clockwise triangles (m x 3, zero-based).</summary>
        public static Mesh FromArrays(double[,] vertices, uint[,] triangles)
        {
            if (triangles.GetLength(1) != 3) throw new ArgumentException("triangles must have 3 columns", nameof(triangles));
            var xyz = Api.Flatten(vertices, nameof(vertices));
            var tri = new uint[triangles.Length];
            Buffer.BlockCopy(triangles, 0, tri, 0, tri.Length * sizeof(uint));
            Api.Check(NativeMethods.morphit_mesh_from_arrays(xyz, (nuint)vertices.GetLength(0), tri, (nuint)triangles.GetLength(0), out var h));
            return new Mesh(h);
        }

        public MeshInfo Info
        {
            get
            {
                Api.Check(NativeMethods.morphit_mesh_get_info(Handle, out var info));
                return new MeshInfo(info);
            }
        }

        /// <summary>Which of the points (n x 3) lie inside the mesh.</summary>
        public bool[] Contains(double[,] points)
        {
            var xyz = Api.Flatten(points, nameof(points));
            int n = points.GetLength(0);
            var inside = new byte[n];
            if (n > 0) Api.Check(NativeMethods.morphit_mesh_contains(Handle, xyz, (nuint)n, inside));
            return Array.ConvertAll(inside, b => b != 0);
        }

        /// <summary>The mesh as packing prepares it: the convex hull of each body (optional), then the union of overlapping bodies.</summary>
        public Mesh Prepared(bool unionOverlappingBodies = true, bool convexHull = false)
        {
            Api.Check(NativeMethods.morphit_mesh_prepare(Handle, unionOverlappingBodies ? 1 : 0, convexHull ? 1 : 0, out var h));
            return new Mesh(h);
        }

        /// <summary>What <see cref="Prepared"/> does, as JSON.</summary>
        public string PrepReportJson(bool unionOverlappingBodies = true, bool convexHull = false) =>
            Api.ReadString((byte[]? b, nuint c, out nuint n) =>
                NativeMethods.morphit_mesh_prep_report_json(Handle, unionOverlappingBodies ? 1 : 0, convexHull ? 1 : 0, b!, c, out n));

        /// <summary>Write .obj or .stl (chosen by the extension).</summary>
        public void Save(string path) => Api.Check(NativeMethods.morphit_mesh_save(Handle, Utf8.Encode(path)!));

        public void Dispose() => Handle.Dispose();
    }

    /// <summary>Optimizer configuration: a preset plus dotted-key overrides ("training.center_lr", ...).</summary>
    public sealed class Config : IDisposable
    {
        internal readonly ConfigHandle Handle;

        private Config(ConfigHandle handle) => Handle = handle;

        /// <summary>A preset: MorphIt-V, -S, -B (default), -Obj or -Obj-mass.</summary>
        public Config(string preset = "MorphIt-B")
        {
            Api.Check(NativeMethods.morphit_config_new(Utf8.Encode(preset)!, out Handle));
        }

        /// <summary>A config from nested JSON (as under "config" in a result file).</summary>
        public static Config FromJson(string json)
        {
            Api.Check(NativeMethods.morphit_config_from_json(Utf8.Encode(json)!, out var h));
            return new Config(h);
        }

        public Config Clone()
        {
            Api.Check(NativeMethods.morphit_config_clone(Handle, out var h));
            return new Config(h);
        }

        public Config Set(string key, double value)
        {
            Api.Check(NativeMethods.morphit_config_set_f64(Handle, Utf8.Encode(key)!, value));
            return this;
        }

        public Config Set(string key, long value)
        {
            Api.Check(NativeMethods.morphit_config_set_i64(Handle, Utf8.Encode(key)!, value));
            return this;
        }

        public Config Set(string key, int value) => Set(key, (long)value);

        public Config Set(string key, bool value)
        {
            Api.Check(NativeMethods.morphit_config_set_bool(Handle, Utf8.Encode(key)!, value ? 1 : 0));
            return this;
        }

        public Config Set(string key, string value)
        {
            Api.Check(NativeMethods.morphit_config_set_str(Handle, Utf8.Encode(key)!, Utf8.Encode(value)!));
            return this;
        }

        /// <summary>Apply {"dotted.key": value, ...} updates; all or nothing.</summary>
        public Config SetJson(string jsonUpdates)
        {
            Api.Check(NativeMethods.morphit_config_set_json(Handle, Utf8.Encode(jsonUpdates)!));
            return this;
        }

        public double GetDouble(string key)
        {
            Api.Check(NativeMethods.morphit_config_get_f64(Handle, Utf8.Encode(key)!, out var v));
            return v;
        }

        public long GetLong(string key)
        {
            Api.Check(NativeMethods.morphit_config_get_i64(Handle, Utf8.Encode(key)!, out var v));
            return v;
        }

        public string ToJson() =>
            Api.ReadString((byte[]? b, nuint c, out nuint n) => NativeMethods.morphit_config_to_json(Handle, b!, c, out n));

        public void Dispose() => Handle.Dispose();
    }

    /// <summary>Library-wide functions.</summary>
    public static class MorphItLibrary
    {
        private static LogFn? logCallback;

        /// <summary>Version of the native library, e.g. "0.1.0".</summary>
        public static string Version => Utf8.Decode(NativeMethods.morphit_version());

        /// <summary>Size the worker pool (default: all cores). Only before the first pack.</summary>
        public static void SetNumThreads(int n) => Api.Check(NativeMethods.morphit_set_num_threads((nuint)n));

        /// <summary>GPU adapter names; index i is the N of device "gpu:N".</summary>
        public static string[] Devices()
        {
            Api.Check(NativeMethods.morphit_device_count(out var count));
            var names = new string[(int)count];
            for (int i = 0; i < names.Length; i++)
            {
                nuint index = (nuint)i;
                names[i] = Api.ReadString((byte[]? b, nuint c, out nuint n) => NativeMethods.morphit_device_name(index, b!, c, out n));
            }
            return names;
        }

        /// <summary>
        /// Receive the library's log messages (level 1 = error ... 5 = trace) up to
        /// <paramref name="maxLevel"/>, from any thread. Process-wide; set it once.
        /// </summary>
        public static void SetLogHandler(Action<int, string> handler, int maxLevel = 3)
        {
            if (handler == null) throw new ArgumentNullException(nameof(handler));
            LogFn fn = (level, message, _) =>
            {
                try
                {
                    handler(level, Utf8.Decode(message));
                }
                catch
                {
                    // Never let an exception unwind into native code.
                }
            };
            Api.Check(NativeMethods.morphit_set_log_callback(fn, IntPtr.Zero, maxLevel));
            logCallback = fn; // keep the delegate alive
        }
    }
}
