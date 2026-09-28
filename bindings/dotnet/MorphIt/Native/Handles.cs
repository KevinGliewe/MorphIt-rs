using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace MorphIt.Native
{
    internal sealed class MeshHandle : SafeHandle
    {
        public MeshHandle() : base(IntPtr.Zero, true) { }
        public override bool IsInvalid => handle == IntPtr.Zero;
        protected override bool ReleaseHandle()
        {
            NativeMethods.morphit_mesh_free(handle);
            return true;
        }
    }

    internal sealed class ConfigHandle : SafeHandle
    {
        public ConfigHandle() : base(IntPtr.Zero, true) { }
        public override bool IsInvalid => handle == IntPtr.Zero;
        protected override bool ReleaseHandle()
        {
            NativeMethods.morphit_config_free(handle);
            return true;
        }
    }

    internal sealed class SessionHandle : SafeHandle
    {
        public SessionHandle() : base(IntPtr.Zero, true) { }
        public override bool IsInvalid => handle == IntPtr.Zero;
        protected override bool ReleaseHandle()
        {
            // A run on another thread keeps the session busy: stop it, then wait.
            NativeMethods.morphit_cancel(handle);
            while (NativeMethods.morphit_session_free(handle) == Status.Busy)
            {
                Thread.Yield();
            }
            return true;
        }
    }

    internal sealed class RobotHandle : SafeHandle
    {
        public RobotHandle() : base(IntPtr.Zero, true) { }
        public override bool IsInvalid => handle == IntPtr.Zero;
        protected override bool ReleaseHandle()
        {
            NativeMethods.morphit_robot_free(handle);
            return true;
        }
    }

    internal sealed class ReportHandle : SafeHandle
    {
        public ReportHandle() : base(IntPtr.Zero, true) { }
        public override bool IsInvalid => handle == IntPtr.Zero;
        protected override bool ReleaseHandle()
        {
            NativeMethods.morphit_report_free(handle);
            return true;
        }
    }

    internal static class Utf8
    {
        /// <summary>NUL-terminated UTF-8, or null for a null string.</summary>
        public static byte[]? Encode(string? s)
        {
            if (s == null) return null;
            var bytes = new byte[Encoding.UTF8.GetByteCount(s) + 1];
            Encoding.UTF8.GetBytes(s, 0, s.Length, bytes, 0);
            return bytes;
        }

        public static string Decode(IntPtr p)
        {
            if (p == IntPtr.Zero) return string.Empty;
            int n = 0;
            while (Marshal.ReadByte(p, n) != 0) n++;
            var bytes = new byte[n];
            Marshal.Copy(p, bytes, 0, n);
            return Encoding.UTF8.GetString(bytes);
        }

        /// <summary>Unmanaged NUL-terminated UTF-8 copy (free with Marshal.FreeHGlobal).</summary>
        public static IntPtr Alloc(string? s)
        {
            if (s == null) return IntPtr.Zero;
            var bytes = Encode(s)!;
            var p = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, p, bytes.Length);
            return p;
        }
    }

    internal delegate Status StringOut(byte[]? buf, nuint capacity, out nuint needed);

    internal static class Api
    {
        /// <summary>Throw <see cref="MorphItException"/> for an error status.</summary>
        public static Status Check(Status s)
        {
            if (s < 0) throw new MorphItException(s, Utf8.Decode(NativeMethods.morphit_last_error()));
            return s;
        }

        /// <summary>The C two-call pattern for string outputs.</summary>
        public static string ReadString(StringOut f)
        {
            Check(f(null, 0, out var needed));
            var buf = new byte[(int)needed];
            Check(f(buf, (nuint)buf.Length, out needed));
            int len = Array.IndexOf(buf, (byte)0);
            return Encoding.UTF8.GetString(buf, 0, len < 0 ? buf.Length : len);
        }

        public static double[] Flatten(double[,] points, string name)
        {
            if (points.GetLength(1) != 3) throw new ArgumentException($"{name} must have 3 columns (x, y, z)", name);
            var flat = new double[points.Length];
            Buffer.BlockCopy(points, 0, flat, 0, flat.Length * sizeof(double));
            return flat;
        }

        public static double[,] Unflatten(double[] flat, int count)
        {
            var points = new double[count, 3];
            Buffer.BlockCopy(flat, 0, points, 0, count * 3 * sizeof(double));
            return points;
        }
    }
}
