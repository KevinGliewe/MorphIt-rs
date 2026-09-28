// P/Invoke declarations of the MorphIt C API (crates/morphit-capi/include/morphit.h).
// Structs mirror the C layout (sequential, natural alignment); size_t is nuint.

using System;
using System.Runtime.InteropServices;

namespace MorphIt.Native
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate int ProgressFn(IntPtr info, IntPtr userData);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void LogFn(int level, IntPtr message, IntPtr userData);

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeStepInfo
    {
        public ulong Iteration;
        public double TotalLoss;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 12)] public double[] WeightedLosses;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 12)] public double[] RawLosses;
        public double PositionGradMag;
        public double RadiusGradMag;
        public nuint NumSpheres;
        public nuint Projected;
        public int DensityControlFired;
        public nuint DensityControlReplaced;
        public int Done;
        public int Converged;
        public double Seconds;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeStateInfo
    {
        public int State;
        public ulong Iteration;
        public ulong TotalIterations;
        public nuint NumSpheres;
        public double TotalLoss;
        public int Running;
        public ulong DensityControlPasses;
        public nuint Pruned;
        public ulong Seed;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeMeshInfo
    {
        public nuint NumVertices;
        public nuint NumFaces;
        public double Volume;
        public double Area;
        public double Scale;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)] public double[] BoundsMin;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)] public double[] BoundsMax;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)] public double[] CenterMass;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 9)] public double[] Inertia;
        public int WindingFlipped;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeObjectOptions
    {
        public IntPtr Name;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)] public double[] Rgba;
        public double TotalMass;
        public int Anchored;
        public int Decimals;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeQualityOptions
    {
        public ulong Seed;
        public nuint SurfaceSamples;
        public nuint VolumeSamples;
        public double BoundsExpand;
        public double Density;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeQualityMetrics
    {
        public nuint ActualN;
        public nuint NOut;
        public nuint NTiny;
        public double RIn;
        public double ROut;
        public double RUni;
        public double DAvgMm;
        public double DMaxMm;
        public double MassAbs;
        public double MassRel;
        public double ComAbs;
        public double ComRel;
        public double IAbs;
        public double IRel;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativePackParams
    {
        public IntPtr Variant;
        public nuint NumSpheres;
        public nuint Iterations;
        public long Seed;
        public int UnionOverlappingBodies;
        public int ConvexHull;
        public IntPtr AdvancedJson;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeAssembleStats
    {
        public nuint LinksWithCollisionsReplaced;
        public nuint MeshCollisionsReplaced;
        public nuint PrimitiveCollisionsRemoved;
        public nuint SphereCollisionsRemoved;
        public nuint SphereChildrenAdded;
    }

    /// <summary>
    /// Strings go in as NUL-terminated UTF-8 byte arrays (<see cref="Utf8.Encode"/>)
    /// and come out through (buffer, capacity, needed) byte buffers.
    /// </summary>
    internal static class NativeMethods
    {
        private const string Lib = "morphit_capi";
        private const CallingConvention Cc = CallingConvention.Cdecl;

        [DllImport(Lib, CallingConvention = Cc)] internal static extern IntPtr morphit_version();
        [DllImport(Lib, CallingConvention = Cc)] internal static extern IntPtr morphit_last_error();
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_set_num_threads(nuint n);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_set_log_callback(LogFn callback, IntPtr userData, int maxLevel);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_device_count(out nuint count);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_device_name(nuint index, byte[] buf, nuint capacity, out nuint needed);

        // Meshes
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_mesh_load(byte[] path, out MeshHandle mesh);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_mesh_from_arrays(double[] xyz, nuint numVertices, uint[] triangles, nuint numTriangles, out MeshHandle mesh);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_mesh_from_bytes(byte[] data, nuint len, byte[] ext, byte[]? name, out MeshHandle mesh);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_mesh_get_info(MeshHandle mesh, out NativeMeshInfo info);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_mesh_contains(MeshHandle mesh, double[] xyz, nuint numPoints, byte[] inside);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_mesh_prepare(MeshHandle mesh, int union, int convexHull, out MeshHandle prepared);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_mesh_prep_report_json(MeshHandle mesh, int union, int convexHull, byte[] buf, nuint capacity, out nuint needed);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_mesh_save(MeshHandle mesh, byte[] path);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern void morphit_mesh_free(IntPtr mesh);

        // Configs
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_new(byte[] preset, out ConfigHandle config);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_from_json(byte[] json, out ConfigHandle config);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_clone(ConfigHandle config, out ConfigHandle clone);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_set_f64(ConfigHandle config, byte[] key, double value);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_set_i64(ConfigHandle config, byte[] key, long value);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_set_bool(ConfigHandle config, byte[] key, int value);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_set_str(ConfigHandle config, byte[] key, byte[] value);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_set_json(ConfigHandle config, byte[] jsonUpdates);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_get_f64(ConfigHandle config, byte[] key, out double value);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_get_i64(ConfigHandle config, byte[] key, out long value);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_config_to_json(ConfigHandle config, byte[] buf, nuint capacity, out nuint needed);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern void morphit_config_free(IntPtr config);

        // Sessions
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_session_new(MeshHandle mesh, ConfigHandle config, out SessionHandle session);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_step(SessionHandle session, out NativeStepInfo info);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_run(SessionHandle session, ProgressFn progress, IntPtr userData);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_cancel(SessionHandle session);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_cancel(IntPtr session);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_finalize(SessionHandle session, out nuint pruned);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_session_free(IntPtr session);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_state(SessionHandle session, out NativeStateInfo state);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_get_centers(SessionHandle session, double[] buf, nuint capacity, out nuint written);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_get_radii(SessionHandle session, double[] buf, nuint capacity, out nuint written);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_get_masses(SessionHandle session, double[] buf, nuint capacity, out nuint written);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_get_last_step(SessionHandle session, out NativeStepInfo info);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_result_json(SessionHandle session, byte[] buf, nuint capacity, out nuint needed);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_result_save(SessionHandle session, byte[] path);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_session_config_json(SessionHandle session, byte[] buf, nuint capacity, out nuint needed);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_session_device(SessionHandle session, byte[] buf, nuint capacity, out nuint needed);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_session_mesh_prep_json(SessionHandle session, byte[] buf, nuint capacity, out nuint needed);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_history_json(SessionHandle session, byte[] buf, nuint capacity, out nuint needed);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_session_mesh(SessionHandle session, out MeshHandle mesh);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_session_evaluate(SessionHandle session, ref NativeQualityOptions options, out NativeQualityMetrics metrics);

        // Export and metrics
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_object_options_default(out NativeObjectOptions options);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_object_urdf(double[] centers, double[] radii, nuint count, ref NativeObjectOptions options, byte[] buf, nuint capacity, out nuint needed, double[] centroid);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_object_mjcf(double[] centers, double[] radii, nuint count, ref NativeObjectOptions options, byte[] buf, nuint capacity, out nuint needed, double[] centroid);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_quality_options_default(out NativeQualityOptions options);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_evaluate_packing(MeshHandle mesh, double[] centers, double[] radii, double[]? masses, nuint count, ref NativeQualityOptions options, out NativeQualityMetrics metrics);

        // Robots
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_robot_new(out RobotHandle robot);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_robot_from_folder(byte[] path, out RobotHandle robot);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_robot_from_zip(byte[] data, nuint len, out RobotHandle robot);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_robot_add_file(RobotHandle robot, byte[] path, byte[] data, nuint len);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern void morphit_robot_free(IntPtr robot);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_robot_inspect(RobotHandle robot, byte[]? urdf, out ReportHandle report);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_report_json(ReportHandle report, byte[] buf, nuint capacity, out nuint needed);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_report_pack_count(ReportHandle report, out nuint count);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_report_pack_item(ReportHandle report, nuint index, byte[] linkBuf, nuint capacity, out nuint needed, out nuint collisionIndex);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern void morphit_report_free(IntPtr report);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_pack_params_default(out NativePackParams parameters);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_robot_pack_link(RobotHandle robot, ReportHandle report, nuint index, ref NativePackParams parameters, byte[]? device, out SessionHandle session);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_robot_set_link_result(RobotHandle robot, ReportHandle report, nuint index, SessionHandle session);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_robot_set_link_result_json(RobotHandle robot, byte[] link, nuint collisionIndex, byte[] resultJson);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_robot_clear_link_results(RobotHandle robot);
        [DllImport(Lib, CallingConvention = Cc)] internal static extern Status morphit_robot_assemble(RobotHandle robot, ReportHandle report, byte[]? baseColor, double colorVariation, byte[] buf, nuint capacity, out nuint needed, out NativeAssembleStats stats);
    }
}
