using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MorphIt.Native;

namespace MorphIt
{
    /// <summary>Options of <see cref="ObjectModel.Urdf"/> and <see cref="ObjectModel.Mjcf"/>.</summary>
    public sealed class ObjectOptions
    {
        public ObjectOptions()
        {
            Api.Check(NativeMethods.morphit_object_options_default(out var d));
            Rgba = d.Rgba;
            TotalMass = d.TotalMass;
            Anchored = d.Anchored != 0;
            Decimals = d.Decimals;
        }

        /// <summary>&lt;robot name&gt; / &lt;mujoco model&gt;.</summary>
        public string Name { get; set; } = "object";
        /// <summary>Sphere color, RGBA in 0..1.</summary>
        public double[] Rgba { get; set; }
        /// <summary>Total mass in kg, split over the spheres in proportion to r^3.</summary>
        public double TotalMass { get; set; }
        /// <summary>Weld the object to the world instead of letting it float.</summary>
        public bool Anchored { get; set; }
        /// <summary>Decimal places of every number written.</summary>
        public int Decimals { get; set; }
    }

    /// <summary>A generated model and the point its sphere positions are relative to.</summary>
    public sealed class ObjectModelText
    {
        internal ObjectModelText(string text, double[] centroid)
        {
            Text = text;
            Centroid = centroid;
        }

        public string Text { get; }
        public IReadOnlyList<double> Centroid { get; }
    }

    /// <summary>Simulator models from spheres.</summary>
    public static class ObjectModel
    {
        /// <summary>The spheres as a URDF: one link per sphere on fixed joints, masses split by volume.</summary>
        public static ObjectModelText Urdf(double[,] centers, double[] radii, ObjectOptions? options = null) =>
            Write(NativeMethods.morphit_object_urdf, centers, radii, options);

        /// <summary>The spheres as MJCF for MuJoCo: a body with one sphere geom per sphere.</summary>
        public static ObjectModelText Mjcf(double[,] centers, double[] radii, ObjectOptions? options = null) =>
            Write(NativeMethods.morphit_object_mjcf, centers, radii, options);

        private delegate Status Writer(double[] centers, double[] radii, nuint count, ref NativeObjectOptions options,
            byte[] buf, nuint capacity, out nuint needed, double[] centroid);

        private static ObjectModelText Write(Writer write, double[,] centers, double[] radii, ObjectOptions? options)
        {
            if (radii == null) throw new ArgumentNullException(nameof(radii));
            var flat = Api.Flatten(centers, nameof(centers));
            if (centers.GetLength(0) != radii.Length) throw new ArgumentException("centers and radii differ in length");
            var o = options ?? new ObjectOptions();
            if (o.Rgba == null || o.Rgba.Length != 4) throw new ArgumentException("Rgba needs 4 values");
            var name = Utf8.Alloc(o.Name);
            try
            {
                var native = new NativeObjectOptions
                {
                    Name = name,
                    Rgba = o.Rgba,
                    TotalMass = o.TotalMass,
                    Anchored = o.Anchored ? 1 : 0,
                    Decimals = o.Decimals,
                };
                var centroid = new double[3];
                var text = Api.ReadString((byte[]? b, nuint c, out nuint n) =>
                    write(flat, radii, (nuint)radii.Length, ref native, b!, c, out n, centroid));
                return new ObjectModelText(text, centroid);
            }
            finally
            {
                Marshal.FreeHGlobal(name);
            }
        }
    }

    /// <summary>Packing quality metrics.</summary>
    public static class Quality
    {
        /// <summary>
        /// Score spheres against a mesh (pass the prepared mesh the spheres were packed on,
        /// e.g. <see cref="Session.GetMesh"/>). <paramref name="masses"/> may be null to derive
        /// them from the density.
        /// </summary>
        public static QualityMetrics Evaluate(Mesh mesh, double[,] centers, double[] radii, double[]? masses = null,
            QualityOptions? options = null)
        {
            if (mesh == null) throw new ArgumentNullException(nameof(mesh));
            var flat = Api.Flatten(centers, nameof(centers));
            if (centers.GetLength(0) != radii.Length || (masses != null && masses.Length != radii.Length))
                throw new ArgumentException("centers, radii and masses differ in length");
            var o = (options ?? new QualityOptions()).ToNative();
            Api.Check(NativeMethods.morphit_evaluate_packing(mesh.Handle, flat, radii, masses, (nuint)radii.Length, ref o, out var q));
            return new QualityMetrics(q);
        }
    }
}
