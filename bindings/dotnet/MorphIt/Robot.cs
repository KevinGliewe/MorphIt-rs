using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MorphIt.Native;

namespace MorphIt
{
    /// <summary>One collision to replace with spheres.</summary>
    public sealed class PackItem
    {
        internal PackItem(int index, string link, int collisionIndex)
        {
            Index = index;
            Link = link;
            CollisionIndex = collisionIndex;
        }

        /// <summary>Position in <see cref="InspectionReport.PackItems"/>; pass it to <see cref="RobotPackage.PackLink"/>.</summary>
        public int Index { get; }
        public string Link { get; }
        /// <summary>Index of the collision within its link.</summary>
        public int CollisionIndex { get; }

        public override string ToString() => $"{Link}[{CollisionIndex}]";
    }

    /// <summary>Parameters of <see cref="RobotPackage.PackLink"/> (the web API's defaults and limits).</summary>
    public sealed class PackParams
    {
        /// <summary>MorphIt-V, MorphIt-S or MorphIt-B.</summary>
        public string Variant { get; set; } = "MorphIt-B";
        /// <summary>1 to 200.</summary>
        public int NumSpheres { get; set; } = 20;
        /// <summary>1 to 1000.</summary>
        public int Iterations { get; set; } = 200;
        /// <summary>Random seed; null draws one at random.</summary>
        public ulong? Seed { get; set; }
        public bool UnionOverlappingBodies { get; set; } = true;
        public bool ConvexHull { get; set; }
        /// <summary>The web UI's advanced overrides as a JSON object, e.g. {"coverage_weight": 2000}.</summary>
        public string? AdvancedJson { get; set; }
    }

    /// <summary>What <see cref="RobotPackage.Assemble"/> changed.</summary>
    public sealed class AssembledUrdf
    {
        internal AssembledUrdf(string urdf, in NativeAssembleStats s)
        {
            Urdf = urdf;
            LinksWithCollisionsReplaced = (int)s.LinksWithCollisionsReplaced;
            MeshCollisionsReplaced = (int)s.MeshCollisionsReplaced;
            PrimitiveCollisionsRemoved = (int)s.PrimitiveCollisionsRemoved;
            SphereCollisionsRemoved = (int)s.SphereCollisionsRemoved;
            SphereChildrenAdded = (int)s.SphereChildrenAdded;
        }

        public string Urdf { get; }
        public int LinksWithCollisionsReplaced { get; }
        public int MeshCollisionsReplaced { get; }
        public int PrimitiveCollisionsRemoved { get; }
        public int SphereCollisionsRemoved { get; }
        public int SphereChildrenAdded { get; }
    }

    /// <summary>The inspection of one URDF: every collision and what to do with it. Immutable.</summary>
    public sealed class InspectionReport : IDisposable
    {
        internal readonly ReportHandle Handle;

        internal InspectionReport(ReportHandle handle) => Handle = handle;

        /// <summary>The full report as JSON (collisions, actions, warnings, ...).</summary>
        public string Json =>
            Api.ReadString((byte[]? b, nuint c, out nuint n) => NativeMethods.morphit_report_json(Handle, b!, c, out n));

        /// <summary>The collisions to pack, in order.</summary>
        public IReadOnlyList<PackItem> PackItems
        {
            get
            {
                Api.Check(NativeMethods.morphit_report_pack_count(Handle, out var count));
                var items = new PackItem[(int)count];
                for (int i = 0; i < items.Length; i++)
                {
                    nuint index = (nuint)i;
                    nuint collision = 0;
                    var link = Api.ReadString((byte[]? b, nuint c, out nuint n) =>
                        NativeMethods.morphit_report_pack_item(Handle, index, b!, c, out n, out collision));
                    items[i] = new PackItem(i, link, (int)collision);
                }
                return items;
            }
        }

        public void Dispose() => Handle.Dispose();
    }

    /// <summary>
    /// A robot description package (URDF plus meshes) held in memory: inspect it,
    /// pack each collision mesh (a normal <see cref="Session"/>), and assemble the
    /// spherical URDF. May be used from several threads.
    /// </summary>
    public sealed class RobotPackage : IDisposable
    {
        internal readonly RobotHandle Handle;

        private RobotPackage(RobotHandle handle) => Handle = handle;

        /// <summary>An empty package; add files with <see cref="AddFile"/>.</summary>
        public RobotPackage()
        {
            Api.Check(NativeMethods.morphit_robot_new(out Handle));
        }

        /// <summary>Every file under the folder (the package root).</summary>
        public static RobotPackage FromFolder(string path)
        {
            Api.Check(NativeMethods.morphit_robot_from_folder(Utf8.Encode(path)!, out var h));
            return new RobotPackage(h);
        }

        /// <summary>A package from the bytes of a .zip archive.</summary>
        public static RobotPackage FromZip(byte[] zip)
        {
            if (zip == null) throw new ArgumentNullException(nameof(zip));
            Api.Check(NativeMethods.morphit_robot_from_zip(zip, (nuint)zip.Length, out var h));
            return new RobotPackage(h);
        }

        /// <summary>Add (or replace) a file under its path relative to the package root.</summary>
        public void AddFile(string path, byte[] data)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            Api.Check(NativeMethods.morphit_robot_add_file(Handle, Utf8.Encode(path)!, data, (nuint)data.Length));
        }

        /// <summary>Inspect a URDF (by file name; null: the only one).</summary>
        public InspectionReport Inspect(string? urdf = null)
        {
            Api.Check(NativeMethods.morphit_robot_inspect(Handle, Utf8.Encode(urdf), out var h));
            return new InspectionReport(h);
        }

        /// <summary>A session packing pack item <paramref name="item"/>; run it, then <see cref="SetLinkResult"/>.</summary>
        public Session PackLink(InspectionReport report, int item, PackParams? parameters = null, string? device = null)
        {
            if (report == null) throw new ArgumentNullException(nameof(report));
            var p = parameters ?? new PackParams();
            var variant = Utf8.Alloc(p.Variant);
            var advanced = Utf8.Alloc(p.AdvancedJson);
            try
            {
                var native = new NativePackParams
                {
                    Variant = variant,
                    NumSpheres = (nuint)Math.Max(0, p.NumSpheres),
                    Iterations = (nuint)Math.Max(0, p.Iterations),
                    Seed = p.Seed.HasValue ? (long)p.Seed.Value : -1,
                    UnionOverlappingBodies = p.UnionOverlappingBodies ? 1 : 0,
                    ConvexHull = p.ConvexHull ? 1 : 0,
                    AdvancedJson = advanced,
                };
                Api.Check(NativeMethods.morphit_robot_pack_link(Handle, report.Handle, (nuint)item, ref native, Utf8.Encode(device), out var s));
                return new Session(s);
            }
            finally
            {
                Marshal.FreeHGlobal(variant);
                Marshal.FreeHGlobal(advanced);
            }
        }

        /// <summary>Record the session's current spheres as the result of pack item <paramref name="item"/>.</summary>
        public void SetLinkResult(InspectionReport report, int item, Session session) =>
            Api.Check(NativeMethods.morphit_robot_set_link_result(Handle, report.Handle, (nuint)item, session.Handle));

        /// <summary>Record the spheres of link[collisionIndex] from a result JSON.</summary>
        public void SetLinkResultJson(string link, int collisionIndex, string resultJson) =>
            Api.Check(NativeMethods.morphit_robot_set_link_result_json(Handle, Utf8.Encode(link)!, (nuint)collisionIndex, Utf8.Encode(resultJson)!));

        public void ClearLinkResults() => Api.Check(NativeMethods.morphit_robot_clear_link_results(Handle));

        /// <summary>
        /// The URDF with every packed collision replaced by sphere links. <paramref name="baseColor"/>
        /// is "#rrggbb"; <paramref name="colorVariation"/> (0..1) spreads the hue over the links.
        /// </summary>
        public AssembledUrdf Assemble(InspectionReport report, string? baseColor = null, double colorVariation = 0.0)
        {
            if (report == null) throw new ArgumentNullException(nameof(report));
            NativeAssembleStats stats = default;
            var color = Utf8.Encode(baseColor);
            var urdf = Api.ReadString((byte[]? b, nuint c, out nuint n) =>
                NativeMethods.morphit_robot_assemble(Handle, report.Handle, color, colorVariation, b!, c, out n, out stats));
            return new AssembledUrdf(urdf, stats);
        }

        public void Dispose() => Handle.Dispose();
    }
}
