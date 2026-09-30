using System;
using System.Collections.Generic;
using MorphIt.Native;

namespace MorphIt
{
    /// <summary>Status codes of the C API (negative values are errors).</summary>
    public enum Status
    {
        Ok = 0,
        Done = 1,
        NullArg = -1,
        InvalidArg = -2,
        Io = -3,
        Mesh = -4,
        Config = -5,
        State = -6,
        Busy = -7,
        Cancelled = -8,
        BufferTooSmall = -9,
        Panic = -10,
        ThreadPool = -11,
    }

    /// <summary>A failed MorphIt call; <see cref="Status"/> tells what kind.</summary>
    public sealed class MorphItException : Exception
    {
        public MorphItException(Status status, string message) : base(message) => Status = status;

        public Status Status { get; }
    }

    public enum SessionState { Running = 0, Converged = 1, Completed = 2, Finalized = 3 }

    /// <summary>How <see cref="Session.Run"/> ended.</summary>
    public enum RunOutcome { Completed, Converged, Cancelled }

    /// <summary>What one iteration did.</summary>
    public sealed class StepInfo
    {
        /// <summary>Loss names, in the order of <see cref="WeightedLosses"/> and <see cref="RawLosses"/>.</summary>
        public static IReadOnlyList<string> LossNames { get; } = new[]
        {
            "coverage_loss", "overlap_penalty", "boundary_penalty", "surface_loss", "containment_loss", "sqem_loss",
            "hausdorff_loss", "mesh_containment_loss", "mass_loss", "com_loss", "inertia_loss", "flatness_loss",
        };

        internal StepInfo(in NativeStepInfo n)
        {
            Iteration = (long)n.Iteration;
            TotalLoss = n.TotalLoss;
            WeightedLosses = n.WeightedLosses;
            RawLosses = n.RawLosses;
            PositionGradMag = n.PositionGradMag;
            RadiusGradMag = n.RadiusGradMag;
            NumSpheres = (int)n.NumSpheres;
            Projected = (int)n.Projected;
            DensityControlFired = n.DensityControlFired != 0;
            DensityControlReplaced = (int)n.DensityControlReplaced;
            Done = n.Done != 0;
            Converged = n.Converged != 0;
            Seconds = n.Seconds;
        }

        /// <summary>Zero-based index of the iteration that ran.</summary>
        public long Iteration { get; }
        public double TotalLoss { get; }
        /// <summary>weight x value per loss, indexed like <see cref="LossNames"/>.</summary>
        public IReadOnlyList<double> WeightedLosses { get; }
        /// <summary>Unweighted value per loss, indexed like <see cref="LossNames"/>.</summary>
        public IReadOnlyList<double> RawLosses { get; }
        public double PositionGradMag { get; }
        public double RadiusGradMag { get; }
        public int NumSpheres { get; }
        public int Projected { get; }
        public bool DensityControlFired { get; }
        public int DensityControlReplaced { get; }
        public bool Done { get; }
        public bool Converged { get; }
        public double Seconds { get; }

        public override string ToString() => $"StepInfo(iteration={Iteration}, total_loss={TotalLoss:G6}, spheres={NumSpheres})";
    }

    /// <summary>Session progress, readable at any time from any thread.</summary>
    public sealed class StateInfo
    {
        internal StateInfo(in NativeStateInfo n)
        {
            State = (SessionState)n.State;
            Iteration = (long)n.Iteration;
            TotalIterations = (long)n.TotalIterations;
            NumSpheres = (int)n.NumSpheres;
            TotalLoss = n.TotalLoss;
            Running = n.Running != 0;
            DensityControlPasses = (long)n.DensityControlPasses;
            Pruned = (int)n.Pruned;
            Seed = n.Seed;
        }

        public SessionState State { get; }
        public long Iteration { get; }
        public long TotalIterations { get; }
        public int NumSpheres { get; }
        /// <summary>Loss of the last iteration (NaN before the first).</summary>
        public double TotalLoss { get; }
        public bool Running { get; }
        public long DensityControlPasses { get; }
        /// <summary>Spheres removed by the final prune (escaped centers, and spheres inside another sphere).</summary>
        public int Pruned { get; }
        public ulong Seed { get; }
    }

    /// <summary>Mesh properties.</summary>
    public sealed class MeshInfo
    {
        internal MeshInfo(in NativeMeshInfo n)
        {
            NumVertices = (int)n.NumVertices;
            NumFaces = (int)n.NumFaces;
            Volume = n.Volume;
            Area = n.Area;
            Scale = n.Scale;
            BoundsMin = n.BoundsMin;
            BoundsMax = n.BoundsMax;
            CenterMass = n.CenterMass;
            Inertia = n.Inertia;
            WindingFlipped = n.WindingFlipped != 0;
        }

        public int NumVertices { get; }
        public int NumFaces { get; }
        public double Volume { get; }
        public double Area { get; }
        /// <summary>Length of the bounding-box diagonal.</summary>
        public double Scale { get; }
        public IReadOnlyList<double> BoundsMin { get; }
        public IReadOnlyList<double> BoundsMax { get; }
        public IReadOnlyList<double> CenterMass { get; }
        /// <summary>Inertia tensor about the center of mass at density 1, row-major 3x3.</summary>
        public IReadOnlyList<double> Inertia { get; }
        /// <summary>True if the input was wound inward and all faces were flipped.</summary>
        public bool WindingFlipped { get; }
    }

    /// <summary>Sampling settings of the quality metrics (defaults match the Python debug_quick_eval.py).</summary>
    public sealed class QualityOptions
    {
        public QualityOptions()
        {
            Api.Check(NativeMethods.morphit_quality_options_default(out var d));
            Seed = d.Seed;
            SurfaceSamples = (int)d.SurfaceSamples;
            VolumeSamples = (int)d.VolumeSamples;
            BoundsExpand = d.BoundsExpand;
            Density = d.Density;
        }

        public ulong Seed { get; set; }
        public int SurfaceSamples { get; set; }
        public int VolumeSamples { get; set; }
        public double BoundsExpand { get; set; }
        public double Density { get; set; }

        internal NativeQualityOptions ToNative() => new NativeQualityOptions
        {
            Seed = Seed,
            SurfaceSamples = (nuint)SurfaceSamples,
            VolumeSamples = (nuint)VolumeSamples,
            BoundsExpand = BoundsExpand,
            Density = Density,
        };
    }

    /// <summary>Quality of a packing (the debug_quick_eval metrics).</summary>
    public sealed class QualityMetrics
    {
        internal QualityMetrics(in NativeQualityMetrics n)
        {
            ActualN = (int)n.ActualN;
            NOut = (int)n.NOut;
            NTiny = (int)n.NTiny;
            RIn = n.RIn;
            ROut = n.ROut;
            RUni = n.RUni;
            DAvgMm = n.DAvgMm;
            DMaxMm = n.DMaxMm;
            MassAbs = n.MassAbs;
            MassRel = n.MassRel;
            ComAbs = n.ComAbs;
            ComRel = n.ComRel;
            IAbs = n.IAbs;
            IRel = n.IRel;
        }

        public int ActualN { get; }
        /// <summary>Spheres whose center lies outside the mesh.</summary>
        public int NOut { get; }
        /// <summary>Spheres with radius below 0.001 x mesh scale.</summary>
        public int NTiny { get; }
        /// <summary>Volume covered by spheres inside the mesh, relative to the mesh volume.</summary>
        public double RIn { get; }
        /// <summary>Volume covered by spheres outside the mesh, relative to the mesh volume.</summary>
        public double ROut { get; }
        /// <summary>Union volume of the spheres relative to the mesh volume.</summary>
        public double RUni { get; }
        /// <summary>Mean absolute surface distance in millimetres (mesh units x 1000).</summary>
        public double DAvgMm { get; }
        public double DMaxMm { get; }
        public double MassAbs { get; }
        public double MassRel { get; }
        public double ComAbs { get; }
        public double ComRel { get; }
        public double IAbs { get; }
        public double IRel { get; }
    }
}
