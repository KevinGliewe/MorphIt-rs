/** A GPU adapter found by `initGpu()`. */
export interface GpuInfo {
  /** Index to use as `model.device = "gpu:N"`. */
  index: number;
  name: string;
  backend: string;
  /** `discrete`, `integrated`, `virtual`, `software` or `other`. */
  kind: string;
  software: boolean;
}

/** What one optimizer iteration did. Loss maps are keyed by loss name. */
export interface StepInfo {
  iteration: number;
  totalLoss: number;
  weightedLosses: Record<string, number>;
  rawLosses: Record<string, number>;
  positionGradMag: number;
  radiusGradMag: number;
  numSpheres: number;
  /** Centers moved back inside the mesh this step. */
  projected: number;
  densityControl: { added: number; removed: number; bad: number } | null;
  done: boolean;
  converged: boolean;
  seconds: number;
}

export interface StepManyOptions {
  /** Stop after this many iterations (default: all remaining). */
  maxSteps?: number;
  /** Stop once this much time has passed (milliseconds). */
  budgetMs?: number;
}

export interface InitInfo {
  seed: number;
  voxelSize: number;
  voxelCandidates: number;
}

export interface MeshInfo {
  vertices: number;
  faces: number;
  volume: number;
  area: number;
  scale: number;
  bounds: [[number, number, number], [number, number, number]];
  centerMass: [number, number, number];
  windingFlipped: boolean;
  sourcePath: string | null;
}

/** Python `MeshPrepReport` keys. */
export interface MeshPrepReport {
  action: string;
  reason: string;
  n_bodies: number;
  n_closed: number;
  n_open: number;
  n_degenerate_dropped: number;
  overlapping: boolean;
  faces_before: number;
  faces_after: number;
  volume_before: number;
  volume_after: number;
  warnings: string[];
}

export interface ObjectModelOptions {
  /** Robot / model name (default `object`). */
  name?: string;
  /** Sphere color as `#rrggbb`. */
  color?: string;
  /** Total mass in kg, split by sphere volume (default 1). */
  totalMass?: number;
  /** Weld to the world instead of a free-floating body. */
  anchored?: boolean;
  /** Decimal places of the numbers written (default 6). */
  decimals?: number;
}

export interface ObjectModel {
  text: string;
  /** Where the model's origin sits in mesh coordinates. */
  centroid: [number, number, number];
}

export interface QualityOptions {
  seed: number;
  surface_samples: number;
  volume_samples: number;
  bounds_expand: number;
  density: number;
}

export interface QualityMetrics {
  actual_n: number;
  n_out: number;
  n_tiny: number;
  r_in: number;
  r_out: number;
  r_uni: number;
  d_avg_mm: number;
  d_max_mm: number;
  mass_abs: number;
  mass_rel: number;
  com_abs: number;
  com_rel: number;
  i_abs: number;
  i_rel: number;
}

export interface LinkQuality {
  link_name: string;
  collision_index: number;
  num_spheres: number;
  d_min: number;
  d_mean: number;
  d_max: number;
  coverage: number;
  area: number;
  volume_est: number;
  spheres: { centers: [number, number, number][]; radii: number[] };
}

export interface Overall {
  num_spheres: number;
  d_min: number;
  d_mean: number;
  d_max: number;
  coverage: number;
}

/** One `<collision>` of the inspected URDF (web API keys). */
export interface CollisionItem {
  link_name: string;
  collision_index: number;
  geometry_type: string;
  action: "pack" | "remove-primitive" | "remove-already-sphere" | "error";
  /** Package path of the resolved mesh. */
  mesh_path: string | null;
  mesh_filename: string | null;
  mesh_scale: [number, number, number];
  origin_xyz: [number, number, number];
  origin_rpy: [number, number, number];
  warning: string | null;
}

export interface InspectionReport {
  robot_dir: string;
  urdf_path: string;
  urdfs_in_folder: string[];
  collisions: CollisionItem[];
  visual_count: number;
  warnings: string[];
  errors: string[];
}

export interface PackParams {
  /** `MorphIt-V`, `MorphIt-S` or `MorphIt-B` (default). */
  variant?: string;
  /** 1..200, default 20. */
  numSpheres?: number;
  /** 1..1000, default 200. */
  iterations?: number;
  seed?: number;
  /** The web UI's flat overrides, e.g. `{ coverage_weight: 2 }`. */
  advanced?: Record<string, number | boolean>;
  /** Merge overlapping closed bodies before packing (default true). */
  unionOverlappingBodies?: boolean;
}

/** A rigid transform: column-major 3x3 rotation, then translation. */
export interface Pose {
  rotation: number[];
  translation: [number, number, number];
}

export interface AssembleResult {
  urdf: string;
  linksWithCollisionsReplaced: number;
  meshCollisionsReplaced: number;
  primitiveCollisionsRemoved: number;
  sphereCollisionsRemoved: number;
  sphereChildrenAdded: number;
  /** `link[idx]: reason; ...` for pack items without spheres (empty if none). */
  skipped: string;
}
