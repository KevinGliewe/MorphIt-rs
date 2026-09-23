//! A packing job: one object mesh, or the `pack` items of a robot one after
//! another. Written once as an async function; [`crate::task::spawn`] runs
//! it on a thread natively and on the page's event loop in the browser. The
//! UI reads the latest [`JobStatus`] snapshot each frame.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use morphit::{Config, LossId, Mesh, PackResult, Session};
use morphit_robot::config::PackParams;
use morphit_robot::inspect::CollisionItem;
use morphit_robot::pack::{link_config, pack_mesh_path};
use morphit_robot::vfs::MemPackage;

use crate::task::{Slot, Yielder, spawn};

/// What to pack.
pub enum JobSpec {
    Object { mesh: Arc<Mesh>, config: Box<Config> },
    Robot { pkg: Arc<MemPackage>, items: Vec<CollisionItem>, params: PackParams, device: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Phase {
    /// Sampling the mesh and placing the first spheres.
    #[default]
    Preparing,
    Running,
    Paused,
    Done,
    Cancelled,
    Failed,
}

/// One finished pack: `None` for the object, `(link, collision index)` for
/// a robot item.
pub type Finished = (Option<(String, usize)>, PackResult);

/// The live state the UI shows.
#[derive(Clone, Default)]
pub struct JobStatus {
    pub phase: Phase,
    /// Index into the robot items (0 for an object).
    pub item: usize,
    pub items: usize,
    pub device: String,
    pub iteration: usize,
    pub total_iterations: usize,
    pub centers: Vec<[f64; 3]>,
    pub radii: Vec<f64>,
    /// `(iteration, total loss, weighted loss terms)` of the current item.
    pub history: Vec<(usize, f64, [f64; LossId::COUNT])>,
    pub finished: Vec<Finished>,
    pub error: Option<String>,
}

/// Shared between the job and the UI.
#[derive(Default)]
pub struct JobShared {
    status: Mutex<JobStatus>,
    version: AtomicU64,
    pub cancel: AtomicBool,
    pub pause: AtomicBool,
    /// Stop iterating the current item and finalize it now.
    pub finish_now: AtomicBool,
}

impl JobShared {
    /// Bumped on every change of the status.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    pub fn status(&self) -> JobStatus {
        self.status.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn update(&self, f: impl FnOnce(&mut JobStatus)) {
        f(&mut self.status.lock().unwrap_or_else(|e| e.into_inner()));
        self.version.fetch_add(1, Ordering::Release);
    }

    pub fn is_running(&self) -> bool {
        !matches!(self.status().phase, Phase::Done | Phase::Cancelled | Phase::Failed)
    }
}

/// Start `spec`; the returned handle observes and steers it.
pub fn start(spec: JobSpec) -> Arc<JobShared> {
    let shared = Arc::new(JobShared::default());
    let s = shared.clone();
    let _: Slot<()> = spawn(async move {
        let mut y = Yielder::default();
        let outcome = run(spec, &s, &mut y).await;
        s.update(|st| match outcome {
            Ok(()) if s.cancel.load(Ordering::Relaxed) => st.phase = Phase::Cancelled,
            Ok(()) => st.phase = Phase::Done,
            Err(e) => {
                st.phase = Phase::Failed;
                st.error = Some(e);
            }
        });
    });
    shared
}

async fn run(spec: JobSpec, s: &JobShared, y: &mut Yielder) -> Result<(), String> {
    match spec {
        JobSpec::Object { mesh, config } => {
            s.update(|st| st.items = 1);
            let r = pack_one(mesh, *config, s, y).await?;
            if let Some(r) = r {
                s.update(|st| st.finished.push((None, r)));
            }
        }
        JobSpec::Robot { pkg, items, params, device } => {
            s.update(|st| st.items = items.len());
            for (i, item) in items.iter().enumerate() {
                if s.cancel.load(Ordering::Relaxed) {
                    break;
                }
                s.update(|st| {
                    st.item = i;
                    st.phase = Phase::Preparing;
                    st.history.clear();
                });
                y.yield_now().await;
                let path = pack_mesh_path(item).map_err(|e| e.to_string())?;
                let mesh = Arc::new(pkg.load_mesh(path).map_err(|e| format!("{path}: {e}"))?);
                let (config, _) =
                    link_config(item, &params, &device, "spheres").map_err(|e| e.to_string())?;
                if let Some(r) = pack_one(mesh, config, s, y).await? {
                    let key = (item.link_name.clone(), item.collision_index);
                    s.update(|st| st.finished.push((Some(key), r)));
                }
            }
        }
    }
    Ok(())
}

/// Pack one mesh; `None` when cancelled.
async fn pack_one(
    mesh: Arc<Mesh>,
    config: Config,
    s: &JobShared,
    y: &mut Yielder,
) -> Result<Option<PackResult>, String> {
    s.update(|st| st.phase = Phase::Preparing);
    y.yield_now().await;
    let mut session = Session::new(config, mesh).map_err(|e| e.to_string())?;
    let publish = |session: &Session, st: &mut JobStatus| {
        st.centers = session.spheres().centers.iter().map(|c| c.to_array()).collect();
        st.radii = session.spheres().radii();
        st.iteration = session.iteration();
        st.total_iterations = session.total_iterations();
        st.device = session.device().to_string();
    };
    s.update(|st| {
        publish(&session, st);
        st.phase = Phase::Running;
        st.history.clear();
    });
    s.finish_now.store(false, Ordering::Relaxed);
    while !session.is_done() {
        if s.pause.load(Ordering::Relaxed) {
            s.update(|st| st.phase = Phase::Paused);
            while s.pause.load(Ordering::Relaxed) && !s.cancel.load(Ordering::Relaxed) {
                y.idle().await;
            }
            s.update(|st| st.phase = Phase::Running);
        }
        if s.cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        if s.finish_now.swap(false, Ordering::Relaxed) {
            break;
        }
        let info = session.step_async().await.map_err(|e| e.to_string())?;
        s.update(|st| {
            publish(&session, st);
            st.history.push((info.iteration, info.total_loss, info.weighted_losses));
        });
        y.maybe_yield().await;
    }
    session.finalize();
    s.update(|st| publish(&session, st));
    Ok(Some(session.result()))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use morphit::glam::DVec3;
    use morphit_robot::inspect::inspect_urdf_in;

    fn wait(job: &JobShared) -> JobStatus {
        let t0 = std::time::Instant::now();
        while job.is_running() {
            assert!(t0.elapsed().as_secs() < 120, "job did not finish");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        job.status()
    }

    fn cube() -> Mesh {
        morphit::shapes::box_mesh(DVec3::ZERO, DVec3::new(0.1, 0.06, 0.04))
    }

    fn params(iterations: usize) -> PackParams {
        PackParams {
            variant: "MorphIt-B".into(),
            num_spheres: 6,
            iterations,
            seed: Some(3),
            advanced: vec![],
            union_overlapping_bodies: true,
        }
    }

    #[test]
    fn object_job_equals_a_direct_pack() {
        let config = params(12).config("cpu", "", "cube.json").unwrap();
        let job = start(JobSpec::Object { mesh: Arc::new(cube()), config: Box::new(config.clone()) });
        let st = wait(&job);
        assert_eq!(st.phase, Phase::Done, "{:?}", st.error);
        assert_eq!(st.history.len(), 12);
        assert_eq!(st.finished.len(), 1);
        let direct = morphit::pack(config, Arc::new(cube())).unwrap();
        assert_eq!(st.finished[0].1, direct);
        assert_eq!(st.radii, direct.radii);
    }

    #[test]
    fn robot_job_packs_every_item_and_can_be_cancelled() {
        let mut stl = String::from("solid c\n");
        for t in cube().triangles() {
            stl += "facet normal 0 0 0\nouter loop\n";
            for v in t {
                stl += &format!("vertex {} {} {}\n", v.x, v.y, v.z);
            }
            stl += "endloop\nendfacet\n";
        }
        stl += "endsolid c\n";
        let urdf = r#"<robot name="r">
  <link name="a"><collision><geometry><mesh filename="package://r/c.stl"/></geometry></collision></link>
  <link name="b"><collision><geometry><mesh filename="c.stl"/></geometry></collision></link>
</robot>"#;
        let pkg = Arc::new(
            MemPackage::from_files([("r/r.urdf", urdf.as_bytes().to_vec()), ("r/c.stl", stl.into_bytes())])
                .unwrap(),
        );
        let items: Vec<_> = inspect_urdf_in(&pkg, "r/r.urdf").unwrap().to_pack().cloned().collect();
        assert_eq!(items.len(), 2);

        let spec = |iterations| JobSpec::Robot {
            pkg: pkg.clone(),
            items: items.clone(),
            params: params(iterations),
            device: "cpu".into(),
        };
        let st = wait(&start(spec(5)));
        assert_eq!(st.phase, Phase::Done, "{:?}", st.error);
        let keys: Vec<_> = st.finished.iter().map(|(k, _)| k.clone().unwrap()).collect();
        assert_eq!(keys, [("a".to_string(), 0), ("b".to_string(), 0)]);

        let job = start(spec(1000));
        job.cancel.store(true, Ordering::Relaxed);
        let st = wait(&job);
        assert_eq!(st.phase, Phase::Cancelled);
        assert!(st.finished.is_empty());
    }

    #[test]
    fn manifest_lists_the_bundled_robots() {
        let robots = crate::io::example_robots();
        assert_eq!(robots.len(), morphit_robot::examples::EXAMPLE_ROBOTS.len());
        let (_, files) = crate::io::ROBOT_FILES.iter().find(|(f, _)| *f == "kinova_description").unwrap();
        assert!(files.iter().any(|f| f.ends_with("m1n4s200_standalone.urdf")));
        assert!(files.iter().all(|f| !f.contains('\\')));
    }
}
