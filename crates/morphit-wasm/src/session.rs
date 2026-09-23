use std::cell::RefCell;
use std::rc::Rc;

use js_sys::Promise;
use morphit::{Session, SessionState, StepInfo};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;
use web_time::Instant;

use crate::config::JsConfig;
use crate::dto::{InitInfoDto, StepInfoDto, to_js};
use crate::err;
use crate::mesh::JsMesh;

/// What JavaScript can still read while a step is in flight.
#[derive(Default)]
struct Snapshot {
    centers: Vec<f64>,
    radii: Vec<f64>,
    masses: Vec<f64>,
    iteration: usize,
    total_iterations: usize,
    state: &'static str,
    done: bool,
}

impl Snapshot {
    fn of(s: &Session) -> Snapshot {
        let sp = s.spheres();
        Snapshot {
            centers: sp.centers.iter().flat_map(|c| c.to_array()).collect(),
            radii: sp.radii(),
            masses: sp.masses(s.problem().density),
            iteration: s.iteration(),
            total_iterations: s.total_iterations(),
            state: match s.state() {
                SessionState::Running => "running",
                SessionState::Converged => "converged",
                SessionState::Completed => "completed",
                SessionState::Finalized => "finalized",
            },
            done: s.is_done(),
        }
    }
}

/// One packing run: initialized from a config and a mesh, advanced step by
/// step, then finalized.
///
/// While a `step`/`stepMany` promise is pending the session is busy: the
/// sphere arrays and counters still read the last completed step, every
/// other method throws.
#[wasm_bindgen(js_name = Session)]
pub struct JsSession {
    inner: Rc<RefCell<Option<Session>>>,
    snap: Rc<RefCell<Snapshot>>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct StepManyOptions {
    max_steps: Option<usize>,
    budget_ms: Option<f64>,
}

fn busy() -> JsError {
    JsError::new("session busy: a step is in progress")
}

impl JsSession {
    pub(crate) fn from_session(s: Session) -> JsSession {
        let snap = Snapshot::of(&s);
        JsSession { inner: Rc::new(RefCell::new(Some(s))), snap: Rc::new(RefCell::new(snap)) }
    }

    fn with<T>(&self, f: impl FnOnce(&Session) -> T) -> Result<T, JsError> {
        self.inner.borrow().as_ref().map(f).ok_or_else(busy)
    }

    /// Run `step_async` up to `max` times or until `budget` elapses; the
    /// session is taken out while the future runs.
    fn run(&self, max: usize, budget_ms: Option<f64>) -> Promise {
        let inner = self.inner.clone();
        let snap = self.snap.clone();
        future_to_promise(async move {
            let mut s = inner.borrow_mut().take().ok_or_else(busy)?;
            let t0 = Instant::now();
            let mut last: Option<StepInfo> = None;
            let mut failure = None;
            for _ in 0..max {
                if s.is_done() {
                    break;
                }
                match s.step_async().await {
                    Ok(info) => last = Some(info),
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                }
                if budget_ms.is_some_and(|b| t0.elapsed().as_secs_f64() * 1e3 >= b) {
                    break;
                }
            }
            *snap.borrow_mut() = Snapshot::of(&s);
            *inner.borrow_mut() = Some(s);
            if let Some(e) = failure {
                return Err(err(e).into());
            }
            match last {
                Some(info) => Ok(to_js(&StepInfoDto::from(&info))?),
                None => Ok(JsValue::UNDEFINED),
            }
        })
    }
}

#[wasm_bindgen(js_class = Session)]
impl JsSession {
    /// Sample the mesh and place the initial spheres.
    #[wasm_bindgen(constructor)]
    pub fn new(config: &JsConfig, mesh: &JsMesh) -> Result<JsSession, JsError> {
        Session::new(config.inner.clone(), mesh.inner.clone()).map(JsSession::from_session).map_err(err)
    }

    /// Run one iteration. Resolves to its `StepInfo`, or `undefined` when no
    /// iterations remain.
    #[wasm_bindgen(unchecked_return_type = "Promise<StepInfo | undefined>")]
    pub fn step(&self) -> Promise {
        self.run(1, None)
    }

    /// Run iterations until `maxSteps` (default: all remaining) have run or
    /// `budgetMs` has elapsed, whichever comes first. Resolves to the last
    /// `StepInfo`, or `undefined` when none ran. Use a budget of 8-16 ms per
    /// animation frame to keep a page responsive.
    #[wasm_bindgen(js_name = stepMany, unchecked_return_type = "Promise<StepInfo | undefined>")]
    pub fn step_many(
        &self,
        #[wasm_bindgen(unchecked_param_type = "StepManyOptions")] options: JsValue,
    ) -> Promise {
        let opts: StepManyOptions = if options.is_undefined() || options.is_null() {
            StepManyOptions::default()
        } else {
            match serde_wasm_bindgen::from_value(options) {
                Ok(o) => o,
                Err(e) => return Promise::reject(&err(e).into()),
            }
        };
        self.run(opts.max_steps.unwrap_or(usize::MAX), opts.budget_ms)
    }

    /// One iteration, synchronously. Throws when the searches run on WebGPU
    /// (see `needsAsync`).
    #[wasm_bindgen(js_name = stepSync, unchecked_return_type = "StepInfo")]
    pub fn step_sync(&self) -> Result<JsValue, JsError> {
        let mut guard = self.inner.borrow_mut();
        let s = guard.as_mut().ok_or_else(busy)?;
        let info = s.step().map_err(err)?;
        *self.snap.borrow_mut() = Snapshot::of(s);
        to_js(&StepInfoDto::from(&info))
    }

    /// No iterations remain (converged, completed or finalized).
    #[wasm_bindgen(getter, js_name = isDone)]
    pub fn is_done(&self) -> bool {
        self.snap.borrow().done
    }

    /// Iterations run so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.snap.borrow().iteration
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.snap.borrow().total_iterations
    }

    /// `running`, `converged`, `completed` or `finalized`.
    #[wasm_bindgen(getter)]
    pub fn state(&self) -> String {
        self.snap.borrow().state.to_string()
    }

    /// Whether `stepSync` is unavailable (WebGPU searches).
    #[wasm_bindgen(getter, js_name = needsAsync)]
    pub fn needs_async(&self) -> Result<bool, JsError> {
        self.with(Session::needs_async)
    }

    /// `cpu` or `gpu:N (name)`.
    #[wasm_bindgen(getter)]
    pub fn device(&self) -> Result<String, JsError> {
        self.with(|s| s.device().to_string())
    }

    #[wasm_bindgen(js_name = initInfo, unchecked_return_type = "InitInfo")]
    pub fn init_info(&self) -> Result<JsValue, JsError> {
        to_js(&InitInfoDto::from(self.with(Session::init_info)?))
    }

    /// Sphere centers as flat `x y z` (from the last completed step).
    pub fn centers(&self) -> Vec<f64> {
        self.snap.borrow().centers.clone()
    }

    pub fn radii(&self) -> Vec<f64> {
        self.snap.borrow().radii.clone()
    }

    pub fn masses(&self) -> Vec<f64> {
        self.snap.borrow().masses.clone()
    }

    #[wasm_bindgen(js_name = lastStep, unchecked_return_type = "StepInfo | undefined")]
    pub fn last_step(&self) -> Result<JsValue, JsError> {
        match self.with(|s| s.last_step().map(StepInfoDto::from))? {
            Some(d) => to_js(&d),
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Remove spheres whose centers left the mesh and end the session.
    /// Returns how many were removed.
    pub fn finalize(&self) -> Result<usize, JsError> {
        let mut guard = self.inner.borrow_mut();
        let s = guard.as_mut().ok_or_else(busy)?;
        let n = s.finalize();
        *self.snap.borrow_mut() = Snapshot::of(s);
        Ok(n)
    }

    /// The current spheres in the Python result JSON schema.
    #[wasm_bindgen(js_name = resultJson)]
    pub fn result_json(&self) -> Result<String, JsError> {
        self.with(|s| s.result().to_json_string())
    }

    /// Per-iteration history (Python `training_history` keys).
    #[wasm_bindgen(js_name = historyJson)]
    pub fn history_json(&self) -> Result<String, JsError> {
        self.with(|s| s.history().to_json().to_string())
    }

    #[wasm_bindgen(js_name = configJson)]
    pub fn config_json(&self) -> Result<String, JsError> {
        self.with(|s| s.config().to_json_value().to_string())
    }

    #[wasm_bindgen(js_name = meshPrepJson)]
    pub fn mesh_prep_json(&self) -> Result<String, JsError> {
        self.with(|s| serde_json::to_string(s.mesh_prep()))?.map_err(err)
    }

    /// The mesh being packed (the union when mesh preparation merged bodies).
    pub fn mesh(&self) -> Result<JsMesh, JsError> {
        self.with(|s| JsMesh { inner: s.mesh().clone() })
    }
}
