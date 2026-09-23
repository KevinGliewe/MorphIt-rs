use morphit::{Config, Preset};
use wasm_bindgen::prelude::*;

use crate::dto::to_js;
use crate::err;

/// Optimizer configuration (the Python `config.py` tree; dotted keys such as
/// `model.num_spheres` or `training.iterations`).
#[wasm_bindgen(js_name = Config)]
#[derive(Clone)]
pub struct JsConfig {
    pub(crate) inner: Config,
}

#[wasm_bindgen(js_class = Config)]
impl JsConfig {
    /// A preset: `MorphIt-V`, `MorphIt-S`, `MorphIt-B`, ...
    #[wasm_bindgen(js_name = fromPreset)]
    pub fn from_preset(name: &str) -> Result<JsConfig, JsError> {
        Config::from_preset_name(name).map(|inner| JsConfig { inner }).map_err(err)
    }

    /// A full or partial config as JSON (missing keys take defaults).
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<JsConfig, JsError> {
        Config::from_json_str(json).map(|inner| JsConfig { inner }).map_err(err)
    }

    /// The preset names.
    pub fn presets() -> Vec<String> {
        Preset::ALL.iter().map(|p| p.name().to_string()).collect()
    }

    /// The value at a dotted key.
    pub fn get(&self, key: &str) -> Result<JsValue, JsError> {
        to_js(&self.inner.get(key).map_err(err)?)
    }

    /// Set a dotted key; the value is type-checked.
    pub fn set(&mut self, key: &str, value: JsValue) -> Result<(), JsError> {
        let v: serde_json::Value = serde_wasm_bindgen::from_value(value).map_err(err)?;
        self.inner.set(key, v).map_err(err)
    }

    /// Apply `{"dotted.key": value, ...}`.
    #[wasm_bindgen(js_name = applyJson)]
    pub fn apply_json(&mut self, json: &str) -> Result<(), JsError> {
        self.inner.apply_updates_json(json).map_err(err)
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsError> {
        serde_json::to_string(&self.inner.to_json_value()).map_err(err)
    }

    #[wasm_bindgen(js_name = clone)]
    pub fn clone_js(&self) -> JsConfig {
        self.clone()
    }
}
