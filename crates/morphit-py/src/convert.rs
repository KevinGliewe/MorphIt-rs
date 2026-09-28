//! Conversions between Rust data and Python objects (dicts via serde, NumPy arrays).

use morphit::glam::DVec3;
use numpy::ndarray::{Array1, Array2};
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::errors;

/// `(centers (n, 3), radii (n,))`.
pub(crate) type CentersRadii<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);
/// Two point sets, each (k, 3).
pub(crate) type PointSets<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

/// Any serde value as plain Python data (dicts, lists, numbers, strings).
pub(crate) fn to_py<'py, T: Serialize + ?Sized>(py: Python<'py>, v: &T) -> PyResult<Bound<'py, PyAny>> {
    pythonize::pythonize(py, v).map_err(|e| errors::value(e.to_string()))
}

/// Plain Python data into a serde type.
pub(crate) fn from_py<T: DeserializeOwned>(v: &Bound<'_, PyAny>) -> PyResult<T> {
    pythonize::depythonize(v).map_err(|e| errors::value(e.to_string()))
}

/// `(n, 3)` float64 array.
pub(crate) fn rows3<'py>(
    py: Python<'py>,
    rows: impl ExactSizeIterator<Item = [f64; 3]>,
) -> Bound<'py, PyArray2<f64>> {
    let n = rows.len();
    let flat: Vec<f64> = rows.flatten().collect();
    Array2::from_shape_vec((n, 3), flat).expect("n x 3").into_pyarray(py)
}

pub(crate) fn dvec3s<'py>(py: Python<'py>, v: &[DVec3]) -> Bound<'py, PyArray2<f64>> {
    rows3(py, v.iter().map(|c| c.to_array()))
}

pub(crate) fn flat3<'py>(py: Python<'py>, flat: &[f64]) -> Bound<'py, PyArray2<f64>> {
    rows3(py, flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]))
}

pub(crate) fn vec1<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(v).into_pyarray(py)
}

/// An `(n, 3)` array-like (NumPy array, list of triples, ...) as rows.
pub(crate) fn read_rows3(a: PyArrayLike2<'_, f64, AllowTypeChange>, what: &str) -> PyResult<Vec<[f64; 3]>> {
    let a = a.as_array();
    if a.ncols() != 3 {
        return Err(errors::value(format!("{what} must have shape (n, 3), got {:?}", a.shape())));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

pub(crate) fn read_vec1(a: PyArrayLike1<'_, f64, AllowTypeChange>) -> Vec<f64> {
    a.as_array().to_vec()
}
