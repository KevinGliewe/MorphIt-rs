//! Robot and object pipelines around the MorphIt optimizer, ported from the
//! Python reference's `src/scripts/create_object_urdf.py` and
//! `src/scripts/robot/*.py`:
//!
//! - [`object_model`]: a packed object as a URDF or MJCF model of rigidly
//!   joined spheres, and reading the spheres back from the URDF.
//! - [`inspect`]: find and classify the `<collision>` elements of a robot
//!   description package.
//! - [`pack`]: pack one collision mesh and write its sphere JSON.
//! - [`assemble`]: rewrite the robot URDF with sphere children in place of
//!   the packed collisions.
//! - [`quality`]: surface-distance and coverage metrics of a packing.
//! - [`vfs`]: robot packages held in memory (browser uploads, zip archives),
//!   with in-memory variants of inspection and assembly.
//! - [`kinematics`]: link poses at the zero joint configuration.
//! - [`examples`]: the registry of the bundled example library.
//!
//! Plus the small helpers the HTTP API shares with them ([`color`],
//! [`history`], [`config`], [`paths`]).

pub mod assemble;
pub mod color;
pub mod config;
pub mod examples;
pub mod history;
pub mod inspect;
pub mod kinematics;
pub mod object_model;
pub mod pack;
pub mod paths;
pub mod quality;
pub mod vfs;
mod xml;

/// Errors of this crate. The `Display` text is what the HTTP API reports.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid input (maps to HTTP 400).
    #[error("{0}")]
    Invalid(String),
    /// File system failure.
    #[error("{0}")]
    Io(String),
    /// Failure inside the optimizer or mesh loading.
    #[error(transparent)]
    Morphit(#[from] morphit::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Python `repr` of a string: single quotes unless the text contains a
/// single quote and no double quote. Used to keep error messages identical.
pub fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `repr` of a list of strings, e.g. `['a.urdf', 'b.urdf']`.
pub fn py_list(items: &[String]) -> String {
    format!("[{}]", items.iter().map(|s| py_repr(s)).collect::<Vec<_>>().join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_reprs() {
        assert_eq!(py_repr("abc"), "'abc'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(py_repr("a\\b"), "'a\\\\b'");
        assert_eq!(py_list(&["a.urdf".into(), "b.urdf".into()]), "['a.urdf', 'b.urdf']");
        assert_eq!(py_list(&[]), "[]");
    }
}
