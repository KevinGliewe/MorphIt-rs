//! Error type shared by the whole crate.

/// Everything that can go wrong while loading a mesh, configuring or running a session.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Reading or writing a file failed.
    #[error("I/O error: {0}")]
    Io(String),
    /// The mesh could not be parsed or is unusable (no faces, no enclosed volume, ...).
    #[error("mesh error: {0}")]
    Mesh(String),
    /// Unknown preset or key, wrong value type, or an invalid value.
    #[error("config error for `{key}`: {msg}")]
    Config { key: String, msg: String },
    /// The operation is not valid in the session's current state.
    #[error("invalid state: {0}")]
    State(String),
    /// A run was cancelled by the caller.
    #[error("cancelled")]
    Cancelled,
}

impl Error {
    pub(crate) fn config(key: impl Into<String>, msg: impl Into<String>) -> Self {
        Error::Config { key: key.into(), msg: msg.into() }
    }
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;
