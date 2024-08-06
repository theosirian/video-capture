use thiserror::Error;

pub(crate) use crate::base::error::BaseError;

#[derive(Error, Debug, Clone)]
pub enum DeviceError {
    #[error(transparent)]
    BaseError(#[from] BaseError),
    #[error("Open failed: {0}")]
    OpenFailed(String),
    #[error("Close failed: {0}")]
    CloseFailed(String),
    #[error("Start failed: {0}")]
    StartFailed(String),
    #[error("Stop failed: {0}")]
    StopFailed(String),
    #[error("Not running: {0}")]
    NotRunning(String),
    #[error("Get failed: {0}")]
    GetFailed(String),
    #[error("Set failed: {0}")]
    SetFailed(String),
    #[error("Read failed: {0}")]
    ReadFailed(String),
    #[error("Write failed: {0}")]
    WriteFailed(String),
}
