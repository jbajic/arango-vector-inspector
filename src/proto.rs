//! Wire protocol between a local UI client and a remote `--serve` process.
//!
//! Frames are length-prefixed bincode: a little-endian `u32` byte count
//! followed by the bincode-encoded payload. bincode (not JSON) keeps the
//! large `f32` vector payloads compact. The stream rides the ssh child's
//! stdin/stdout, so nothing else may be written to those descriptors.

use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::io::{Read, Write};

use crate::scan::ScanResult;

/// Bumped on any incompatible change to `Request`/`Response`. The server sends
/// its version in the opening `Hello`; the client refuses a mismatch.
pub const PROTOCOL_VERSION: u32 = 1;

/// A request from the client to the serve process.
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Request {
    /// Scan the VectorIndex and definitions CFs.
    Scan,
    /// Read every document vector in the given centroid lists.
    ReadVectors {
        object_id: u64,
        list_ids: Vec<u64>,
        dim: usize,
    },
}

/// A response from the serve process to the client.
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Response {
    /// Sent once, unsolicited, right after the DB opens.
    Hello {
        version: u32,
    },
    Scan(ScanResult),
    Vectors(Vec<(u64, u64, Vec<f32>)>),
    /// A recoverable server-side error; the connection stays usable.
    Error(String),
}

/// Serialize `msg` as one length-prefixed frame and flush.
pub fn write_frame<W: Write, T: Serialize>(w: &mut W, msg: &T) -> Result<()> {
    let bytes = bincode::serialize(msg).context("bincode serialize")?;
    let len = u32::try_from(bytes.len()).context("frame too large")?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

/// Read one length-prefixed frame. Returns an error on EOF (peer closed).
pub fn read_frame<R: Read, T: DeserializeOwned>(r: &mut R) -> Result<T> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)
        .context("reading frame length (peer closed?)")?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).context("reading frame body")?;
    bincode::deserialize(&buf).context("bincode deserialize")
}

/// Convenience for turning an unexpected response variant into an error.
pub fn unexpected(what: &str) -> anyhow::Error {
    anyhow!("unexpected response to {what}")
}
