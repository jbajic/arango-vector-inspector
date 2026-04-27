//! Decode FAISS-serialized IVF index blobs and extract centroid vectors.
//!
//! ArangoDB stores trained vector index data as a VPack-wrapped FAISS index
//! (`{codeData: <faiss bytes>}`). We reconstruct the centroid matrix by:
//!  1. writing the FAISS bytes to a temp file (the C API has no buffered
//!     reader),
//!  2. calling `faiss_read_index`,
//!  3. casting to `IndexIVF`, grabbing its quantizer (a flat index that
//!     stores the centroids), and reconstructing each centroid in turn.
//!
//! The result is `Centroids { dim, vectors: [nlist][dim] f32 }`.

use anyhow::{Context, Result, anyhow};
use faiss_sys::{
    FaissIndex, faiss_IndexIVF_cast, faiss_IndexIVF_nlist, faiss_IndexIVF_quantizer,
    faiss_Index_d, faiss_Index_free, faiss_Index_reconstruct, faiss_get_last_error,
    faiss_read_index_fname,
};
use std::ffi::{CStr, CString};
use std::io::Write;

use crate::vpack::Slice;

pub struct Centroids {
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
}

/// Extract codeData bytes from the VPack `TrainedData` value at the sentinel
/// key. Handles both the binary (0xc0..=0xc7) and array-of-bytes encodings.
pub fn extract_code_data(trained_value: &[u8]) -> Result<Vec<u8>> {
    let v = Slice::new(trained_value);
    let code = v
        .get("codeData")
        .ok_or_else(|| anyhow!("trained value has no `codeData` field"))?;
    if let Some(b) = code.as_binary() {
        return Ok(b.to_vec());
    }
    if code.is_array() {
        return code
            .array_iter()?
            .map(|s| s.as_byte())
            .collect::<Option<Vec<u8>>>()
            .ok_or_else(|| anyhow!("non-byte element in codeData array"));
    }
    Err(anyhow!(
        "codeData has unexpected VPack type 0x{:02x}",
        code.type_byte()
    ))
}

pub fn read_centroids(faiss_bytes: &[u8]) -> Result<Centroids> {
    // The C API only takes a filename. Write to a tempfile in this process.
    let mut tf = tempfile::NamedTempFile::new().context("creating temp file")?;
    tf.write_all(faiss_bytes).context("writing faiss bytes")?;
    tf.flush().context("flushing faiss bytes")?;
    let path_cstr = CString::new(tf.path().as_os_str().as_encoded_bytes())
        .context("temp file path contained nul byte")?;

    let index = OwnedFaissIndex::read(&path_cstr)?;
    let ivf = unsafe { faiss_IndexIVF_cast(index.as_ptr()) };
    if ivf.is_null() {
        return Err(anyhow!("FAISS index is not an IndexIVF — got something else"));
    }
    let nlist = unsafe { faiss_IndexIVF_nlist(ivf) } as usize;
    let dim = unsafe { faiss_Index_d(index.as_ptr()) } as usize;
    let quantizer = unsafe { faiss_IndexIVF_quantizer(ivf) };
    if quantizer.is_null() {
        return Err(anyhow!("IndexIVF has no quantizer"));
    }

    let mut vectors = Vec::with_capacity(nlist);
    let mut buf = vec![0f32; dim];
    for i in 0..nlist {
        let rc = unsafe { faiss_Index_reconstruct(quantizer, i as i64, buf.as_mut_ptr()) };
        if rc != 0 {
            return Err(anyhow!(
                "faiss_Index_reconstruct({i}) returned {rc}: {}",
                last_faiss_error()
            ));
        }
        vectors.push(buf.clone());
    }
    Ok(Centroids { dim, vectors })
}

/// Owns a `FaissIndex` pointer and frees it on drop.
struct OwnedFaissIndex(*mut FaissIndex);

impl OwnedFaissIndex {
    fn read(path: &CStr) -> Result<Self> {
        let mut idx: *mut FaissIndex = std::ptr::null_mut();
        let rc = unsafe { faiss_read_index_fname(path.as_ptr(), 0, &mut idx) };
        if rc != 0 {
            return Err(anyhow!(
                "faiss_read_index_fname returned {rc}: {}",
                last_faiss_error()
            ));
        }
        Ok(OwnedFaissIndex(idx))
    }

    fn as_ptr(&self) -> *mut FaissIndex {
        self.0
    }
}

impl Drop for OwnedFaissIndex {
    fn drop(&mut self) {
        unsafe { faiss_Index_free(self.0) };
    }
}

fn last_faiss_error() -> String {
    let p = unsafe { faiss_get_last_error() };
    if p.is_null() {
        return "(no message)".into();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}
