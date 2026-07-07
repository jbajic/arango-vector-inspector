//! IVF search over the VectorIndex CF.
//!
//! ArangoDB stores document vectors in RocksDB under:
//!   key = objectId(8 BE) | listNumber(8 BE) | revisionId(8 BE)   (24 bytes)
//!   value = raw little-endian f32 array  (dim * 4 bytes)
//!
//! A search proceeds in two phases:
//!  1. Brute-force scan of the centroid matrix to find the `nProbe` nearest
//!     centroid lists (using the index metric).
//!  2. Sequential scan of those lists in RocksDB, computing a score for every
//!     stored vector, then returning the top-K results.

use anyhow::{Context, Result, anyhow};
use rocksdb::{DB, IteratorMode};

// ---- Metric -----------------------------------------------------------------

/// Distance / similarity metric.  Drives how scores are computed and which
/// direction "better" means. Client-side only: the serve process ships raw
/// vectors and never scores.
#[cfg(feature = "ui")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean L2 distance — smaller is better.
    L2,
    /// Cosine similarity — larger is better (assumes normalized vectors).
    Cosine,
    /// Inner-product / dot-product — larger is better.
    Ip,
}

#[cfg(feature = "ui")]
impl Metric {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "cosine" => Self::Cosine,
            "ip" | "dot" | "inner_product" | "inner-product" | "innerproduct" => Self::Ip,
            _ => Self::L2,
        }
    }

    /// Human-readable label for a score value.
    pub fn score_label(self) -> &'static str {
        match self {
            Self::L2 => "dist",
            Self::Cosine | Self::Ip => "sim",
        }
    }

    /// Compute the internal score between `a` and `b`.
    /// Always returns a value where **lower means more similar**, regardless
    /// of metric:
    ///   L2      → L2 distance          (natural ascending)
    ///   Cosine  → −dot_product          (negate so ascending = most-similar-first)
    ///   IP      → −dot_product
    pub fn raw_score(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2 => a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            Self::Cosine | Self::Ip => -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
        }
    }

    /// Convert the internal raw score to a display value.
    /// L2 → same value; cosine/IP → negate back to the natural similarity.
    pub fn display_value(self, raw: f32) -> f32 {
        match self {
            Self::L2 => raw,
            Self::Cosine | Self::Ip => -raw,
        }
    }
}

// ---- Public types -----------------------------------------------------------

#[cfg(feature = "ui")]
pub struct SearchHit {
    pub list_id: u64,
    pub doc_id: u64,
    /// Raw internal score (lower = better).  Use `Metric::display_value` to
    /// convert for presentation.
    pub score: f32,
    pub vector: Vec<f32>,
}

// ---- Parse ------------------------------------------------------------------

/// Parse a float array string into `Vec<f32>`.
/// Accepts plain comma/whitespace-separated values as well as JSON-style
/// `[0.1, 0.2, ...]` notation — square brackets are stripped.
#[cfg(feature = "ui")]
pub fn parse_query_vector(text: &str) -> Result<Vec<f32>> {
    let stripped = text.trim().trim_start_matches('[').trim_end_matches(']');
    stripped
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .enumerate()
        .map(|(i, s)| {
            s.trim()
                .parse::<f32>()
                .with_context(|| format!("element {i}: {:?} is not a valid float", s.trim()))
        })
        .collect()
}

// ---- Search -----------------------------------------------------------------

/// Find the `n_probe` centroid lists that are most relevant to `query`.
/// Sorted best-first (lowest raw score = most similar).
#[cfg(feature = "ui")]
pub fn find_closest_centroids(
    query: &[f32],
    centroids: &[Vec<f32>],
    n_probe: usize,
    metric: Metric,
) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, metric.raw_score(query, c)))
        .collect();
    scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(n_probe);
    scores
}

/// Score `raw` document vectors against `query` and return the `top_k` best
/// hits (lowest raw score first). Pure — the DB read happens elsewhere.
#[cfg(feature = "ui")]
pub fn rank_hits(
    query: &[f32],
    raw: Vec<(u64, u64, Vec<f32>)>,
    top_k: usize,
    metric: Metric,
) -> Vec<SearchHit> {
    let mut hits: Vec<SearchHit> = raw
        .into_iter()
        .map(|(list_id, doc_id, vector)| {
            let score = metric.raw_score(query, &vector);
            SearchHit {
                list_id,
                doc_id,
                score,
                vector,
            }
        })
        .collect();

    hits.sort_unstable_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(top_k);
    hits
}

// ---- DB read ----------------------------------------------------------------

/// Read all document vectors stored in the given centroid lists of an already
/// open DB. Returns `(list_id, doc_id, vector)` for each entry whose value
/// length matches `dim * 4` bytes (raw f32 little-endian).
pub(crate) fn read_vectors_open(
    db: &DB,
    object_id: u64,
    list_ids: &[u64],
    dim: usize,
) -> Result<Vec<(u64, u64, Vec<f32>)>> {
    let cf = db
        .cf_handle("VectorIndex")
        .ok_or_else(|| anyhow!("VectorIndex CF not found"))?;

    let expected_len = dim * std::mem::size_of::<f32>();
    let mut out = Vec::new();

    let mut sorted_ids = list_ids.to_vec();
    sorted_ids.sort_unstable();

    for &list_id in &sorted_ids {
        let mut prefix = [0u8; 16];
        prefix[0..8].copy_from_slice(&object_id.to_be_bytes());
        prefix[8..16].copy_from_slice(&list_id.to_be_bytes());

        let iter = db.iterator_cf(
            &cf,
            IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );
        for item in iter {
            let (key, value) = item.context("iterating VectorIndex")?;
            if key.len() < 16 || key[0..16] != prefix {
                break;
            }
            if key.len() != 24 || value.len() != expected_len {
                continue;
            }
            let doc_id = u64::from_be_bytes(key[16..24].try_into().unwrap());
            let vector: Vec<f32> = value
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            out.push((list_id, doc_id, vector));
        }
    }

    Ok(out)
}
