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
use rocksdb::IteratorMode;

// ---- Metric -----------------------------------------------------------------

/// Distance / similarity metric.  Drives how scores are computed and which
/// direction "better" means.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean L2 distance — smaller is better.
    L2,
    /// Cosine similarity — larger is better (assumes normalized vectors).
    Cosine,
    /// Inner-product / dot-product — larger is better.
    Ip,
}

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

    /// True when results should be presented in ascending score order.
    /// (L2: ascending distance; cosine/IP: descending similarity, i.e. the
    /// internal score is negated so the sort is still ascending.)
    pub fn ascending(self) -> bool {
        true // internal scores are always "lower = better"
    }

    /// Compute the internal score between `a` and `b`.
    /// Always returns a value where **lower means more similar**, regardless
    /// of metric:
    ///   L2      → L2 distance          (natural ascending)
    ///   Cosine  → −dot_product          (negate so ascending = most-similar-first)
    ///   IP      → −dot_product
    pub fn raw_score(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2 => a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt(),
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

pub struct SearchHit {
    pub list_id: u64,
    pub doc_id: u64,
    /// Raw internal score (lower = better).  Use `Metric::display_value` to
    /// convert for presentation.
    pub score: f32,
    pub vector: Vec<f32>,
}

pub struct SearchResult {
    pub metric: Metric,
    /// `(centroid_index, raw_score)` sorted best-first.
    pub closest_centroids: Vec<(usize, f32)>,
    /// Top-K document hits sorted best-first.
    pub hits: Vec<SearchHit>,
}

// ---- Parse ------------------------------------------------------------------

/// Parse a float array string into `Vec<f32>`.
/// Accepts plain comma/whitespace-separated values as well as JSON-style
/// `[0.1, 0.2, ...]` notation — square brackets are stripped.
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
    scores.sort_unstable_by(|a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    scores.truncate(n_probe);
    scores
}

/// Full IVF search: probe `n_probe` centroid lists and return the `top_k`
/// best-matching document vectors.
pub fn run_search(
    db_path: &str,
    object_id: u64,
    query: &[f32],
    centroids: &[Vec<f32>],
    n_probe: usize,
    top_k: usize,
    dim: usize,
    metric: Metric,
) -> Result<SearchResult> {
    let closest = find_closest_centroids(query, centroids, n_probe, metric);

    if dim == 0 || closest.is_empty() {
        return Ok(SearchResult { metric, closest_centroids: closest, hits: Vec::new() });
    }

    let list_ids: Vec<u64> = closest.iter().map(|(i, _)| *i as u64).collect();
    let raw = read_vectors_for_lists(db_path, object_id, &list_ids, dim)
        .context("reading vectors from VectorIndex CF")?;

    let mut hits: Vec<SearchHit> = raw
        .into_iter()
        .map(|(list_id, doc_id, vector)| {
            let score = metric.raw_score(query, &vector);
            SearchHit { list_id, doc_id, score, vector }
        })
        .collect();

    hits.sort_unstable_by(|a, b| {
        a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(top_k);

    Ok(SearchResult { metric, closest_centroids: closest, hits })
}

// ---- DB read ----------------------------------------------------------------

/// Read all document vectors stored in the given centroid lists.
/// Returns `(list_id, doc_id, vector)` for each entry whose value length
/// matches `dim * 4` bytes (raw f32 little-endian).
fn read_vectors_for_lists(
    db_path: &str,
    object_id: u64,
    list_ids: &[u64],
    dim: usize,
) -> Result<Vec<(u64, u64, Vec<f32>)>> {
    let opened = crate::scan::open_for_reading(db_path)
        .context("opening DB for vector read")?;
    let db = &opened.db;

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
