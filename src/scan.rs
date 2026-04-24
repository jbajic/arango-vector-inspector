//! RocksDB scans — VectorIndex CF (counts/trained markers) and definitions CF
//! (index metadata via VPack).

use anyhow::{Context, Result, anyhow};
use rocksdb::{ColumnFamilyDescriptor, DB, IteratorMode, Options};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::vpack::Slice;

/// Per-vector-index accumulated stats from the VectorIndex CF scan.
#[derive(Default, Debug)]
pub struct PerIndexStats {
    /// Sentinel (objectId, UINT64_MAX) present => index has trained data.
    pub trained: bool,
    /// listNumber -> number of document entries in that list.
    pub lists: BTreeMap<u64, u64>,
    /// Keys we could not classify (unexpected length, non-fatal).
    pub anomalies: u64,
}

impl PerIndexStats {
    pub fn total_vectors(&self) -> u64 {
        self.lists.values().copied().sum()
    }
    pub fn non_empty_lists(&self) -> u64 {
        self.lists.values().filter(|c| **c > 0).count() as u64
    }
    /// Largest listNumber observed (exclusive of the UINT64_MAX sentinel).
    pub fn max_list_number(&self) -> Option<u64> {
        self.lists.keys().next_back().copied()
    }
}

/// Index metadata recovered from the definitions CF.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct IndexMeta {
    pub object_id: u64,
    pub name: String,
    pub collection_name: String,
    pub database_id: u64,
    pub dimension: Option<u64>,
    pub metric: Option<String>,
    /// Fixed nLists if stored as a plain integer. None if scaling spec or absent.
    pub n_lists: Option<u64>,
}

pub struct ScanResult {
    pub indexes: HashMap<u64, PerIndexStats>,
    pub metadata: HashMap<u64, IndexMeta>,
    /// Number of keys in the VectorIndex CF that had an unexpected shape.
    pub anomalies: u64,
}

/// CFs that ArangoDB opens with RocksDBVPackComparator (name must match).
/// We never iterate them, so bytewise ordering is fine for read-only access.
const VPACK_COMPARATOR_CFS: &[&str] = &["VPackIndex", "MdiPrefixed"];

fn make_cf_descriptors(cf_names: &[String]) -> Vec<ColumnFamilyDescriptor> {
    cf_names
        .iter()
        .map(|n| {
            let mut opts = Options::default();
            if VPACK_COMPARATOR_CFS.contains(&n.as_str()) {
                opts.set_comparator("RocksDBVPackComparator", Box::new(|a, b| a.cmp(b)));
            }
            ColumnFamilyDescriptor::new(n, opts)
        })
        .collect()
}

/// Opens the RocksDB instance. ArangoDB stores its WAL in a `journals/`
/// subdirectory, so if that directory exists we open as a secondary instance
/// (which tails the WAL) rather than plain read-only (which only sees flushed
/// SST files). A temp directory is used for the secondary's own metadata and
/// is cleaned up on drop.
fn open_db(db_path: &str, cf_names: &[String]) -> Result<DB> {
    let journals = Path::new(db_path).join("journals");
    if journals.exists() {
        let secondary_path = tempfile::tempdir()
            .context("failed to create temp dir for secondary instance")?;
        let mut db_opts = Options::default();
        db_opts.create_if_missing(false);
        db_opts.set_wal_dir(&journals);
        let db = DB::open_cf_descriptors_as_secondary(
            &db_opts,
            db_path,
            secondary_path.path().to_str().unwrap(),
            make_cf_descriptors(cf_names),
        )
        .with_context(|| format!("open_cf_descriptors_as_secondary on {}", db_path))?;
        db.try_catch_up_with_primary()
            .context("try_catch_up_with_primary failed")?;
        // Keep tempdir alive until DB is dropped by leaking it; the OS will
        // clean it up when the process exits.
        std::mem::forget(secondary_path);
        return Ok(db);
    }

    let mut db_opts = Options::default();
    db_opts.create_if_missing(false);
    DB::open_cf_descriptors_read_only(&db_opts, db_path, make_cf_descriptors(cf_names), false)
        .with_context(|| format!("open_cf_descriptors_read_only on {}", db_path))
}

pub fn scan(db_path: &str) -> Result<ScanResult> {
    let cf_names = DB::list_cf(&Options::default(), db_path)
        .with_context(|| format!("list_cf on {}", db_path))?;

    let db = open_db(db_path, &cf_names)?;

    let indexes = scan_vector_cf(&db)?;
    let mut result = ScanResult {
        anomalies: indexes.values().map(|s| s.anomalies).sum(),
        indexes,
        metadata: HashMap::new(),
    };
    result.metadata = scan_definitions_cf(&db).unwrap_or_else(|e| {
        eprintln!("warning: failed to scan definitions CF: {e:#}");
        HashMap::new()
    });
    Ok(result)
}

fn scan_vector_cf(db: &DB) -> Result<HashMap<u64, PerIndexStats>> {
    let cf = db
        .cf_handle("VectorIndex")
        .ok_or_else(|| anyhow!("VectorIndex column family not found"))?;

    let mut by_id: HashMap<u64, PerIndexStats> = HashMap::new();
    let iter = db.iterator_cf(&cf, IteratorMode::Start);
    for item in iter {
        let (key, _value) = item?;
        if key.len() < 16 {
            // malformed — bucket under objectId 0 as an anomaly
            by_id.entry(0).or_default().anomalies += 1;
            continue;
        }
        let object_id = u64::from_be_bytes(key[0..8].try_into().unwrap());
        let list_number = u64::from_be_bytes(key[8..16].try_into().unwrap());
        let stats = by_id.entry(object_id).or_default();

        if list_number == u64::MAX {
            stats.trained = true;
            continue;
        }
        match key.len() {
            16 => {
                stats.lists.entry(list_number).or_insert(0);
            }
            24 => {
                *stats.lists.entry(list_number).or_insert(0) += 1;
            }
            _ => {
                stats.anomalies += 1;
            }
        }
    }
    Ok(by_id)
}

// ---- definitions CF scan ------------------------------------------------

fn scan_definitions_cf(db: &DB) -> Result<HashMap<u64, IndexMeta>> {
    let cf = db
        .cf_handle("default")
        .ok_or_else(|| anyhow!("default (definitions) column family not found"))?;

    let mut out = HashMap::new();
    let iter = db.iterator_cf(&cf, IteratorMode::Start);
    for item in iter {
        let (key, value) = item?;
        // Collection entries in the definitions CF are keyed:
        //   '1' (0x31)  |  databaseId(8 BE)  |  collectionId(8 BE)   => 17 bytes
        if key.len() != 17 || key[0] != b'1' {
            continue;
        }
        let database_id = u64::from_be_bytes(key[1..9].try_into().unwrap());
        let coll = Slice::new(&value);
        if !coll.is_object() {
            continue;
        }
        let collection_name = coll
            .get("name")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let Some(indexes) = coll.get("indexes") else {
            continue;
        };
        if !indexes.is_array() {
            continue;
        }
        let Ok(iter) = indexes.array_iter() else {
            continue;
        };
        for idx_slice in iter {
            let Some(type_s) = idx_slice.get("type").and_then(|s| s.as_str()) else {
                continue;
            };
            if type_s != "vector" {
                continue;
            }
            let Some(object_id) = idx_slice
                .get("objectId")
                .and_then(|s| s.as_str())
                .and_then(|s| s.parse::<u64>().ok())
            else {
                continue;
            };
            let name = idx_slice
                .get("name")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();

            // Vector index params are nested under "params" in current
            // versions; older/newer schemas may put them at the top level.
            let params = idx_slice.get("params").unwrap_or(idx_slice);
            let dimension = params.get("dimension").and_then(|s| s.as_u64());
            let metric = params
                .get("metric")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let n_lists = params.get("nLists").and_then(|s| s.as_u64());

            out.insert(
                object_id,
                IndexMeta {
                    object_id,
                    name,
                    collection_name: collection_name.clone(),
                    database_id,
                    dimension,
                    metric,
                    n_lists,
                },
            );
        }
    }
    Ok(out)
}
