mod scan;
mod stats;
mod vpack;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use serde::Serialize;

use crate::scan::{IndexMeta, PerIndexStats, ScanResult};
use crate::stats::{Distribution, distribution};

#[derive(Parser, Debug)]
#[command(
    name = "arango-vector-inspect",
    about = "Offline inspector for ArangoDB RocksDB vector indexes, only for single server",
    version
)]
struct Args {
    /// Path to the RocksDB data directory (typically
    /// <datadir>/engine-rocksdb). arangod must not be running.
    #[arg(long)]
    db: String,

    /// Limit output to a single index by objectId (decimal).
    #[arg(long)]
    index_id: Option<u64>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Format {
    Text,
    Json,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let result = scan::scan(&args.db)?;
    match args.format {
        Format::Text => print_text(&args, &result),
        Format::Json => print_json(&args, &result)?,
    }
    Ok(())
}

// ---- JSON output ---------------------------------------------------------

#[derive(Serialize)]
struct JsonReport<'a> {
    db_path: &'a str,
    total_indexes: usize,
    anomalies: u64,
    indexes: Vec<JsonIndex>,
}

#[derive(Serialize)]
struct JsonIndex {
    object_id: u64,
    object_id_hex: String,
    name: Option<String>,
    collection: Option<String>,
    database_id: Option<u64>,
    dimension: Option<u64>,
    metric: Option<String>,
    configured_n_lists: Option<u64>,
    trained: bool,
    total_vectors: u64,
    non_empty_centroids: u64,
    max_list_number_observed: Option<u64>,
    empty_centroids: Option<u64>,
    distribution: Option<Distribution>,
}

fn print_json(args: &Args, r: &ScanResult) -> Result<()> {
    let indexes: Vec<JsonIndex> = filtered(args, r)
        .map(|(oid, s)| to_json_index(oid, s, r.metadata.get(&oid)))
        .collect();
    let report = JsonReport {
        db_path: &args.db,
        total_indexes: indexes.len(),
        anomalies: r.anomalies,
        indexes,
    };
    serde_json::to_writer_pretty(std::io::stdout(), &report)?;
    println!();
    Ok(())
}

fn to_json_index(oid: u64, s: &PerIndexStats, meta: Option<&IndexMeta>) -> JsonIndex {
    let counts = centroid_counts(s, meta);
    let dist = distribution(&counts);
    let n_lists = resolved_n_lists(s, meta);
    let empty = n_lists.map(|n| n.saturating_sub(s.non_empty_lists()));
    JsonIndex {
        object_id: oid,
        object_id_hex: format!("0x{:016x}", oid),
        name: meta.map(|m| m.name.clone()),
        collection: meta.map(|m| m.collection_name.clone()),
        database_id: meta.map(|m| m.database_id),
        dimension: meta.and_then(|m| m.dimension),
        metric: meta.and_then(|m| m.metric.clone()),
        configured_n_lists: n_lists,
        trained: s.trained,
        total_vectors: s.total_vectors(),
        non_empty_centroids: s.non_empty_lists(),
        max_list_number_observed: s.max_list_number(),
        empty_centroids: empty,
        distribution: dist,
    }
}

// ---- text output ---------------------------------------------------------

fn print_text(args: &Args, r: &ScanResult) {
    println!("DB: {}", args.db);
    let all: Vec<_> = filtered(args, r).collect();
    println!("Vector indexes found: {}", all.len());
    if r.anomalies > 0 {
        println!("Anomalies (malformed keys): {}", r.anomalies);
    }
    println!();

    for (oid, s) in all {
        let meta = r.metadata.get(&oid);
        print_index(oid, s, meta);
        println!();
    }
}

fn print_index(oid: u64, s: &PerIndexStats, meta: Option<&IndexMeta>) {
    let header = match meta {
        Some(m) => format!(
            "Index {}/{} (objectId {} / 0x{:016x})",
            m.collection_name, m.name, oid, oid
        ),
        None => format!(
            "Index objectId {} / 0x{:016x}  (no definition found — orphaned or definitions CF unreadable)",
            oid, oid
        ),
    };
    println!("{}", header);
    println!("{}", "-".repeat(header.len()));

    if let Some(m) = meta {
        if let Some(d) = m.dimension {
            println!("  dimension:          {}", d);
        }
        if let Some(ref met) = m.metric {
            println!("  metric:             {}", met);
        }
        if let Some(n) = m.n_lists.as_ref().map(|p| p.resolve(s.total_vectors())) {
            println!("  configured nLists:  {}", n);
        }
    }
    println!(
        "  trained:            {}",
        if s.trained { "yes" } else { "no" }
    );
    println!("  total vectors:      {}", s.total_vectors());
    println!("  non-empty centroids:{}", pad(s.non_empty_lists()));
    if let Some(max_list) = s.max_list_number() {
        println!("  max list# observed: {}", max_list);
    }
    if let Some(n) = resolved_n_lists(s, meta) {
        let empty = n.saturating_sub(s.non_empty_lists());
        println!("  empty centroids:    {} of {}", empty, n);
    } else {
        println!("  empty centroids:    (unknown — configured nLists not available)");
    }

    let counts = centroid_counts(s, meta);
    let Some(dist) = distribution(&counts) else {
        return;
    };
    println!();
    println!(
        "  Distribution (vectors per centroid{}):",
        if resolved_n_lists(s, meta).is_some() {
            ", including empties"
        } else {
            ", populated only"
        }
    );
    println!("    min:      {}", dist.min);
    println!("    max:      {}", dist.max);
    println!("    mean:     {:.2}", dist.mean);
    println!("    median:   {}", dist.median);
    println!("    p95:      {}", dist.p95);
    println!("    p99:      {}", dist.p99);
    println!("    stddev:   {:.2}", dist.stddev);
    println!();
    println!("    Histogram:");
    let col = dist
        .histogram
        .iter()
        .map(|b| b.label.len())
        .max()
        .unwrap_or(6);
    for b in &dist.histogram {
        println!("      {:width$}  {:>10}", b.label, b.count, width = col);
    }
}

fn pad<T: std::fmt::Display>(v: T) -> String {
    format!(" {}", v)
}

// ---- helpers -------------------------------------------------------------

fn filtered<'a>(
    args: &'a Args,
    r: &'a ScanResult,
) -> impl Iterator<Item = (u64, &'a PerIndexStats)> + 'a {
    let mut ids: Vec<u64> = r.indexes.keys().copied().collect();
    ids.sort_unstable();
    ids.into_iter()
        .filter(move |oid| args.index_id.map_or(true, |want| want == *oid))
        .map(move |oid| (oid, &r.indexes[&oid]))
}

fn resolved_n_lists(s: &PerIndexStats, meta: Option<&IndexMeta>) -> Option<u64> {
    meta?.n_lists.as_ref().map(|p| p.resolve(s.total_vectors()))
}

/// Build the per-centroid count vector. If we know the configured nLists we
/// fill in zeros for empty lists so distribution stats include them; otherwise
/// we only report populated lists.
fn centroid_counts(s: &PerIndexStats, meta: Option<&IndexMeta>) -> Vec<u64> {
    if let Some(n) = resolved_n_lists(s, meta) {
        let mut v = vec![0u64; n as usize];
        for (&list, &count) in &s.lists {
            if (list as usize) < v.len() {
                v[list as usize] = count;
            } else {
                v.push(count);
            }
        }
        v
    } else {
        s.lists.values().copied().collect()
    }
}
