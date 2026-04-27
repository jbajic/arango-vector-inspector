mod centroids;
mod projection;
mod scan;
mod stats;
mod ui;
mod vpack;

use anyhow::{Context, Result};
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

    /// If set, decode each trained index via FAISS and print centroid summary
    /// (count, dim, first-row preview, norm range).
    #[arg(long)]
    centroids: bool,

    /// Open a window visualizing the centroids of one index as a 2D Voronoi
    /// diagram. If --index-id is unset, the first vector index found is shown.
    #[arg(long)]
    ui: bool,

    /// Projection method used to embed centroids into 2D for the UI.
    /// `tsne` is non-linear (slower, prettier); `pca` is fast but tends to
    /// render spherical centroid clouds as disks.
    #[arg(long, value_enum, default_value_t = Projection::Tsne)]
    projection: Projection,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Projection {
    Pca,
    Tsne,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Format {
    Text,
    Json,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let result = scan::scan(&args.db)?;
    if args.ui {
        return run_ui(&args, &result);
    }
    if args.centroids {
        print_centroids_info(&args, &result)?;
    }
    match args.format {
        Format::Text => print_text(&args, &result),
        Format::Json => print_json(&args, &result)?,
    }
    Ok(())
}

fn run_ui(args: &Args, r: &ScanResult) -> Result<()> {
    let candidates: Vec<(u64, &PerIndexStats)> = filtered(args, r)
        .filter(|(_, s)| s.trained_value.is_some())
        .collect();
    if candidates.is_empty() {
        return Err(anyhow::anyhow!("no vector indexes with trained data found"));
    }

    let mut views: Vec<ui::IndexView> = Vec::with_capacity(candidates.len());
    for (oid, stats) in candidates {
        eprintln!(
            "Loading index {oid}{}...",
            if matches!(args.projection, Projection::Tsne) {
                " (t-SNE may take a few seconds)"
            } else {
                ""
            }
        );
        let view = build_index_view(oid, stats, r.metadata.get(&oid), args)
            .with_context(|| format!("preparing index {oid}"))?;
        views.push(view);
    }

    ui::run(ui::ViewerData { indexes: views })
}

fn build_index_view(
    oid: u64,
    stats: &PerIndexStats,
    meta: Option<&IndexMeta>,
    args: &Args,
) -> Result<ui::IndexView> {
    let value = stats
        .trained_value
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("index {oid} has no trained data"))?;
    let bytes =
        centroids::extract_code_data(value).context("extracting codeData from trained value")?;
    let c = centroids::read_centroids(&bytes).context("decoding FAISS index")?;
    let points = match args.projection {
        Projection::Pca => projection::pca_2d(&c.vectors).context("PCA projection")?,
        Projection::Tsne => projection::tsne_2d(&c.vectors).context("t-SNE projection")?,
    };

    let centroid_views: Vec<ui::CentroidView> = points
        .iter()
        .enumerate()
        .map(|(i, &p)| ui::CentroidView {
            point: p,
            count: stats.lists.get(&(i as u64)).copied().unwrap_or(0),
        })
        .collect();

    let title = match meta {
        Some(m) => format!("{}/{}", m.collection_name, m.name),
        None => format!("objectId {oid}"),
    };
    let overview = build_overview_kvs(oid, stats, meta, c.vectors.len(), c.dim);

    let (cells, cell_areas, bbox) = ui::build_cells(&points);
    let max_count = centroid_views.iter().map(|c| c.count).max().unwrap_or(1);
    let mut sorted_counts: Vec<(u64, usize)> = centroid_views
        .iter()
        .enumerate()
        .map(|(i, c)| (c.count, i))
        .collect();
    sorted_counts.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

    Ok(ui::IndexView {
        title,
        overview,
        centroids: centroid_views,
        high_dim: c.vectors,
        cells,
        cell_areas,
        bbox,
        max_count,
        sorted_counts,
    })
}

fn build_overview_kvs(
    oid: u64,
    s: &PerIndexStats,
    meta: Option<&IndexMeta>,
    nlist: usize,
    dim: usize,
) -> Vec<(String, String)> {
    let mut kvs = Vec::new();
    if let Some(m) = meta {
        kvs.push(("collection".into(), m.collection_name.clone()));
        kvs.push(("index name".into(), m.name.clone()));
    }
    kvs.push(("objectId".into(), format!("{oid}  (0x{oid:016x})")));
    if let Some(m) = meta {
        if let Some(d) = m.dimension {
            kvs.push(("dimension".into(), d.to_string()));
        }
        if let Some(met) = &m.metric {
            kvs.push(("metric".into(), met.clone()));
        }
    }
    kvs.push(("dim (from FAISS)".into(), dim.to_string()));
    kvs.push(("centroids (nLists)".into(), nlist.to_string()));
    kvs.push((
        "trained".into(),
        if s.trained { "yes" } else { "no" }.to_string(),
    ));
    kvs.push(("total vectors".into(), s.total_vectors().to_string()));
    let dead = (nlist as u64).saturating_sub(s.non_empty_lists());
    kvs.push(("dead centroids".into(), format!("{dead} of {nlist}")));

    let counts: Vec<u64> = (0..nlist as u64)
        .map(|i| s.lists.get(&i).copied().unwrap_or(0))
        .collect();
    if let Some(dist) = stats::distribution(&counts) {
        kvs.push(("count min".into(), dist.min.to_string()));
        kvs.push(("count max".into(), dist.max.to_string()));
        kvs.push(("count mean".into(), format!("{:.2}", dist.mean)));
        kvs.push(("count median".into(), dist.median.to_string()));
        kvs.push(("count p95".into(), dist.p95.to_string()));
        kvs.push(("count p99".into(), dist.p99.to_string()));
        kvs.push(("count stddev".into(), format!("{:.2}", dist.stddev)));
    }
    kvs
}

fn print_centroids_info(args: &Args, r: &ScanResult) -> Result<()> {
    for (oid, s) in filtered(args, r) {
        let Some(value) = &s.trained_value else {
            println!("index {oid}: no trained data");
            continue;
        };
        let bytes = centroids::extract_code_data(value)
            .with_context(|| format!("index {oid}: extracting codeData"))?;
        let c = centroids::read_centroids(&bytes)
            .with_context(|| format!("index {oid}: parsing FAISS index"))?;
        let nlist = c.vectors.len();
        let norms: Vec<f32> = c
            .vectors
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();
        let nmin = norms.iter().cloned().fold(f32::INFINITY, f32::min);
        let nmax = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "index {oid}: nlist={nlist}, dim={}, norm range=[{:.3}, {:.3}]",
            c.dim, nmin, nmax
        );
        if let Some(first) = c.vectors.first() {
            let preview: Vec<String> = first.iter().take(8).map(|x| format!("{x:.4}")).collect();
            println!("  first centroid first 8 dims: [{}]", preview.join(", "));
        }
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
    dead_centroids: Option<u64>,
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
    let dead = n_lists.map(|n| n.saturating_sub(s.non_empty_lists()));
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
        dead_centroids: dead,
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
    println!("  non-empty centroids: {}", s.non_empty_lists());
    if let Some(max_list) = s.max_list_number() {
        println!("  max list# observed: {}", max_list);
    }
    if let Some(n) = resolved_n_lists(s, meta) {
        let dead = n.saturating_sub(s.non_empty_lists());
        println!("  dead centroids:     {} of {}", dead, n);
    } else {
        println!("  dead centroids:     (unknown — configured nLists not available)");
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
