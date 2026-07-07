# arango-vector-inspector

Inspect and visualize ArangoDB RocksDB **vector indexes** — centroid layout,
list population, distribution stats, and live IVF search — either offline from
a data directory or **remotely over SSH**, with a native local UI.

It reads the RocksDB data directory directly, so no query language and no
patched server are involved. If `arangod` is running, the tool opens the DB as
a RocksDB *secondary* instance and tails the WAL, so live (still-unflushed)
data shows up too.

---

## The one idea to understand first

The vector data usually lives on a server (e.g. a GCP box) with **no display**.
You still want the real graphical UI — not a text dump, not VNC.

The tool solves this by splitting cleanly in two:

- **Reading RocksDB** happens *on the machine with the data*.
- **Rendering the UI** happens *on your laptop*.

They talk over your normal `ssh` connection. **Only structured data crosses the
wire — never pixels.** There is no server daemon, no open port, and no firewall
change: the tool just runs `ssh host -- arango-vector-inspector --serve …` and
speaks a small binary protocol over that pipe.

```
   your laptop (has a display)                 remote host (has the data)
  ┌───────────────────────────┐               ┌─────────────────────────┐
  │  --ui   (egui window)      │   ssh stdio   │  --serve                │
  │  FAISS decode, t-SNE,      │◄─────────────►│  opens RocksDB once,    │
  │  Voronoi, scoring          │  bincode      │  answers scan + vector  │
  │                            │   frames      │  read requests          │
  └───────────────────────────┘               └─────────────────────────┘
```

---

## Build

Requires Rust 1.85+ (`edition = "2024"`).

### Full build (local UI) — default

```bash
cargo build --release
```

This includes the GUI (egui/OpenGL) and the FAISS-based centroid decoder. The
first build compiles a bundled `libfaiss_c.so`, so it needs `cmake`, a C++
compiler, BLAS, and OpenMP available once.

### Lean build (headless remote reader)

For the machine that only *serves* data over SSH, skip the GUI and FAISS
entirely:

```bash
cargo build --release --no-default-features --features serve
```

This variant depends on RocksDB + serde only. No OpenGL, no FAISS — it compiles
quickly and runs on a bare headless server. It can do everything the remote end
needs (`--serve`) and the headless text/JSON reports.

---

## Usage

### Local — inspect a data directory on this machine

```bash
arango-vector-inspector --db /var/lib/arangodb3/engine-rocksdb
```

The `--db` path is the `engine-rocksdb/` subdirectory of the ArangoDB data
directory.

### Remote — data on another machine, UI on yours

```bash
# 1. Put the binary on the remote host once (build the lean variant there,
#    or scp a matching release build):
#      cargo build --release --no-default-features --features serve

# 2. From your laptop:
arango-vector-inspector --ui \
  --remote you@gcp-host \
  --db /var/lib/arangodb3/engine-rocksdb
```

Here `--db` is the path **on the remote host**. Authentication is just your
normal SSH keys / `~/.ssh/config` — the tool never handles credentials. The
window opens locally; searches and scans run against the live remote DB.

The remote flag works for headless output too, e.g.
`--remote you@gcp-host --format json`.

### Open the interactive UI

Add `--ui` to any of the above. Tabs:

| Tab | What it shows |
|-----|----------------|
| **Overview** | Key/value summary: dimension, metric, nLists, totals, distribution stats |
| **Voronoi** | Centroids projected to 2D as a Voronoi diagram; pan/zoom, color by count, click a cell for details |
| **Histogram** | Vectors-per-centroid distribution |
| **Area vs Count** | Cell area vs. its vector count (spots over/under-full centroids) |
| **Search** | Enter a query vector → live IVF search (probe *nProbe* lists, top *K* hits). Works remotely. |

### Options

| Flag | Description |
|------|-------------|
| `--db <PATH>` | RocksDB data directory. With `--remote`, this is the path on the remote host. **(required)** |
| `--remote <SSH-DEST>` | Read the DB on another machine over SSH (e.g. `user@host`). UI still runs locally. |
| `--remote-bin <PATH>` | Name/path of the inspector binary on the remote host (default: `arango-vector-inspector` on `PATH`). |
| `--index-id <ID>` | Limit to a single index by objectId (decimal). |
| `--ui` | Open the interactive visualization window. |
| `--projection pca\|tsne` | 2D embedding for the UI. `tsne` (default) is slower but avoids the "everything on a disk" artifact; `pca` is fast. |
| `--centroids` | Print a per-index centroid summary (count, dim, first-row preview, norm range). |
| `--format text\|json` | Output format for the non-UI report (default: `text`). |

### Example — text output

```
DB: /var/lib/arangodb3/engine-rocksdb
Vector indexes found: 1

Index myCollection/vec_idx (objectId 250 / 0x00000000000000fa)
--------------------------------------------------------------
  dimension:          1024
  metric:             l2
  configured nLists:  400
  trained:            yes
  total vectors:      10000
  non-empty centroids: 400
  max list# observed: 399
  dead centroids:     0 of 400

  Distribution (vectors per centroid, including empties):
    min:      6
    max:      54
    mean:     25.00
    median:   27
    p95:      38
    p99:      44
    stddev:   8.55

    Histogram:
      0                0
      1-10             8
      11-100         392
      101-1k           0
      1k-10k           0
      10k+             0
```

---

## How it works (for the curious / for agents)

The entire surface that touches RocksDB is **two operations**, captured by the
`DataSource` trait in `src/source.rs`:

1. `scan()` → per-index metadata, per-list vector counts, and the trained FAISS
   blob.
2. `read_vectors_for_lists(object_id, list_ids, dim)` → the raw vectors in the
   probed centroid lists (the DB half of a search).

Everything else — FAISS centroid decode, PCA/t-SNE projection, Voronoi cells,
and search scoring — is **pure compute that runs on the UI side**, on tiny
payloads (a few MB of centroids, not the whole dataset).

Two implementations of the trait:

- `LocalSource` — opens the RocksDB directory here and calls the readers
  directly. The DB is opened once and reused; a secondary instance is caught up
  to the primary before each read so live data is visible.
- `RemoteSource` — spawns `ssh <host> -- <bin> --serve --db <path>`, keeps the
  pipe open, and RPCs over it.

The wire protocol (`src/proto.rs`) is length-prefixed **bincode** frames over
stdio (`u32` length + payload). The server sends a version `Hello` on connect;
a mismatch aborts with a clear message, so keep both ends built from the same
source. Because stdout carries the frame stream, the serve process writes **all
diagnostics to stderr** (which SSH forwards back to you).

Source map:

| File | Responsibility |
|------|----------------|
| `src/scan.rs` | Open RocksDB (read-only or secondary+WAL); scan the VectorIndex and definitions CFs. |
| `src/search.rs` | IVF search: `read_vectors_open` (DB read) + pure scoring helpers. |
| `src/centroids.rs` | Decode the FAISS IVF blob into a centroid matrix *(ui feature)*. |
| `src/projection.rs` | PCA / t-SNE embedding of centroids to 2D *(ui feature)*. |
| `src/vpack.rs` | Minimal VelocyPack reader for index definitions. |
| `src/source.rs` | `DataSource` trait, `LocalSource`, `RemoteSource`, and the `--serve` loop. |
| `src/proto.rs` | Client/server request/response types and framing. |
| `src/ui.rs` | egui viewer *(ui feature)*. |
| `src/main.rs` | CLI, wiring, text/JSON reports. |

---

## Limitations

- **Single-server only.** Cluster shards live in separate RocksDB instances
  under each DBserver's data directory; point the tool at each one separately.
- **Remote requires the binary on the far end** and SSH access. Build the lean
  `serve` variant there, or copy a matching release build.
- **Search is synchronous** in this version: a slow link briefly blocks the UI
  during a query round-trip. (An already-open remote DB reading only *nProbe*
  lists is typically sub-second.)
- The 2D projection is lossy — relative neighborhoods are roughly preserved,
  absolute distances are not.
