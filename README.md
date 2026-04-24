# arango-vector-inspector

Offline inspector for ArangoDB RocksDB vector indexes (single-server only).

Reads the RocksDB data directory directly — no running `arangod` required for
offline use. If `arangod` is running, the tool automatically opens the DB as a
RocksDB secondary instance and tails the WAL so live (unflushed) data is
visible.

## Build

Requires Rust 1.85+ (`edition = "2024"`).

```bash
cargo build --release
```
## Usage

```
arango-vector-inspector --db <path-to-engine-rocksdb>
```

The `--db` path is the `engine-rocksdb/` subdirectory inside the ArangoDB data
directory (e.g. `/var/lib/arangodb3/engine-rocksdb`).

### Options

| Flag | Description |
|------|-------------|
| `--db <PATH>` | Path to the RocksDB data directory (required) |
| `--index-id <ID>` | Show only the index with this objectId (decimal) |
| `--format text\|json` | Output format (default: `text`) |

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
  empty centroids:    0 of 400

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

## Limitations

- Single-server only. Cluster shards live in separate RocksDB instances under
  each DBserver's data directory; run the tool against each one separately.
