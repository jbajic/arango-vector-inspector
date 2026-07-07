//! Data-source abstraction. The whole surface that touches RocksDB is two
//! operations; everything else (FAISS decode, projection, Voronoi, scoring) is
//! pure compute that runs wherever the UI runs.
//!
//!  * [`LocalSource`] reads a RocksDB directory on this machine.
//!  * [`RemoteSource`] drives a `--serve` process on another machine over an
//!    ssh stdio pipe — only structured data crosses the wire, no display.
//!
//! [`run_serve`] is the other end: it opens the DB once and answers framed
//! requests until the client disconnects.

use anyhow::{Context, Result, anyhow};
use std::io::BufReader;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use crate::proto::{self, PROTOCOL_VERSION, Request, Response};
use crate::scan::{OpenedDb, ScanResult};

/// A read-only view of a vector index store. Implementations may be local or
/// remote; callers never know which.
pub trait DataSource: Send + Sync {
    fn scan(&self) -> Result<ScanResult>;
    fn read_vectors_for_lists(
        &self,
        object_id: u64,
        list_ids: &[u64],
        dim: usize,
    ) -> Result<Vec<(u64, u64, Vec<f32>)>>;
}

// ---- Local ------------------------------------------------------------------

/// Reads a RocksDB data directory on the local filesystem. The DB is opened
/// once and reused across calls (a secondary instance is caught up to the
/// primary before each read so live WAL data is visible).
pub struct LocalSource {
    opened: OpenedDb,
}

impl LocalSource {
    pub fn open(db_path: &str) -> Result<Self> {
        let opened = crate::scan::open_for_reading(db_path)
            .with_context(|| format!("opening RocksDB at {db_path}"))?;
        Ok(Self { opened })
    }
}

impl DataSource for LocalSource {
    fn scan(&self) -> Result<ScanResult> {
        self.opened.catch_up()?;
        crate::scan::scan_open(&self.opened.db)
    }

    fn read_vectors_for_lists(
        &self,
        object_id: u64,
        list_ids: &[u64],
        dim: usize,
    ) -> Result<Vec<(u64, u64, Vec<f32>)>> {
        self.opened.catch_up()?;
        crate::search::read_vectors_open(&self.opened.db, object_id, list_ids, dim)
    }
}

// ---- Remote -----------------------------------------------------------------

/// Drives `ssh <host> <remote_bin> --serve --db <path>` and talks the framed
/// protocol over its stdin/stdout. The child's stderr is inherited, so remote
/// warnings and ssh diagnostics surface locally.
pub struct RemoteSource {
    conn: Mutex<Conn>,
}

struct Conn {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl RemoteSource {
    pub fn spawn(host: &str, db_path: &str, remote_bin: &str) -> Result<Self> {
        let mut child = Command::new("ssh")
            .arg(host)
            .arg("--")
            .arg(remote_bin)
            .arg("--serve")
            .arg("--db")
            .arg(db_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("spawning ssh — is ssh on PATH and the host reachable?")?;

        let stdin = child.stdin.take().expect("piped stdin");
        let mut stdout = BufReader::new(child.stdout.take().expect("piped stdout"));

        // The server greets with its protocol version before anything else.
        match proto::read_frame::<_, Response>(&mut stdout).context(
            "reading handshake — is arango-vector-inspector installed on the remote host?",
        )? {
            Response::Hello { version } if version == PROTOCOL_VERSION => {}
            Response::Hello { version } => {
                return Err(anyhow!(
                    "protocol mismatch: remote speaks v{version}, local expects v{PROTOCOL_VERSION} \
                     — rebuild the remote binary from the same source"
                ));
            }
            _ => return Err(anyhow!("remote did not send a handshake")),
        }

        Ok(Self {
            conn: Mutex::new(Conn {
                child,
                stdin,
                stdout,
            }),
        })
    }

    fn request(&self, req: &Request) -> Result<Response> {
        let mut conn = self.conn.lock().unwrap();
        proto::write_frame(&mut conn.stdin, req).context("sending request to remote")?;
        proto::read_frame(&mut conn.stdout).context("reading response from remote")
    }
}

impl DataSource for RemoteSource {
    fn scan(&self) -> Result<ScanResult> {
        match self.request(&Request::Scan)? {
            Response::Scan(s) => Ok(s),
            Response::Error(e) => Err(anyhow!("remote scan failed: {e}")),
            _ => Err(proto::unexpected("Scan")),
        }
    }

    fn read_vectors_for_lists(
        &self,
        object_id: u64,
        list_ids: &[u64],
        dim: usize,
    ) -> Result<Vec<(u64, u64, Vec<f32>)>> {
        let req = Request::ReadVectors {
            object_id,
            list_ids: list_ids.to_vec(),
            dim,
        };
        match self.request(&req)? {
            Response::Vectors(v) => Ok(v),
            Response::Error(e) => Err(anyhow!("remote vector read failed: {e}")),
            _ => Err(proto::unexpected("ReadVectors")),
        }
    }
}

impl Drop for RemoteSource {
    fn drop(&mut self) {
        // Closing stdin makes the serve loop hit EOF and exit; then reap it.
        if let Ok(mut conn) = self.conn.lock() {
            let _ = conn.child.wait();
        }
    }
}

// ---- Serve ------------------------------------------------------------------

/// The `--serve` end: open the DB once, greet the client, then answer framed
/// requests until stdin closes. Nothing but frames may go to stdout, so all
/// diagnostics use stderr.
pub fn run_serve(db_path: &str) -> Result<()> {
    let source = LocalSource::open(db_path)?;

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();

    proto::write_frame(
        &mut writer,
        &Response::Hello {
            version: PROTOCOL_VERSION,
        },
    )?;

    loop {
        let req: Request = match proto::read_frame(&mut reader) {
            Ok(req) => req,
            // EOF / broken pipe: the client went away — clean shutdown.
            Err(_) => return Ok(()),
        };

        let resp = match req {
            Request::Scan => match source.scan() {
                Ok(s) => Response::Scan(s),
                Err(e) => Response::Error(format!("{e:#}")),
            },
            Request::ReadVectors {
                object_id,
                list_ids,
                dim,
            } => match source.read_vectors_for_lists(object_id, &list_ids, dim) {
                Ok(v) => Response::Vectors(v),
                Err(e) => Response::Error(format!("{e:#}")),
            },
        };

        proto::write_frame(&mut writer, &resp)?;
    }
}
