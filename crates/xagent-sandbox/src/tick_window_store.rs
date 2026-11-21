//! SQLite persistence for the tick window: a writer that keeps the database
//! equal to the in-memory window and a read-only loader.

use std::collections::HashSet;
use std::fmt;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};

use crate::tick_window::{
    decode_frame, encode_frame, Pin, PinKind, TickWindow, TickWindowConfig, TickWindowError,
    WindowFrame,
};

const WRITER_PRAGMA: &str =
    "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;";
const BUSY_TIMEOUT_MS: u64 = 5_000;
const KIND_REPLACEMENT: &str = "replacement";
const KIND_SEASON: &str = "season";

/// Errors from the tick window store.
#[derive(Debug)]
pub enum TickWindowStoreError {
    Sqlite(rusqlite::Error),
    Window(TickWindowError),
    NoSupervisorRun,
    UnknownPinKind(String),
    TooManyRows { table: &'static str, limit: usize },
    TickMismatch { row: u64, frame: u64 },
}

impl fmt::Display for TickWindowStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(e) => write!(f, "sqlite error: {e}"),
            Self::Window(e) => write!(f, "tick window error: {e}"),
            Self::NoSupervisorRun => write!(f, "no supervisor run in database"),
            Self::UnknownPinKind(k) => write!(f, "unknown pin kind {k:?}"),
            Self::TooManyRows { table, limit } => {
                write!(f, "table {table} holds more than {limit} rows")
            }
            Self::TickMismatch { row, frame } => {
                write!(f, "row tick {row} does not match frame tick {frame}")
            }
        }
    }
}

impl std::error::Error for TickWindowStoreError {}

impl From<rusqlite::Error> for TickWindowStoreError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sqlite(e)
    }
}

impl From<TickWindowError> for TickWindowStoreError {
    fn from(e: TickWindowError) -> Self {
        Self::Window(e)
    }
}

/// Creates the replay tables if absent.
pub fn ensure_schema(db: &Connection) -> rusqlite::Result<()> {
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS supervisor_replay_frame (
            run_id INTEGER NOT NULL,
            tick INTEGER NOT NULL,
            frame BLOB NOT NULL,
            PRIMARY KEY (run_id, tick)
        );
        CREATE TABLE IF NOT EXISTS supervisor_replay_pin (
            run_id INTEGER NOT NULL,
            tick INTEGER NOT NULL,
            kind TEXT NOT NULL,
            slot INTEGER,
            PRIMARY KEY (run_id, tick, kind)
        );",
    )
}

fn to_i64<T>(v: T) -> rusqlite::Result<i64>
where
    T: TryInto<i64>,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    v.try_into()
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
}

fn from_i64(v: i64) -> rusqlite::Result<u64> {
    u64::try_from(v).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(0, v))
}

/// Writer keeping the database equal to its in-memory window.
pub struct TickWindowStore {
    conn: Connection,
    run_id: i64,
    window: TickWindow,
}

impl TickWindowStore {
    /// Opens `db_path` read-write, selects the newest run with
    /// `mode = 'supervisor'`, ensures the replay schema and starts with an
    /// empty window. Returns `NoSupervisorRun` when no such run exists.
    pub fn open_for_latest_supervisor_run(
        db_path: &str,
        config: TickWindowConfig,
    ) -> Result<Self, TickWindowStoreError> {
        let window = TickWindow::new(config)?;
        let conn = Connection::open(db_path)?;
        conn.execute_batch(WRITER_PRAGMA)?;
        let run_id: i64 = conn
            .query_row(
                "SELECT id FROM run WHERE mode = 'supervisor' ORDER BY id DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()?
            .ok_or(TickWindowStoreError::NoSupervisorRun)?;
        ensure_schema(&conn)?;
        Ok(Self {
            conn,
            run_id,
            window,
        })
    }

    /// The in-memory window the database mirrors.
    pub fn window(&self) -> &TickWindow {
        &self.window
    }

    /// Id of the supervisor run this store writes to.
    pub fn run_id(&self) -> i64 {
        self.run_id
    }

    /// Captures and persists a frame; returns false (writing nothing) when rejected.
    pub fn record_frame(&mut self, frame: &WindowFrame) -> Result<bool, TickWindowStoreError> {
        if !self.window.capture(frame.clone()) {
            return Ok(false);
        }
        let blob = encode_frame(frame);
        let tick = to_i64(frame.tick)?;
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT OR REPLACE INTO supervisor_replay_frame (run_id, tick, frame) VALUES (?1, ?2, ?3)",
            params![self.run_id, tick, blob],
        )?;
        sync(&tx, self.run_id, &self.window)?;
        tx.commit()?;
        Ok(true)
    }

    /// Records a pin and persists the resulting window state.
    pub fn record_pin(&mut self, pin: Pin) -> Result<(), TickWindowStoreError> {
        self.window.pin(pin);
        let tx = self.conn.transaction()?;
        sync(&tx, self.run_id, &self.window)?;
        tx.commit()?;
        Ok(())
    }
}

fn sync(tx: &Connection, run_id: i64, window: &TickWindow) -> rusqlite::Result<()> {
    let held: HashSet<i64> = window
        .frames()
        .map(|f| to_i64(f.tick))
        .collect::<rusqlite::Result<_>>()?;
    let stored: Vec<i64> = {
        let mut stmt = tx.prepare("SELECT tick FROM supervisor_replay_frame WHERE run_id = ?1")?;
        let rows = stmt.query_map(params![run_id], |r| r.get(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    for tick in stored.into_iter().filter(|t| !held.contains(t)) {
        tx.execute(
            "DELETE FROM supervisor_replay_frame WHERE run_id = ?1 AND tick = ?2",
            params![run_id, tick],
        )?;
    }
    tx.execute(
        "DELETE FROM supervisor_replay_pin WHERE run_id = ?1",
        params![run_id],
    )?;
    for pin in window.pins() {
        let (kind, slot) = match pin.kind {
            PinKind::Replacement { slot } => (KIND_REPLACEMENT, Some(to_i64(slot)?)),
            PinKind::SeasonFlip => (KIND_SEASON, None),
        };
        tx.execute(
            "INSERT INTO supervisor_replay_pin (run_id, tick, kind, slot) VALUES (?1, ?2, ?3, ?4)",
            params![run_id, to_i64(pin.tick)?, kind, slot],
        )?;
    }
    Ok(())
}

fn table_exists(conn: &Connection, name: &str) -> rusqlite::Result<bool> {
    let found: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
            params![name],
            |r| r.get(0),
        )
        .optional()?;
    Ok(found.is_some())
}

/// Loads a window without writing to the database.
pub fn load_read_only(
    db_path: &str,
    run_id: Option<i64>,
    config: TickWindowConfig,
) -> Result<TickWindow, TickWindowStoreError> {
    let mut window = TickWindow::new(config.clone())?;
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))?;
    let run_id: i64 = match run_id {
        None => conn
            .query_row(
                "SELECT id FROM run WHERE mode = 'supervisor' ORDER BY id DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()?,
        Some(id) => conn
            .query_row(
                "SELECT id FROM run WHERE id = ?1 AND mode = 'supervisor'",
                params![id],
                |r| r.get(0),
            )
            .optional()?,
    }
    .ok_or(TickWindowStoreError::NoSupervisorRun)?;

    if !table_exists(&conn, "supervisor_replay_frame")?
        || !table_exists(&conn, "supervisor_replay_pin")?
    {
        return Ok(window);
    }

    let mut pins = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT tick, kind, slot FROM supervisor_replay_pin WHERE run_id = ?1 ORDER BY rowid LIMIT ?2",
        )?;
        let limit = config.max_pins + 1;
        let rows = stmt.query_map(params![run_id, to_i64(limit)?], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<i64>>(2)?,
            ))
        })?;
        for row in rows {
            let (tick, kind, slot) = row?;
            if pins.len() >= config.max_pins {
                return Err(TickWindowStoreError::TooManyRows {
                    table: "supervisor_replay_pin",
                    limit: config.max_pins,
                });
            }
            let kind = match kind.as_str() {
                KIND_REPLACEMENT => {
                    let slot = slot.ok_or(rusqlite::Error::InvalidColumnType(
                        2,
                        "slot".to_string(),
                        rusqlite::types::Type::Null,
                    ))?;
                    PinKind::Replacement {
                        slot: usize::try_from(slot)
                            .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(2, slot))?,
                    }
                }
                KIND_SEASON => PinKind::SeasonFlip,
                _ => return Err(TickWindowStoreError::UnknownPinKind(kind)),
            };
            pins.push(Pin {
                tick: from_i64(tick)?,
                kind,
            });
        }
    }
    for pin in pins {
        window.pin(pin);
    }

    let limit = config.max_frames + config.max_pins * config.max_frames_per_pin + 1;
    let mut stmt = conn.prepare(
        "SELECT tick, frame FROM supervisor_replay_frame WHERE run_id = ?1 ORDER BY tick LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![run_id, to_i64(limit)?], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
    })?;
    let mut count = 0usize;
    for row in rows {
        let (tick, blob) = row?;
        count += 1;
        if count >= limit {
            return Err(TickWindowStoreError::TooManyRows {
                table: "supervisor_replay_frame",
                limit: limit - 1,
            });
        }
        let frame = decode_frame(&blob)?;
        let row_tick = from_i64(tick)?;
        if frame.tick != row_tick {
            return Err(TickWindowStoreError::TickMismatch {
                row: row_tick,
                frame: frame.tick,
            });
        }
        window.capture(frame);
    }
    Ok(window)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick_window::FRAME_CODEC_VERSION;

    fn frame(tick: u64) -> WindowFrame {
        WindowFrame {
            tick,
            agents: Vec::new(),
            food: vec![[0.0, 0.0, 0.0, 1.0]],
        }
    }

    fn small() -> TickWindowConfig {
        TickWindowConfig {
            window_ticks: 10,
            max_frames: 5,
            pin_half_window_ticks: 2,
            max_pins: 2,
            max_frames_per_pin: 3,
        }
    }

    fn seed(path: &str, mode: Option<&str>) {
        let db = Connection::open(path).unwrap();
        db.execute_batch("CREATE TABLE run (id INTEGER PRIMARY KEY, mode TEXT);")
            .unwrap();
        if let Some(m) = mode {
            db.execute("INSERT INTO run (id, mode) VALUES (1, ?1)", params![m])
                .unwrap();
        }
    }

    fn path(dir: &tempfile::TempDir) -> String {
        dir.path().join("t.db").to_str().unwrap().to_string()
    }

    fn disk_ticks(p: &str) -> Vec<u64> {
        let db = Connection::open(p).unwrap();
        let mut s = db
            .prepare("SELECT tick FROM supervisor_replay_frame ORDER BY tick")
            .unwrap();
        s.query_map([], |r| r.get::<_, i64>(0))
            .unwrap()
            .map(|t| u64::try_from(t.unwrap()).unwrap())
            .collect()
    }

    fn disk_pin_count(p: &str) -> i64 {
        Connection::open(p)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM supervisor_replay_pin", [], |r| {
                r.get(0)
            })
            .unwrap()
    }

    #[test]
    fn store_schema_is_idempotent() {
        let db = Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();
        ensure_schema(&db).unwrap();
    }

    #[test]
    fn store_loads_pre_existing_supervisor_database() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        {
            let db = Connection::open(&p).unwrap();
            crate::supervisor_store::ensure_schema(&db).unwrap();
        }
        let w = load_read_only(&p, None, small()).unwrap();
        assert!(w.is_empty());
        assert!(load_read_only(&p, Some(1), small()).unwrap().is_empty());
    }

    #[test]
    fn store_loads_database_without_replay_tables() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        assert!(load_read_only(&p, None, small()).unwrap().is_empty());
    }

    #[test]
    fn store_refuses_database_without_supervisor_run() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("generation"));
        assert!(matches!(
            load_read_only(&p, None, small()),
            Err(TickWindowStoreError::NoSupervisorRun)
        ));
        assert!(matches!(
            load_read_only(&p, Some(1), small()),
            Err(TickWindowStoreError::NoSupervisorRun)
        ));
        assert!(matches!(
            TickWindowStore::open_for_latest_supervisor_run(&p, small()),
            Err(TickWindowStoreError::NoSupervisorRun)
        ));
    }

    #[test]
    fn store_disk_bounds_match_window() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        let mut s = TickWindowStore::open_for_latest_supervisor_run(&p, small()).unwrap();
        for t in 1..=40u64 {
            assert!(s.record_frame(&frame(t)).unwrap());
            if t % 7 == 0 {
                s.record_pin(Pin {
                    tick: t,
                    kind: PinKind::SeasonFlip,
                })
                .unwrap();
            }
            let held: Vec<u64> = s.window().frames().map(|f| f.tick).collect();
            assert_eq!(disk_ticks(&p), held);
            assert_eq!(
                disk_pin_count(&p),
                i64::try_from(s.window().pins().len()).unwrap()
            );
            assert!(disk_pin_count(&p) <= 2);
        }
        assert!(!s.record_frame(&frame(5)).unwrap());
        assert_eq!(disk_ticks(&p).last(), Some(&40));
    }

    #[test]
    fn store_disk_pins_survive_rolling_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        let mut s = TickWindowStore::open_for_latest_supervisor_run(&p, small()).unwrap();
        s.record_pin(Pin {
            tick: 5,
            kind: PinKind::Replacement { slot: 3 },
        })
        .unwrap();
        for t in 1..=60u64 {
            s.record_frame(&frame(t)).unwrap();
        }
        let ticks = disk_ticks(&p);
        for t in [4, 5, 6] {
            assert!(ticks.contains(&t), "tick {t} missing");
        }
        let loaded = load_read_only(&p, None, small()).unwrap();
        assert_eq!(loaded.pins(), s.window().pins());
        let a: Vec<u64> = loaded.frames().map(|f| f.tick).collect();
        assert_eq!(a, ticks);
    }

    #[test]
    fn store_load_read_only_leaves_bytes_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        {
            let mut s = TickWindowStore::open_for_latest_supervisor_run(&p, small()).unwrap();
            for t in 1..=5u64 {
                s.record_frame(&frame(t)).unwrap();
            }
        }
        let before = std::fs::read(&p).unwrap();
        let w = load_read_only(&p, None, small()).unwrap();
        assert_eq!(w.len(), 5);
        assert_eq!(std::fs::read(&p).unwrap(), before);
    }

    fn insert_blob(p: &str, tick: i64, blob: &[u8]) {
        let db = Connection::open(p).unwrap();
        ensure_schema(&db).unwrap();
        db.execute(
            "INSERT INTO supervisor_replay_frame (run_id, tick, frame) VALUES (1, ?1, ?2)",
            params![tick, blob],
        )
        .unwrap();
    }

    #[test]
    fn store_load_rejects_wrong_version_blob() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        let mut blob = encode_frame(&frame(3));
        blob[..4].copy_from_slice(&(FRAME_CODEC_VERSION + 1).to_le_bytes());
        insert_blob(&p, 3, &blob);
        assert!(matches!(
            load_read_only(&p, None, small()),
            Err(TickWindowStoreError::Window(
                TickWindowError::UnsupportedVersion(_)
            ))
        ));
    }

    #[test]
    fn store_load_rejects_oversized_row_count() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        let cfg = small();
        let limit = cfg.max_frames + cfg.max_pins * cfg.max_frames_per_pin + 1;
        for t in 0..u64::try_from(limit).unwrap() {
            insert_blob(&p, i64::try_from(t).unwrap(), &encode_frame(&frame(t)));
        }
        assert!(matches!(
            load_read_only(&p, None, cfg),
            Err(TickWindowStoreError::TooManyRows { .. })
        ));
    }

    #[test]
    fn store_load_rejects_tick_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        insert_blob(&p, 4, &encode_frame(&frame(9)));
        assert!(matches!(
            load_read_only(&p, None, small()),
            Err(TickWindowStoreError::TickMismatch { row: 4, frame: 9 })
        ));
    }

    #[test]
    fn store_load_rejects_unknown_pin_kind() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        {
            let db = Connection::open(&p).unwrap();
            ensure_schema(&db).unwrap();
            db.execute(
                "INSERT INTO supervisor_replay_pin (run_id, tick, kind, slot) VALUES (1, 1, 'bogus', NULL)",
                [],
            )
            .unwrap();
        }
        assert!(matches!(
            load_read_only(&p, None, small()),
            Err(TickWindowStoreError::UnknownPinKind(_))
        ));
    }

    #[test]
    fn store_load_rejects_replacement_pin_without_slot() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        {
            let db = Connection::open(&p).unwrap();
            ensure_schema(&db).unwrap();
            db.execute(
                "INSERT INTO supervisor_replay_pin (run_id, tick, kind, slot) VALUES (1, 1, 'replacement', NULL)",
                [],
            )
            .unwrap();
        }
        assert!(matches!(
            load_read_only(&p, None, small()),
            Err(TickWindowStoreError::Sqlite(
                rusqlite::Error::InvalidColumnType(2, _, _)
            ))
        ));
    }

    #[test]
    fn store_disk_keeps_nearest_capped_frames_when_pin_range_overflows() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        seed(&p, Some("supervisor"));
        let mut s = TickWindowStore::open_for_latest_supervisor_run(&p, small()).unwrap();
        s.record_pin(Pin {
            tick: 10,
            kind: PinKind::SeasonFlip,
        })
        .unwrap();
        for t in 0..=40 {
            s.record_frame(&frame(t)).unwrap();
        }
        let mem: Vec<u64> = s.window().frames().map(|f| f.tick).collect();
        assert_eq!(disk_ticks(&p), mem);
        for t in [9, 10, 11] {
            assert!(mem.contains(&t), "nearest in-range tick {t} must survive");
        }
        for t in [8, 12] {
            assert!(
                !mem.contains(&t),
                "in-range tick {t} beyond the per-pin cap"
            );
        }
    }
}
