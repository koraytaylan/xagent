//! Binary free-life tick log: a JSON header line, then little-endian `f32` rows.

use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::metrics::{FreeLifeSample, ENCODING_WIDTH, PATHWAY_INPUTS};

/// Largest integer an `f32` stores exactly.
pub const MAX_EXACT_F32_INTEGER: u64 = 1 << 24;
/// Reader cap on the header line length in bytes.
pub const MAX_HEADER_BYTES: usize = 65_536;
const FORMAT_NAME: &str = "xagent-free-life-tick-log";
const FORMAT_VERSION: u32 = 1;
const ENDIANNESS: &str = "little";
const LEADING_COLUMNS: [&str; 11] = [
    "brain_tick",
    "agent",
    "alive",
    "meals_total",
    "deaths_total",
    "on_hazard",
    "motor_turn",
    "food_distance",
    "food_bearing",
    "hazard_distance",
    "hazard_bearing",
];
const TRAILING_COLUMNS: [&str; 10] = [
    "reward",
    "value",
    "td_error",
    "forward_noise",
    "turn_noise",
    "turn_innovation",
    "encoding_turn_output",
    "scent_turn_output",
    "vision_turn_output",
    "encoding_forward_output",
];
/// Number of columns in every row.
pub const COLUMN_COUNT: usize =
    LEADING_COLUMNS.len() + ENCODING_WIDTH + PATHWAY_INPUTS + TRAILING_COLUMNS.len();
const BYTES_PER_VALUE: u64 = 4;

/// The tick log header line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TickLogHeader {
    pub format: String,
    pub version: u32,
    pub endianness: String,
    pub columns: Vec<String>,
    pub rows: u64,
}

/// The column names in row order.
pub fn columns() -> Vec<String> {
    let mut cols: Vec<String> = LEADING_COLUMNS.iter().map(|c| (*c).to_string()).collect();
    cols.extend((0..ENCODING_WIDTH).map(|i| format!("encoding_{i}")));
    cols.extend((0..PATHWAY_INPUTS).map(|i| format!("pathway_input_{i}")));
    cols.extend(TRAILING_COLUMNS.iter().map(|c| (*c).to_string()));
    cols
}

fn header_for(rows: u64) -> TickLogHeader {
    TickLogHeader {
        format: FORMAT_NAME.to_string(),
        version: FORMAT_VERSION,
        endianness: ENDIANNESS.to_string(),
        columns: columns(),
        rows,
    }
}

fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Validating tick log writer.
pub struct TickLogWriter {
    out: BufWriter<File>,
    expected_rows: u64,
    rows_written: u64,
}

impl TickLogWriter {
    pub fn create(path: &Path, expected_rows: u64) -> io::Result<Self> {
        let mut out = BufWriter::new(File::create(path)?);
        let header = serde_json::to_string(&header_for(expected_rows))
            .map_err(|e| invalid(e.to_string()))?;
        out.write_all(header.as_bytes())?;
        out.write_all(b"\n")?;
        Ok(Self {
            out,
            expected_rows,
            rows_written: 0,
        })
    }

    pub fn write_sample(&mut self, agent: u32, sample: &FreeLifeSample) -> io::Result<()> {
        if sample.encoding.len() != ENCODING_WIDTH {
            return Err(invalid(format!(
                "encoding has {} values, expected {ENCODING_WIDTH}",
                sample.encoding.len()
            )));
        }
        for (name, value) in [
            ("brain_tick", sample.brain_tick),
            ("agent", u64::from(agent)),
            ("meals_total", sample.meals_total),
            ("deaths_total", sample.deaths_total),
        ] {
            if value > MAX_EXACT_F32_INTEGER {
                return Err(invalid(format!("{name} {value} is not exact in f32")));
            }
        }
        let mut row: Vec<f32> = Vec::with_capacity(COLUMN_COUNT);
        row.push(integer_f32(sample.brain_tick)?);
        row.push(integer_f32(u64::from(agent))?);
        row.push(f32::from(u8::from(sample.alive)));
        row.push(integer_f32(sample.meals_total)?);
        row.push(integer_f32(sample.deaths_total)?);
        row.push(f32::from(u8::from(sample.on_hazard)));
        row.extend([
            sample.motor_turn,
            sample.food_distance,
            sample.food_bearing,
            sample.hazard_distance,
            sample.hazard_bearing,
        ]);
        row.extend(&sample.encoding);
        row.extend(sample.pathway_inputs);
        row.extend([
            sample.reward,
            sample.value,
            sample.td_error,
            sample.forward_noise,
            sample.turn_noise,
            sample.turn_innovation,
            sample.encoding_turn_output,
            sample.scent_turn_output,
            sample.vision_turn_output,
            sample.encoding_forward_output,
        ]);
        debug_assert_eq!(row.len(), COLUMN_COUNT);
        for v in row {
            self.out.write_all(&v.to_le_bytes())?;
        }
        self.rows_written += 1;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<()> {
        self.out.flush()?;
        if self.rows_written != self.expected_rows {
            return Err(invalid(format!(
                "wrote {} rows, expected {}",
                self.rows_written, self.expected_rows
            )));
        }
        Ok(())
    }
}

/// Exact `f32` of an integer at most 2^24.
///
/// The value is split into two `u16` halves, each exact in `f32`, and
/// recombined; the sum stays at most 2^24 so it is exact too.
fn integer_f32(n: u64) -> io::Result<f32> {
    if n > MAX_EXACT_F32_INTEGER {
        return Err(invalid(format!("{n} is not exact in f32")));
    }
    let hi = u16::try_from(n >> 16).map_err(|e| invalid(e.to_string()))?;
    let lo = u16::try_from(n & 0xFFFF).map_err(|e| invalid(e.to_string()))?;
    Ok(f32::from(hi) * 65_536.0 + f32::from(lo))
}

/// Why a tick log could not be read.
#[derive(Debug)]
pub enum TickLogError {
    Io(io::Error),
    HeaderTooLong,
    HeaderWithoutNewline,
    MalformedHeader(String),
    WrongColumns,
    SizeMismatch { expected: u64, actual: u64 },
}

impl fmt::Display for TickLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "tick log I/O error: {e}"),
            Self::HeaderTooLong => write!(f, "tick log header exceeds {MAX_HEADER_BYTES} bytes"),
            Self::HeaderWithoutNewline => write!(f, "tick log header has no terminating newline"),
            Self::MalformedHeader(m) => write!(f, "malformed tick log header: {m}"),
            Self::WrongColumns => write!(f, "tick log column list is not the specified one"),
            Self::SizeMismatch { expected, actual } => {
                write!(f, "tick log payload is {actual} bytes, expected {expected}")
            }
        }
    }
}

impl std::error::Error for TickLogError {}

impl From<io::Error> for TickLogError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// A fully loaded tick log.
#[derive(Clone, Debug, PartialEq)]
pub struct TickLog {
    pub columns: Vec<String>,
    pub rows: u64,
    /// Row-major: row `r`, column `c` at `r * columns.len() + c`.
    pub values: Vec<f32>,
}

impl TickLog {
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }
}

fn read_header_line(reader: &mut BufReader<File>) -> Result<Vec<u8>, TickLogError> {
    let mut line = Vec::new();
    let mut limited = reader.by_ref().take(
        u64::try_from(MAX_HEADER_BYTES)
            .unwrap_or(u64::MAX)
            .saturating_add(1),
    );
    limited.read_until(b'\n', &mut line)?;
    if line.last() != Some(&b'\n') {
        return Err(if line.len() > MAX_HEADER_BYTES {
            TickLogError::HeaderTooLong
        } else {
            TickLogError::HeaderWithoutNewline
        });
    }
    if line.len() > MAX_HEADER_BYTES {
        return Err(TickLogError::HeaderTooLong);
    }
    Ok(line)
}

pub fn read_tick_log(path: &Path) -> Result<TickLog, TickLogError> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let line = read_header_line(&mut reader)?;
    let header: TickLogHeader =
        serde_json::from_slice(&line).map_err(|e| TickLogError::MalformedHeader(e.to_string()))?;
    let reference = header_for(header.rows);
    if header.format != reference.format
        || header.version != reference.version
        || header.endianness != reference.endianness
    {
        return Err(TickLogError::MalformedHeader(
            "unexpected format, version or endianness".to_string(),
        ));
    }
    if header.columns != reference.columns {
        return Err(TickLogError::WrongColumns);
    }
    let header_len =
        u64::try_from(line.len()).map_err(|e| TickLogError::MalformedHeader(e.to_string()))?;
    let column_count =
        u64::try_from(COLUMN_COUNT).map_err(|e| TickLogError::MalformedHeader(e.to_string()))?;
    let actual = file_len - header_len;
    let expected = header
        .rows
        .checked_mul(column_count)
        .and_then(|v| v.checked_mul(BYTES_PER_VALUE));
    let expected = match expected {
        Some(e) if e == actual => e,
        Some(e) => {
            return Err(TickLogError::SizeMismatch {
                expected: e,
                actual,
            })
        }
        None => {
            return Err(TickLogError::SizeMismatch {
                expected: u64::MAX,
                actual,
            })
        }
    };
    let mut bytes = vec![
        0_u8;
        usize::try_from(expected)
            .map_err(|_| { TickLogError::SizeMismatch { expected, actual } })?
    ];
    reader.read_exact(&mut bytes)?;
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok(TickLog {
        columns: header.columns,
        rows: header.rows,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact(n: u64) -> f32 {
        integer_f32(n).unwrap()
    }

    fn exact_usize(n: usize) -> f32 {
        exact(u64::try_from(n).unwrap())
    }

    fn seed_for(agent: u32, tick: u64) -> f32 {
        10.0 * exact(u64::from(agent)) + exact(tick)
    }

    fn sample(tick: u64, seed: f32) -> FreeLifeSample {
        FreeLifeSample {
            brain_tick: tick,
            alive: true,
            meals_total: tick + 2,
            deaths_total: 1,
            on_hazard: true,
            motor_turn: seed + 0.1,
            food_distance: seed + 0.2,
            food_bearing: seed + 0.3,
            hazard_distance: seed + 0.4,
            hazard_bearing: seed + 0.5,
            encoding: (0..ENCODING_WIDTH).map(|i| seed + exact_usize(i)).collect(),
            pathway_inputs: std::array::from_fn(|i| seed * 2.0 + exact_usize(i)),
            reward: seed + 0.6,
            value: seed + 0.7,
            td_error: seed + 0.8,
            forward_noise: seed + 0.9,
            turn_noise: seed + 1.1,
            turn_innovation: seed + 1.2,
            encoding_turn_output: seed + 1.3,
            scent_turn_output: seed + 1.4,
            vision_turn_output: seed + 1.5,
            encoding_forward_output: seed + 1.6,
        }
    }

    fn write_valid(path: &Path) {
        let mut w = TickLogWriter::create(path, 4).unwrap();
        for tick in 0..2 {
            for agent in 0..2_u32 {
                w.write_sample(agent, &sample(tick, seed_for(agent, tick)))
                    .unwrap();
            }
        }
        w.finish().unwrap();
    }

    fn rewrite_header(path: &Path, header: &str, keep_newline: bool) {
        let bytes = std::fs::read(path).unwrap();
        let split = bytes.iter().position(|b| *b == b'\n').unwrap();
        let mut out = header.as_bytes().to_vec();
        if keep_newline {
            out.push(b'\n');
        }
        out.extend_from_slice(&bytes[split + 1..]);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn column_count_matches_columns() {
        let cols = columns();
        assert_eq!(cols.len(), COLUMN_COUNT);
        assert_eq!(COLUMN_COUNT, 159);
        assert_eq!(cols[0], "brain_tick");
        assert_eq!(cols[11], "encoding_0");
        assert_eq!(cols[COLUMN_COUNT - 1], "encoding_forward_output");
    }

    #[test]
    fn round_trip_all_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.bin");
        write_valid(&path);
        let log = read_tick_log(&path).unwrap();
        assert_eq!(log.rows, 4);
        assert_eq!(log.values.len(), 4 * COLUMN_COUNT);
        let mut row = 0;
        for tick in 0..2_u64 {
            for agent in 0..2_u32 {
                let s = sample(tick, seed_for(agent, tick));
                let get = |name: &str| {
                    let c = log.column_index(name).unwrap();
                    log.values[row * COLUMN_COUNT + c]
                };
                assert_eq!(get("brain_tick"), exact(tick));
                assert_eq!(get("agent"), exact(u64::from(agent)));
                assert_eq!(get("alive"), 1.0);
                assert_eq!(get("meals_total"), exact(s.meals_total));
                assert_eq!(get("deaths_total"), 1.0);
                assert_eq!(get("on_hazard"), 1.0);
                assert_eq!(get("motor_turn"), s.motor_turn);
                assert_eq!(get("food_distance"), s.food_distance);
                assert_eq!(get("food_bearing"), s.food_bearing);
                assert_eq!(get("hazard_distance"), s.hazard_distance);
                assert_eq!(get("hazard_bearing"), s.hazard_bearing);
                for i in 0..ENCODING_WIDTH {
                    assert_eq!(get(&format!("encoding_{i}")), s.encoding[i]);
                }
                for i in 0..PATHWAY_INPUTS {
                    assert_eq!(get(&format!("pathway_input_{i}")), s.pathway_inputs[i]);
                }
                assert_eq!(get("reward"), s.reward);
                assert_eq!(get("value"), s.value);
                assert_eq!(get("td_error"), s.td_error);
                assert_eq!(get("forward_noise"), s.forward_noise);
                assert_eq!(get("turn_noise"), s.turn_noise);
                assert_eq!(get("turn_innovation"), s.turn_innovation);
                assert_eq!(get("encoding_turn_output"), s.encoding_turn_output);
                assert_eq!(get("scent_turn_output"), s.scent_turn_output);
                assert_eq!(get("vision_turn_output"), s.vision_turn_output);
                assert_eq!(get("encoding_forward_output"), s.encoding_forward_output);
                row += 1;
            }
        }
    }

    #[test]
    fn writer_finish_short_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.bin");
        let mut w = TickLogWriter::create(&path, 2).unwrap();
        w.write_sample(0, &sample(0, 1.0)).unwrap();
        assert!(w.finish().is_err());
    }

    #[test]
    fn reader_rejects_truncated_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.bin");
        write_valid(&path);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.pop();
        std::fs::write(&path, bytes).unwrap();
        assert!(matches!(
            read_tick_log(&path),
            Err(TickLogError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn reader_rejects_wrong_row_count() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.bin");
        write_valid(&path);
        let header = serde_json::to_string(&header_for(5)).unwrap();
        rewrite_header(&path, &header, true);
        assert!(matches!(
            read_tick_log(&path),
            Err(TickLogError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn reader_rejects_wrong_column_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.bin");
        write_valid(&path);
        let mut header = header_for(4);
        header.columns.swap(0, 1);
        rewrite_header(&path, &serde_json::to_string(&header).unwrap(), true);
        assert!(matches!(
            read_tick_log(&path),
            Err(TickLogError::WrongColumns)
        ));
    }

    #[test]
    fn reader_rejects_malformed_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.bin");
        write_valid(&path);
        rewrite_header(&path, "{not json", true);
        assert!(matches!(
            read_tick_log(&path),
            Err(TickLogError::MalformedHeader(_))
        ));
    }

    #[test]
    fn reader_rejects_unknown_header_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.bin");
        write_valid(&path);
        let mut value = serde_json::to_value(header_for(4)).unwrap();
        value["extra"] = serde_json::json!(1);
        rewrite_header(&path, &value.to_string(), true);
        assert!(matches!(
            read_tick_log(&path),
            Err(TickLogError::MalformedHeader(_))
        ));
    }

    #[test]
    fn reader_rejects_header_without_newline() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.bin");
        std::fs::write(&path, b"{\"format\":\"xagent-free-life-tick-log\"").unwrap();
        assert!(matches!(
            read_tick_log(&path),
            Err(TickLogError::HeaderWithoutNewline)
        ));
    }
}
