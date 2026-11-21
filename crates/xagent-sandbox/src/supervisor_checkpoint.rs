//! Pure, GPU-free versioned checkpoint envelope with a trailing FNV-1a-64 checksum.

use std::fmt;

/// Current envelope format version.
pub const FORMAT_VERSION: u32 = 1;
/// Largest checkpoint `decode` accepts.
pub const MAX_CHECKPOINT_BYTES: usize = 512 * 1024 * 1024;
const MAGIC: [u8; 8] = *b"XAGCKPT\0";
const MAX_SECTIONS: usize = 256;
const MAX_NAME_BYTES: usize = 64;

/// One checkpoint: a tick and named opaque byte sections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub tick: u64,
    pub sections: Vec<(String, Vec<u8>)>,
}

impl Checkpoint {
    /// Bytes of the first section with this name.
    pub fn section(&self, name: &str) -> Result<&[u8], CheckpointError> {
        self.sections
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
            .ok_or(CheckpointError::Missing)
    }
}

/// Reasons a checkpoint is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointError {
    Truncated,
    BadMagic,
    VersionMismatch { found: u32, expected: u32 },
    ChecksumMismatch,
    TooLarge { len: usize, max: usize },
    Malformed(String),
    ConfigMismatch,
    Missing,
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => {
                write!(
                    f,
                    "checkpoint is truncated; it ends before its declared content"
                )
            }
            Self::BadMagic => write!(f, "checkpoint does not start with the expected magic bytes"),
            Self::VersionMismatch { found, expected } => write!(
                f,
                "checkpoint format version {found} is not supported (expected {expected})"
            ),
            Self::ChecksumMismatch => write!(f, "checkpoint checksum does not match its content"),
            Self::TooLarge { len, max } => {
                write!(
                    f,
                    "checkpoint is {len} bytes, larger than the {max} byte limit"
                )
            }
            Self::Malformed(why) => write!(f, "checkpoint is malformed: {why}"),
            Self::ConfigMismatch => {
                write!(f, "checkpoint was written for a different configuration")
            }
            Self::Missing => write!(f, "checkpoint is missing a required section"),
        }
    }
}

impl std::error::Error for CheckpointError {}

/// FNV-1a 64-bit offset basis (the standard initial hash value).
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime (the standard per-byte multiplier).
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Fixed header size: magic (8) + version (4) + tick (8) + section count (4).
const HEADER_BYTES: usize = 24;
/// Byte offset of the `u32` format version, right after the 8-byte magic.
const VERSION_OFFSET: usize = 8;
/// Byte offset of the `u64` tick, right after the 4-byte version.
const TICK_OFFSET: usize = 12;
/// Byte offset of the `u32` section count, right after the 8-byte tick.
const COUNT_OFFSET: usize = 20;
/// Width of a section's `u16` name length field.
const NAME_LEN_BYTES: usize = 2;
/// Width of a section's `u64` data length field.
const DATA_LEN_BYTES: usize = 8;
/// Width of the trailing `u64` checksum.
const CHECKSUM_BYTES: usize = 8;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = FNV1A64_OFFSET_BASIS;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV1A64_PRIME);
    }
    h
}

/// Serialise a checkpoint. Panics if it has over 256 sections or a name over 64 bytes.
pub fn encode(checkpoint: &Checkpoint) -> Vec<u8> {
    assert!(
        checkpoint.sections.len() <= MAX_SECTIONS,
        "too many checkpoint sections"
    );
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&checkpoint.tick.to_le_bytes());
    let count = u32::try_from(checkpoint.sections.len()).expect("section count asserted <= 256");
    out.extend_from_slice(&count.to_le_bytes());
    for (name, data) in &checkpoint.sections {
        assert!(
            name.len() <= MAX_NAME_BYTES,
            "checkpoint section name too long"
        );
        let name_len = u16::try_from(name.len()).expect("name length asserted <= 64");
        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        let data_len = u64::try_from(data.len()).expect("usize fits in u64");
        out.extend_from_slice(&data_len.to_le_bytes());
        out.extend_from_slice(data);
    }
    let sum = fnv1a64(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

fn take<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], CheckpointError> {
    let end = pos.checked_add(n).ok_or(CheckpointError::Truncated)?;
    let s = bytes.get(*pos..end).ok_or(CheckpointError::Truncated)?;
    *pos = end;
    Ok(s)
}

/// Decode and fully validate a checkpoint.
pub fn decode(bytes: &[u8]) -> Result<Checkpoint, CheckpointError> {
    if bytes.len() > MAX_CHECKPOINT_BYTES {
        return Err(CheckpointError::TooLarge {
            len: bytes.len(),
            max: MAX_CHECKPOINT_BYTES,
        });
    }
    // Framing walk.
    let mut pos = 0usize;
    let header = take(bytes, &mut pos, HEADER_BYTES)?;
    let count = u32::from_le_bytes(header[COUNT_OFFSET..HEADER_BYTES].try_into().unwrap());
    let count = usize::try_from(count).map_err(|_| CheckpointError::Truncated)?;
    if count > MAX_SECTIONS {
        return Err(CheckpointError::Malformed("too many sections".into()));
    }
    let mut ranges = Vec::with_capacity(count);
    for _ in 0..count {
        let nl = u16::from_le_bytes(take(bytes, &mut pos, NAME_LEN_BYTES)?.try_into().unwrap());
        let nl = usize::from(nl);
        if nl > MAX_NAME_BYTES {
            return Err(CheckpointError::Malformed("section name too long".into()));
        }
        let name_start = pos;
        take(bytes, &mut pos, nl)?;
        let dl = u64::from_le_bytes(take(bytes, &mut pos, DATA_LEN_BYTES)?.try_into().unwrap());
        let dl = usize::try_from(dl).map_err(|_| CheckpointError::Truncated)?;
        let data_start = pos;
        take(bytes, &mut pos, dl)?;
        ranges.push((name_start..name_start + nl, data_start..data_start + dl));
    }
    let remaining = bytes.len() - pos;
    if remaining < CHECKSUM_BYTES {
        return Err(CheckpointError::Truncated);
    }
    if remaining > CHECKSUM_BYTES {
        return Err(CheckpointError::Malformed("trailing bytes".into()));
    }
    if bytes[..MAGIC.len()] != MAGIC {
        return Err(CheckpointError::BadMagic);
    }
    let found = u32::from_le_bytes(bytes[VERSION_OFFSET..TICK_OFFSET].try_into().unwrap());
    if found != FORMAT_VERSION {
        return Err(CheckpointError::VersionMismatch {
            found,
            expected: FORMAT_VERSION,
        });
    }
    let trailer = u64::from_le_bytes(bytes[pos..].try_into().unwrap());
    if fnv1a64(&bytes[..pos]) != trailer {
        return Err(CheckpointError::ChecksumMismatch);
    }
    let tick = u64::from_le_bytes(bytes[TICK_OFFSET..COUNT_OFFSET].try_into().unwrap());
    let mut sections: Vec<(String, Vec<u8>)> = Vec::with_capacity(count);
    for (nr, dr) in ranges {
        let name = std::str::from_utf8(&bytes[nr])
            .map_err(|_| CheckpointError::Malformed("section name is not UTF-8".into()))?;
        if sections.iter().any(|(n, _)| n == name) {
            return Err(CheckpointError::Malformed(format!(
                "duplicate section name {name}"
            )));
        }
        sections.push((name.to_string(), bytes[dr].to_vec()));
    }
    Ok(Checkpoint { tick, sections })
}

/// Serde helpers storing `f32` by bit pattern.
pub mod f32_bits {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Serialise an `f32` as its `to_bits()` `u32` pattern.
    pub fn serialize<S: Serializer>(v: &f32, s: S) -> Result<S::Ok, S::Error> {
        v.to_bits().serialize(s)
    }

    /// Deserialise an `f32` from its `u32` bit pattern via `from_bits`.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
        Ok(f32::from_bits(u32::deserialize(d)?))
    }

    /// Same pair for `Vec<f32>`.
    pub mod vec {
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        /// Serialise a slice of `f32` as a sequence of `to_bits()` `u32` patterns.
        pub fn serialize<S: Serializer>(v: &[f32], s: S) -> Result<S::Ok, S::Error> {
            let bits: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
            bits.serialize(s)
        }

        /// Deserialise a `Vec<f32>` from a sequence of `u32` bit patterns.
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
            Ok(Vec::<u32>::deserialize(d)?
                .into_iter()
                .map(f32::from_bits)
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    fn sample() -> Checkpoint {
        Checkpoint {
            tick: 42,
            sections: vec![
                ("config".to_string(), b"{\"a\":1}".to_vec()),
                ("run".to_string(), vec![1, 2, 3]),
                ("kernel.x".to_string(), Vec::new()),
            ],
        }
    }

    #[test]
    fn round_trip_restores_sections() {
        let c = sample();
        let d = decode(&encode(&c)).unwrap();
        assert_eq!(c, d);
        assert_eq!(d.section("run").unwrap(), &[1, 2, 3]);
        assert_eq!(d.section("nope"), Err(CheckpointError::Missing));
    }

    #[test]
    fn every_truncation_is_refused_as_truncated() {
        let e = encode(&sample());
        for len in 0..e.len() {
            assert_eq!(decode(&e[..len]), Err(CheckpointError::Truncated), "{len}");
        }
    }

    #[test]
    fn every_flipped_byte_is_refused() {
        let e = encode(&sample());
        for i in 0..e.len() {
            let mut b = e.clone();
            b[i] ^= 0xFF;
            assert!(decode(&b).is_err(), "{i}");
        }
    }

    #[test]
    fn version_two_is_refused() {
        let mut e = encode(&sample());
        e[VERSION_OFFSET..TICK_OFFSET].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            decode(&e),
            Err(CheckpointError::VersionMismatch {
                found: 2,
                expected: 1
            })
        );
    }

    #[test]
    fn oversize_is_refused_as_too_large() {
        let b = vec![0u8; MAX_CHECKPOINT_BYTES + 1];
        assert!(matches!(decode(&b), Err(CheckpointError::TooLarge { .. })));
    }

    #[test]
    fn bad_magic_is_refused() {
        let mut e = encode(&sample());
        e[0] = b'Y';
        assert_eq!(decode(&e), Err(CheckpointError::BadMagic));
    }

    #[test]
    fn duplicate_names_are_malformed() {
        let c = Checkpoint {
            tick: 1,
            sections: vec![("a".into(), vec![1]), ("a".into(), vec![2])],
        };
        assert!(matches!(
            decode(&encode(&c)),
            Err(CheckpointError::Malformed(_))
        ));
    }

    #[derive(Serialize, Deserialize)]
    struct Floats {
        #[serde(with = "f32_bits")]
        one: f32,
        #[serde(with = "f32_bits::vec")]
        many: Vec<f32>,
    }

    #[test]
    fn f32_bits_round_trip_bit_patterns() {
        let many = vec![-0.0f32, f32::from_bits(1), f32::INFINITY, f32::NEG_INFINITY];
        let f = Floats {
            one: f32::from_bits(0x7FC0_1234),
            many: many.clone(),
        };
        let json = serde_json::to_string(&f).unwrap();
        let g: Floats = serde_json::from_str(&json).unwrap();
        assert_eq!(g.one.to_bits(), 0x7FC0_1234);
        assert_eq!(g.many.len(), many.len());
        for (a, b) in g.many.iter().zip(&many) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
