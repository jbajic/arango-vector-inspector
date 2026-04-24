//! Minimal VPack slice reader.
//!
//! Only supports the value types ArangoDB emits for collection and index
//! definitions: objects (0x0b-0x0e, 0x14), arrays (0x02-0x09, 0x13), strings,
//! unsigned ints, small ints, booleans, null, doubles. Unknown/unsupported
//! types are handled conservatively — `value_len` returns `None` and the
//! parser bails out of that sub-tree.

use anyhow::{Result, anyhow};

#[derive(Clone, Copy)]
pub struct Slice<'a> {
    bytes: &'a [u8],
}

#[allow(dead_code)]
impl<'a> Slice<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Slice { bytes }
    }

    pub fn type_byte(&self) -> u8 {
        self.bytes[0]
    }

    pub fn raw(&self) -> &'a [u8] {
        self.bytes
    }

    /// Total byte length of this VPack value, starting at byte 0.
    pub fn value_len(&self) -> Result<usize> {
        value_len(self.bytes)
    }

    pub fn is_object(&self) -> bool {
        matches!(self.bytes[0], 0x0a..=0x12 | 0x14)
    }

    pub fn is_array(&self) -> bool {
        matches!(self.bytes[0], 0x01..=0x09 | 0x13)
    }

    pub fn is_string(&self) -> bool {
        matches!(self.bytes[0], 0x40..=0xbf)
    }

    pub fn as_str(&self) -> Option<&'a str> {
        let b = self.bytes;
        match b[0] {
            0x40..=0xbe => {
                let len = (b[0] - 0x40) as usize;
                std::str::from_utf8(b.get(1..1 + len)?).ok()
            }
            0xbf => {
                let len = u64::from_le_bytes(b.get(1..9)?.try_into().ok()?) as usize;
                std::str::from_utf8(b.get(9..9 + len)?).ok()
            }
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        let b = self.bytes;
        match b[0] {
            0x28..=0x2f => {
                let len = (b[0] - 0x27) as usize;
                let slice = b.get(1..1 + len)?;
                let mut buf = [0u8; 8];
                buf[..len].copy_from_slice(slice);
                Some(u64::from_le_bytes(buf))
            }
            0x30..=0x39 => Some((b[0] - 0x30) as u64),
            _ => None,
        }
    }

    /// Find a member by key in an object. Linear scan — fine for small
    /// objects like index definitions.
    pub fn get(&self, key: &str) -> Option<Slice<'a>> {
        for (k, v) in self.object_iter().ok()? {
            if k.as_str() == Some(key) {
                return Some(v);
            }
        }
        None
    }

    pub fn object_iter(&self) -> Result<ObjectIter<'a>> {
        ObjectIter::new(self.bytes)
    }

    pub fn array_iter(&self) -> Result<ArrayIter<'a>> {
        ArrayIter::new(self.bytes)
    }
}

// ---- length calculation --------------------------------------------------

fn value_len(b: &[u8]) -> Result<usize> {
    if b.is_empty() {
        return Err(anyhow!("empty vpack value"));
    }
    let t = b[0];
    let len = match t {
        0x00 => return Err(anyhow!("vpack: invalid type 0x00")),
        0x01 => 1,                              // empty array
        0x0a => 1,                              // empty object
        0x02..=0x05 => header_len(b, t)?,       // array no index table
        0x06..=0x09 => header_len(b, t)?,       // array with index table
        0x0b..=0x0e => header_len(b, t)?,       // object with index table
        0x13 | 0x14 => compact_len(b)?,         // compact array/object
        0x17 => 1,                              // illegal
        0x18 | 0x19 | 0x1a => 1,                // null, false, true
        0x1b => 9,                              // double
        0x1c => 9,                              // UTC date
        0x1d => 9,                              // external ptr (8 bytes)
        0x1e | 0x1f => 1,                       // minKey, maxKey
        0x20..=0x27 => 1 + (t - 0x1f) as usize, // signed int
        0x28..=0x2f => 1 + (t - 0x27) as usize, // unsigned int
        0x30..=0x39 => 1,                       // small uint 0..9
        0x3a..=0x3f => 1,                       // small int -6..-1
        0x40..=0xbe => 1 + (t - 0x40) as usize, // short string
        0xbf => {
            if b.len() < 9 {
                return Err(anyhow!("vpack: truncated long string header"));
            }
            let n = u64::from_le_bytes(b[1..9].try_into().unwrap()) as usize;
            9 + n
        }
        0xc0..=0xc7 => {
            let w = (t - 0xbf) as usize;
            if b.len() < 1 + w {
                return Err(anyhow!("vpack: truncated binary header"));
            }
            let mut buf = [0u8; 8];
            buf[..w].copy_from_slice(&b[1..1 + w]);
            1 + w + u64::from_le_bytes(buf) as usize
        }
        _ => return Err(anyhow!("vpack: unsupported type 0x{:02x}", t)),
    };
    if len > b.len() {
        return Err(anyhow!(
            "vpack: value len {} exceeds buffer {}",
            len,
            b.len()
        ));
    }
    Ok(len)
}

fn header_len(b: &[u8], t: u8) -> Result<usize> {
    // width of the whole-length field depends on type
    let w = match t {
        0x02 | 0x06 | 0x0b => 1,
        0x03 | 0x07 | 0x0c => 2,
        0x04 | 0x08 | 0x0d => 4,
        0x05 | 0x09 | 0x0e => 8,
        _ => unreachable!(),
    };
    if b.len() < 1 + w {
        return Err(anyhow!("vpack: truncated header"));
    }
    let mut buf = [0u8; 8];
    buf[..w].copy_from_slice(&b[1..1 + w]);
    Ok(u64::from_le_bytes(buf) as usize)
}

fn compact_len(b: &[u8]) -> Result<usize> {
    // byte length encoded as varlen starting at offset 1
    let (len, _) = read_varlen(&b[1..])?;
    Ok(len as usize)
}

/// VPack variable-length unsigned int. Bytes with high bit set are
/// continuation; each byte contributes 7 bits, little-endian.
fn read_varlen(b: &[u8]) -> Result<(u64, usize)> {
    let mut v: u64 = 0;
    let mut shift = 0;
    for (i, &byte) in b.iter().enumerate() {
        v |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((v, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return Err(anyhow!("vpack: varlen too large"));
        }
    }
    Err(anyhow!("vpack: truncated varlen"))
}

fn read_varlen_be(b: &[u8]) -> Result<(u64, usize)> {
    // compact types store the trailing count as a big-endian varlen
    // (read right-to-left). Each byte contributes 7 bits; the highest-addr
    // byte has its high bit clear, earlier bytes have it set.
    let mut v: u64 = 0;
    let mut i = b.len();
    loop {
        if i == 0 {
            return Err(anyhow!("vpack: truncated be-varlen"));
        }
        i -= 1;
        let byte = b[i];
        v = (v << 7) | ((byte & 0x7f) as u64);
        if byte & 0x80 == 0 {
            return Ok((v, b.len() - i));
        }
    }
}

// ---- object iteration ----------------------------------------------------

pub struct ObjectIter<'a> {
    base: &'a [u8],
    kind: ObjKind,
    idx: usize,
    count: usize,
    // for indexed objects:
    offset_width: usize,
    offset_table_start: usize,
    // for compact objects (0x14):
    compact_cursor: usize,
    compact_end: usize,
}

enum ObjKind {
    Empty,
    Indexed,
    Compact,
}

impl<'a> ObjectIter<'a> {
    fn new(b: &'a [u8]) -> Result<Self> {
        let t = *b.first().ok_or_else(|| anyhow!("vpack: empty"))?;
        match t {
            0x0a => Ok(Self {
                base: b,
                kind: ObjKind::Empty,
                idx: 0,
                count: 0,
                offset_width: 0,
                offset_table_start: 0,
                compact_cursor: 0,
                compact_end: 0,
            }),
            0x0b..=0x0e => {
                let (w, count_at_end) = match t {
                    0x0b => (1, false),
                    0x0c => (2, false),
                    0x0d => (4, false),
                    0x0e => (8, true),
                    _ => unreachable!(),
                };
                let total = header_len(b, t)?;
                let count;
                let offset_table_start;
                if count_at_end {
                    // 0x0e: count is 8 bytes at end; offsets precede it
                    if total < 1 + w + 8 {
                        return Err(anyhow!("vpack: truncated 0x0e header"));
                    }
                    count = u64::from_le_bytes(b[total - 8..total].try_into().unwrap()) as usize;
                    offset_table_start = total - 8 - count * w;
                } else {
                    if b.len() < 1 + w + w {
                        return Err(anyhow!("vpack: truncated indexed-obj header"));
                    }
                    let mut buf = [0u8; 8];
                    buf[..w].copy_from_slice(&b[1 + w..1 + 2 * w]);
                    count = u64::from_le_bytes(buf) as usize;
                    offset_table_start = total - count * w;
                }
                Ok(Self {
                    base: b,
                    kind: ObjKind::Indexed,
                    idx: 0,
                    count,
                    offset_width: w,
                    offset_table_start,
                    compact_cursor: 0,
                    compact_end: 0,
                })
            }
            0x14 => {
                // compact object: [0x14][byte_len varlen][members][count varlen_be]
                let byte_len = compact_len(b)?;
                let (_, bl_w) = read_varlen(&b[1..])?;
                let mut end = byte_len;
                let (count, count_w) = read_varlen_be(&b[..end])?;
                end -= count_w;
                Ok(Self {
                    base: b,
                    kind: ObjKind::Compact,
                    idx: 0,
                    count: count as usize,
                    offset_width: 0,
                    offset_table_start: 0,
                    compact_cursor: 1 + bl_w,
                    compact_end: end,
                })
            }
            _ => Err(anyhow!("vpack: type 0x{:02x} is not an object", t)),
        }
    }
}

impl<'a> Iterator for ObjectIter<'a> {
    type Item = (Slice<'a>, Slice<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.count {
            return None;
        }
        let (key_start, value_start) = match self.kind {
            ObjKind::Empty => return None,
            ObjKind::Indexed => {
                let w = self.offset_width;
                let off_pos = self.offset_table_start + self.idx * w;
                let mut buf = [0u8; 8];
                buf[..w].copy_from_slice(&self.base[off_pos..off_pos + w]);
                let key_off = u64::from_le_bytes(buf) as usize;
                let key_len = value_len(&self.base[key_off..]).ok()?;
                (key_off, key_off + key_len)
            }
            ObjKind::Compact => {
                let key_off = self.compact_cursor;
                let key_len = value_len(&self.base[key_off..]).ok()?;
                let val_off = key_off + key_len;
                let val_len = value_len(&self.base[val_off..]).ok()?;
                self.compact_cursor = val_off + val_len;
                if self.compact_cursor > self.compact_end {
                    return None;
                }
                (key_off, val_off)
            }
        };
        self.idx += 1;
        let key = Slice::new(&self.base[key_start..]);
        let value = Slice::new(&self.base[value_start..]);
        Some((key, value))
    }
}

// ---- array iteration -----------------------------------------------------

pub struct ArrayIter<'a> {
    base: &'a [u8],
    idx: usize,
    count: usize,
    cursor: usize,    // for no-index-table and compact arrays
    item_size: usize, // for no-index-table arrays (0 = variable / use offsets)
    offset_width: usize,
    offset_table_start: usize,
    kind: ArrKind,
}

enum ArrKind {
    Empty,
    Flat,    // 0x02-0x05: items are same-sized, back-to-back
    Indexed, // 0x06-0x09: offset table
    Compact, // 0x13
}

impl<'a> ArrayIter<'a> {
    fn new(b: &'a [u8]) -> Result<Self> {
        let t = *b.first().ok_or_else(|| anyhow!("vpack: empty"))?;
        match t {
            0x01 => Ok(Self {
                base: b,
                idx: 0,
                count: 0,
                cursor: 0,
                item_size: 0,
                offset_width: 0,
                offset_table_start: 0,
                kind: ArrKind::Empty,
            }),
            0x02..=0x05 => {
                let w = match t {
                    0x02 => 1,
                    0x03 => 2,
                    0x04 => 4,
                    0x05 => 8,
                    _ => unreachable!(),
                };
                let total = header_len(b, t)?;
                // skip padding zero bytes after the length field until first
                // value starts (VPack aligns values in 0x02-0x05).
                let mut cursor = 1 + w;
                while cursor < total && b[cursor] == 0 {
                    cursor += 1;
                }
                if cursor >= total {
                    return Ok(Self {
                        base: b,
                        idx: 0,
                        count: 0,
                        cursor,
                        item_size: 0,
                        offset_width: 0,
                        offset_table_start: 0,
                        kind: ArrKind::Empty,
                    });
                }
                let item_size = value_len(&b[cursor..total])?;
                let count = (total - cursor) / item_size;
                Ok(Self {
                    base: b,
                    idx: 0,
                    count,
                    cursor,
                    item_size,
                    offset_width: 0,
                    offset_table_start: 0,
                    kind: ArrKind::Flat,
                })
            }
            0x06..=0x09 => {
                let (w, count_at_end) = match t {
                    0x06 => (1, false),
                    0x07 => (2, false),
                    0x08 => (4, false),
                    0x09 => (8, true),
                    _ => unreachable!(),
                };
                let total = header_len(b, t)?;
                let count;
                let offset_table_start;
                if count_at_end {
                    if total < 1 + w + 8 {
                        return Err(anyhow!("vpack: truncated 0x09 header"));
                    }
                    count = u64::from_le_bytes(b[total - 8..total].try_into().unwrap()) as usize;
                    offset_table_start = total - 8 - count * w;
                } else {
                    if b.len() < 1 + 2 * w {
                        return Err(anyhow!("vpack: truncated indexed-arr header"));
                    }
                    let mut buf = [0u8; 8];
                    buf[..w].copy_from_slice(&b[1 + w..1 + 2 * w]);
                    count = u64::from_le_bytes(buf) as usize;
                    offset_table_start = total - count * w;
                }
                Ok(Self {
                    base: b,
                    idx: 0,
                    count,
                    cursor: 0,
                    item_size: 0,
                    offset_width: w,
                    offset_table_start,
                    kind: ArrKind::Indexed,
                })
            }
            0x13 => {
                // compact array: [0x13][byte_len varlen][items][count varlen_be]
                let byte_len = compact_len(b)?;
                let (_, bl_w) = read_varlen(&b[1..])?;
                let mut end = byte_len;
                let (count, count_w) = read_varlen_be(&b[..end])?;
                end -= count_w;
                Ok(Self {
                    base: b,
                    idx: 0,
                    count: count as usize,
                    cursor: 1 + bl_w,
                    item_size: 0,
                    offset_width: 0,
                    offset_table_start: end,
                    kind: ArrKind::Compact,
                })
            }
            _ => Err(anyhow!("vpack: type 0x{:02x} is not an array", t)),
        }
    }
}

impl<'a> Iterator for ArrayIter<'a> {
    type Item = Slice<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.count {
            return None;
        }
        let start = match self.kind {
            ArrKind::Empty => return None,
            ArrKind::Flat => {
                let s = self.cursor;
                self.cursor += self.item_size;
                s
            }
            ArrKind::Indexed => {
                let w = self.offset_width;
                let off_pos = self.offset_table_start + self.idx * w;
                let mut buf = [0u8; 8];
                buf[..w].copy_from_slice(&self.base[off_pos..off_pos + w]);
                u64::from_le_bytes(buf) as usize
            }
            ArrKind::Compact => {
                let s = self.cursor;
                let n = value_len(&self.base[s..self.offset_table_start]).ok()?;
                self.cursor += n;
                s
            }
        };
        self.idx += 1;
        Some(Slice::new(&self.base[start..]))
    }
}
