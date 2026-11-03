//! Just enough bencode for `.torrent` facts and scrape replies.

use anyhow::{anyhow, bail, Result};
use sha1::{Digest, Sha1};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum B {
    Int(i64),
    Bytes(Vec<u8>),
    List(Vec<B>),
    Dict(BTreeMap<Vec<u8>, B>),
}

impl B {
    pub fn get(&self, k: &str) -> Option<&B> {
        match self {
            B::Dict(d) => d.get(k.as_bytes()),
            _ => None,
        }
    }
    pub fn int(&self) -> Option<i64> {
        match self {
            B::Int(i) => Some(*i),
            _ => None,
        }
    }
    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            B::Bytes(b) => Some(b),
            _ => None,
        }
    }
    pub fn str(&self) -> Option<String> {
        self.bytes().map(|b| String::from_utf8_lossy(b).to_string())
    }
}

/// Decodes one value at `i`; returns it and the index after it. `info_span`
/// records where the top-level `info` dictionary's bytes are.
fn decode_at(data: &[u8], i: usize, depth: usize, info_span: &mut Option<(usize, usize)>) -> Result<(B, usize)> {
    if depth > 64 {
        bail!("bencode nested too deep");
    }
    let c = *data.get(i).ok_or_else(|| anyhow!("bencode truncated"))?;
    match c {
        b'i' => {
            let end = data[i..].iter().position(|&b| b == b'e').ok_or_else(|| anyhow!("int unterminated"))? + i;
            let s = std::str::from_utf8(&data[i + 1..end])?;
            Ok((B::Int(s.parse()?), end + 1))
        }
        b'l' => {
            let mut j = i + 1;
            let mut out = Vec::new();
            while *data.get(j).ok_or_else(|| anyhow!("list unterminated"))? != b'e' {
                let (v, n) = decode_at(data, j, depth + 1, info_span)?;
                out.push(v);
                j = n;
            }
            Ok((B::List(out), j + 1))
        }
        b'd' => {
            let mut j = i + 1;
            let mut out = BTreeMap::new();
            while *data.get(j).ok_or_else(|| anyhow!("dict unterminated"))? != b'e' {
                let (k, n) = decode_at(data, j, depth + 1, info_span)?;
                let k = match k {
                    B::Bytes(b) => b,
                    _ => bail!("dict key not a string"),
                };
                let start = n;
                let (v, n2) = decode_at(data, n, depth + 1, info_span)?;
                if depth == 0 && k == b"info" {
                    *info_span = Some((start, n2));
                }
                out.insert(k, v);
                j = n2;
            }
            Ok((B::Dict(out), j + 1))
        }
        b'0'..=b'9' => {
            let colon = data[i..].iter().position(|&b| b == b':').ok_or_else(|| anyhow!("string length unterminated"))? + i;
            let n: usize = std::str::from_utf8(&data[i..colon])?.parse()?;
            let end = colon + 1 + n;
            if end > data.len() {
                bail!("string past the end");
            }
            Ok((B::Bytes(data[colon + 1..end].to_vec()), end))
        }
        _ => bail!("bencode: unexpected byte {c}"),
    }
}

pub fn decode(data: &[u8]) -> Result<B> {
    let mut span = None;
    Ok(decode_at(data, 0, 0, &mut span)?.0)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TorrentFacts {
    pub hash: String,
    pub private: bool,
    pub size: f64,
    pub name: String,
    pub files: usize,
    pub piece_length: i64,
    /// Carries the passkey: never logged, never stored in an event.
    #[serde(skip)]
    pub announce: String,
}

pub fn torrent_facts(raw: &[u8]) -> Result<TorrentFacts> {
    let mut span = None;
    let (meta, _) = decode_at(raw, 0, 0, &mut span)?;
    let info = meta.get("info").ok_or_else(|| anyhow!("no info dictionary"))?;
    let (s, e) = span.ok_or_else(|| anyhow!("no info span"))?;
    let hash = hex::encode(Sha1::digest(&raw[s..e]));
    let private = info.get("private").and_then(|v| v.int()) == Some(1);
    let (size, files) = match info.get("files") {
        Some(B::List(fs)) => (fs.iter().filter_map(|f| f.get("length").and_then(|l| l.int())).sum::<i64>() as f64, fs.len()),
        _ => (info.get("length").and_then(|l| l.int()).unwrap_or(0) as f64, 1),
    };
    Ok(TorrentFacts {
        hash,
        private,
        size,
        name: info.get("name").and_then(|n| n.str()).unwrap_or_default(),
        files,
        piece_length: info.get("piece length").and_then(|v| v.int()).unwrap_or(0),
        announce: meta.get("announce").and_then(|a| a.str()).unwrap_or_default(),
    })
}

/// (complete, incomplete, downloaded) per info-hash from a scrape reply.
pub fn scrape_reply(raw: &[u8]) -> Result<BTreeMap<String, (i64, i64, i64)>> {
    let b = decode(raw)?;
    if let Some(f) = b.get("failure reason") {
        bail!("scrape refused: {}", f.str().unwrap_or_default());
    }
    let mut out = BTreeMap::new();
    if let Some(B::Dict(files)) = b.get("files") {
        for (k, v) in files {
            out.insert(
                hex::encode(k),
                (
                    v.get("complete").and_then(|x| x.int()).unwrap_or(0),
                    v.get("incomplete").and_then(|x| x.int()).unwrap_or(0),
                    v.get("downloaded").and_then(|x| x.int()).unwrap_or(0),
                ),
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<u8> {
        let info = b"d6:lengthi1000e4:name4:test12:piece lengthi16384e6:pieces0:7:privatei1ee";
        let mut t = b"d8:announce33:https://tracker.test/a/x/announce4:info".to_vec();
        t.extend_from_slice(info);
        t.push(b'e');
        t
    }

    #[test]
    fn facts() {
        let raw = sample();
        let f = torrent_facts(&raw).unwrap();
        assert!(f.private);
        assert_eq!(f.size, 1000.0);
        assert_eq!(f.name, "test");
        let info = b"d6:lengthi1000e4:name4:test12:piece lengthi16384e6:pieces0:7:privatei1ee";
        assert_eq!(f.hash, hex::encode(Sha1::digest(info)));
        assert_eq!(f.announce, "https://tracker.test/a/x/announce");
    }

    #[test]
    fn scrape() {
        let mut raw = b"d5:filesd20:".to_vec();
        raw.extend_from_slice(&[0xab; 20]);
        raw.extend_from_slice(b"d8:completei5e10:downloadedi9e10:incompletei7eeee");
        let m = scrape_reply(&raw).unwrap();
        assert_eq!(m.get(&"ab".repeat(20)), Some(&(5, 7, 9)));
    }
}
