use anyhow::{anyhow, Result};

// Convert raw bytes to a Latin-1 (ISO-8859-1) string where each byte maps 1:1
// to a Unicode code point U+0000..=U+00FF. This round-trips with
// `latin1_string_to_bytes`.
pub fn bytes_to_latin1_string(bytes: &[u8]) -> String {
    // Safe: all u8 map to valid Unicode scalar values in this range
    bytes.iter().map(|&b| char::from(b)).collect()
}

// Convert a Latin-1 string back to raw bytes. Errors if any char is outside
// the U+0000..=U+00FF range (should not happen if produced by the function above).
pub fn latin1_string_to_bytes(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        let v = ch as u32;
        if v <= 0xFF {
            out.push(v as u8);
        } else {
            return Err(anyhow!(
                "Non-Latin-1 character encountered during decode: U+{:04X}",
                v
            ));
        }
    }
    Ok(out)
}

