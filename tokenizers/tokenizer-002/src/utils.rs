use std::iter::FromIterator;

/// Converts an arbitrary byte slice into a Latin-1 string, preserving raw values.
pub fn bytes_to_latin1_string(bytes: &[u8]) -> String {
    String::from_iter(bytes.iter().map(|&b| char::from_u32(b as u32).unwrap()))
}

/// Converts a Latin-1 string back into the raw byte sequence.
pub fn latin1_string_to_bytes(text: &str) -> Vec<u8> {
    text.chars().map(|c| c as u8).collect()
}

/// Returns true if `len` is present in the allowed lengths collection.
#[inline]
pub fn is_allowed_length(len: usize, allowed: &[usize]) -> bool {
    allowed.iter().any(|&value| value == len)
}
