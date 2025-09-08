use crate::mapping::bytes_to_latin1_string;
use anyhow::{Context, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Clone, Debug)]
pub enum ChunkingMode {
    // Entire file as a single input
    Complete,
    // Fixed-size chunks in bytes
    Fixed { size: usize },
    // Random chunk sizes between 2^min_exp and 2^max_exp inclusive
    Random { min_exp: u8, max_exp: u8, seed: u64 },
}

#[derive(Clone, Debug, Default)]
pub struct IngestConfig {
    pub follow_symlinks: bool,
    pub add_boundaries: bool, // emit <|start|> and <|end|>
    pub hidden: bool,         // include hidden files
    pub min_file_size: Option<u64>,
    pub max_file_size: Option<u64>,
    // Drop chunks whose estimated Shannon entropy (bits/byte) exceeds this cutoff
    // when `entropy_filter` is true. Default configured via CLI.
    pub entropy_filter: bool,
    pub entropy_cutoff: f64,
    // Only apply entropy filter if chunk length is strictly greater than this.
    pub entropy_min_len: usize,
}

fn should_include(entry: &walkdir::DirEntry, cfg: &IngestConfig) -> bool {
    if !cfg.hidden {
        if let Some(name) = entry.file_name().to_str() {
            if name.starts_with('.') {
                return false;
            }
        }
    }
    if let Ok(md) = entry.metadata() {
        if !md.is_file() {
            return false;
        }
        if let Some(min) = cfg.min_file_size {
            if md.len() < min {
                return false;
            }
        }
        if let Some(max) = cfg.max_file_size {
            if md.len() > max {
                return false;
            }
        }
    }
    true
}

pub fn collect_files<P: AsRef<Path>>(roots: &[P], cfg: &IngestConfig) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for root in roots {
        let mut it = WalkDir::new(root);
        if cfg.follow_symlinks {
            it = it.follow_links(true);
        }
        for e in it.into_iter().filter_map(|e| e.ok()) {
            if should_include(&e, cfg) {
                paths.push(e.path().to_path_buf());
            }
        }
    }
    paths
}

// Iterator over latin-1 encoded training sequences emitted from files.
// It yields sequences as Strings. Boundary markers are emitted as standalone sequences.
pub struct CorpusIter {
    files: Vec<PathBuf>,
    cfg: IngestConfig,
    mode: ChunkingMode,

    // Per-file state
    cur_idx: usize,
    reader: Option<BufReader<File>>,
    // boundary emission state per file
    emitted_start: bool,
    emitted_end: bool,
    // For random mode
    rng: Option<StdRng>,
}

impl CorpusIter {
    pub fn new(files: Vec<PathBuf>, mode: ChunkingMode, cfg: IngestConfig) -> Self {
        let rng = match &mode {
            ChunkingMode::Random { seed, .. } => Some(StdRng::seed_from_u64(*seed)),
            _ => None,
        };
        Self {
            files,
            cfg,
            mode,
            cur_idx: 0,
            reader: None,
            emitted_start: false,
            emitted_end: false,
            rng,
        }
    }

    fn open_next_file(&mut self) -> Result<()> {
        self.reader = None;
        self.emitted_start = false;
        self.emitted_end = false;
        while self.cur_idx < self.files.len() {
            let path = &self.files[self.cur_idx];
            match File::open(path) {
                Ok(f) => {
                    self.reader = Some(BufReader::new(f));
                    return Ok(());
                }
                Err(_) => {
                    // skip unreadable file
                    self.cur_idx += 1;
                    continue;
                }
            }
        }
        Ok(())
    }

    fn next_chunk_len(&mut self) -> usize {
        match self.mode {
            ChunkingMode::Complete => usize::MAX,
            ChunkingMode::Fixed { size } => size,
            ChunkingMode::Random { min_exp, max_exp, .. } => {
                let min = min_exp.min(max_exp);
                let max = max_exp.max(min_exp);
                if let Some(rng) = &mut self.rng {
                    let p = rng.gen_range(min..=max);
                    1usize << p
                } else {
                    // Fallback deterministic mid-range
                    1usize << ((min + max) / 2)
                }
            }
        }
    }
}

#[inline]
fn shannon_entropy_bits_per_byte(bytes: &[u8]) -> f64 {
    // Extremely fast, branch-light entropy estimate over 256 bins.
    // Computes H = (n * ln n - sum_i c_i * ln c_i) / (n * ln 2)
    // Safe for empty/degenerate inputs.
    let n = bytes.len();
    if n <= 1 {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &b in bytes {
        counts[b as usize] = counts[b as usize].saturating_add(1);
    }
    let n_f = n as f64;
    let ln_n = n_f.ln();
    let mut sum = 0.0_f64;
    for &c in &counts {
        if c != 0 {
            let cf = c as f64;
            sum += cf * cf.ln();
        }
    }
    let h_nats = n_f * ln_n - sum;
    h_nats / (n_f * std::f64::consts::LN_2)
}

impl Iterator for CorpusIter {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.reader.is_none() {
                if self.cur_idx >= self.files.len() {
                    return None;
                }
                if self.open_next_file().is_err() {
                    return None;
                }
                // On new file, optionally emit start boundary first
                if self.cfg.add_boundaries && !self.emitted_start {
                    self.emitted_start = true;
                    return Some("<|start|>".to_string());
                }
            }

            // Determine chunk size before borrowing reader mutably
            let target = self.next_chunk_len();

            let reader = match self.reader.as_mut() {
                Some(r) => r,
                None => continue,
            };

            // Read chunk into a small buffer repeatedly until target or EOF
            let mut buf = Vec::with_capacity(target.min(64 * 1024));
            let mut tmp = [0u8; 8192];
            let mut read_total = 0usize;
            loop {
                if read_total >= target {
                    break;
                }
                let to_read = (target - read_total).min(tmp.len());
                match reader.read(&mut tmp[..to_read]) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        buf.extend_from_slice(&tmp[..n]);
                        read_total += n;
                    }
                    Err(_) => {
                        // Skip unreadable; drop file
                        read_total = 0;
                        break;
                    }
                }
            }

            if read_total == 0 {
                // EOF or error. Optionally emit end boundary, then advance to next file.
                if self.cfg.add_boundaries && !self.emitted_end {
                    self.emitted_end = true;
                    return Some("<|end|>".to_string());
                }
                // Advance to next file
                self.cur_idx += 1;
                self.reader = None;
                continue;
            }

            // For Complete mode, keep reading until EOF to get the full file
            if let ChunkingMode::Complete = self.mode {
                let mut rest = Vec::new();
                if reader.read_to_end(&mut rest).is_ok() {
                    buf.extend_from_slice(&rest);
                }
            }

            // Convert to latin-1 string and yield
            // Optional entropy filter before converting
            if self.cfg.entropy_filter && buf.len() > self.cfg.entropy_min_len {
                let h = shannon_entropy_bits_per_byte(&buf);
                if h > self.cfg.entropy_cutoff {
                    // Skip this chunk; continue reading from current file
                    continue;
                }
            }
            let s = bytes_to_latin1_string(&buf);
            // Empty strings are unusual; skip
            if s.is_empty() {
                continue;
            }
            return Some(s);
        }
    }
}
