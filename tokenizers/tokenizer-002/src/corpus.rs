use crate::config::IngestConfig;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Collects file paths from the provided list, expanding directories according to configuration.
pub fn collect_paths<P: AsRef<Path>>(inputs: &[P], cfg: &IngestConfig) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        let path = input.as_ref();
        if !path.exists() {
            anyhow::bail!("Input path {:?} does not exist", path);
        }
        let metadata = path
            .symlink_metadata()
            .with_context(|| format!("Unable to stat {:?}", path))?;
        if metadata.is_dir() {
            if cfg.recursive {
                let walker = WalkDir::new(path).follow_links(cfg.follow_symlinks);
                for entry in walker {
                    let entry = entry?;
                    if entry.file_type().is_file() {
                        files.push(entry.path().to_path_buf());
                    }
                }
            } else {
                for entry in std::fs::read_dir(path)? {
                    let entry = entry?;
                    let entry_path = entry.path();
                    if entry_path.is_file() {
                        files.push(entry_path);
                    }
                }
            }
        } else if metadata.is_file() {
            files.push(path.to_path_buf());
        }
    }
    if files.is_empty() {
        anyhow::bail!("No files discovered in the provided inputs");
    }
    Ok(files)
}

/// Materialises binary data from a list of paths into a collection of byte sequences.
///
/// Each file is optionally chunked, yielding one sequence per chunk.
pub fn load_binary_corpus<P: AsRef<Path>>(
    inputs: &[P],
    cfg: &IngestConfig,
) -> Result<Vec<Vec<u8>>> {
    let file_paths = collect_paths(inputs, cfg)?;
    let mut sequences = Vec::new();
    for file_path in file_paths {
        let mut file =
            File::open(&file_path).with_context(|| format!("Failed to open {:?}", file_path))?;
        if cfg.chunk_size == 0 {
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)
                .with_context(|| format!("Failed to read {:?}", file_path))?;
            if !buffer.is_empty() {
                sequences.push(buffer);
            }
            continue;
        }

        loop {
            let mut buffer = vec![0u8; cfg.chunk_size];
            let read = file
                .read(&mut buffer)
                .with_context(|| format!("Failed to read chunk from {:?}", file_path))?;
            if read == 0 {
                break;
            }
            buffer.truncate(read);
            sequences.push(buffer);
        }
    }
    if sequences.is_empty() {
        anyhow::bail!("No binary data could be loaded from the provided inputs");
    }
    Ok(sequences)
}
