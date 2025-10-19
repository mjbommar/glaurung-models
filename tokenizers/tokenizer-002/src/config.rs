use serde::{Deserialize, Serialize};

/// Configuration for byte-pair encoding (BPE) training on raw binary data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Target vocabulary size including base byte tokens (256) and added special tokens.
    pub target_vocab_size: usize,
    /// Minimum frequency a merge pair must reach to be selected.
    pub min_frequency: usize,
    /// Explicit list of allowed token lengths in bytes.
    pub allowed_token_lengths: Vec<usize>,
    /// Whether to emit progress information to stderr.
    pub show_progress: bool,
    /// Special tokens appended to the tokenizer after training.
    pub special_tokens: Vec<String>,
    /// Frequency threshold that indicates we're approaching a plateau.
    pub plateau_frequency_floor: usize,
    /// How many consecutive iterations below the plateau floor before stopping.
    pub plateau_patience: usize,
    /// Stop when best frequency falls below the initial frequency divided by this factor.
    pub plateau_frequency_divisor: usize,
    /// Optional hard cap on merge iterations regardless of target vocab.
    pub max_merge_iterations: Option<usize>,
    /// Enable plateau-based early stopping; otherwise plateau metrics are observational only.
    pub plateau_stop_enabled: bool,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            target_vocab_size: 32_768,
            min_frequency: 4,
            allowed_token_lengths: vec![1, 2, 4, 8],
            show_progress: true,
            special_tokens: vec![
                "<s>".into(),
                "</s>".into(),
                "<pad>".into(),
                "<unk>".into(),
                "<cls>".into(),
                "<sep>".into(),
                "<mask>".into(),
            ],
            plateau_frequency_floor: 128,
            plateau_patience: 32,
            plateau_frequency_divisor: 512,
            max_merge_iterations: None,
            plateau_stop_enabled: false,
        }
    }
}

/// Configuration controlling how the binary corpus is materialised.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestConfig {
    /// Number of bytes per chunk. A value of 0 reads each file as a single sequence.
    pub chunk_size: usize,
    /// Walk directories recursively when ingesting paths.
    pub recursive: bool,
    /// Follow symbolic links during directory traversal.
    pub follow_symlinks: bool,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            chunk_size: 8192,
            recursive: true,
            follow_symlinks: false,
        }
    }
}
