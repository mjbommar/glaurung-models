use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tokenizer_002::config::{IngestConfig, TrainerConfig};
use tokenizer_002::corpus::load_binary_corpus;
use tokenizer_002::trainer::BytePairTrainer;

#[derive(Debug, Parser)]
#[command(
    name = "tokenizer-002-train",
    about = "Train a constrained byte-level BPE tokenizer for binary corpora"
)]
struct Args {
    /// Output path for the Hugging Face tokenizer JSON.
    #[arg(short, long)]
    output: PathBuf,

    /// Input files or directories containing binary data.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Target vocabulary size (includes base 256 byte tokens and special tokens).
    #[arg(long, default_value_t = 32_768)]
    vocab_size: usize,

    /// Minimum frequency a pair must have to be merged.
    #[arg(long, default_value_t = 4)]
    min_frequency: usize,

    /// Chunk size in bytes when ingesting files. 0 reads entire files.
    #[arg(long, default_value_t = 8192)]
    chunk_size: usize,

    /// Do not traverse directories recursively.
    #[arg(long, default_value_t = false)]
    no_recursive: bool,

    /// Allowed token lengths, comma separated (default: 1,2,4,8).
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,2,4,8",
        num_args = 1..=4
    )]
    allowed_lengths: Vec<usize>,

    /// Frequency floor that signals the merge histogram has plateaued.
    #[arg(long, default_value_t = 128)]
    plateau_frequency: usize,

    /// Stop once best frequency falls below the initial frequency divided by this value.
    #[arg(long, default_value_t = 512)]
    plateau_divisor: usize,

    /// Number of consecutive low-frequency iterations tolerated before stopping.
    #[arg(long, default_value_t = 32)]
    plateau_patience: usize,

    /// Hard cap on merge iterations regardless of target vocab size.
    #[arg(long)]
    max_merges: Option<usize>,

    /// Enable plateau-based early stopping (disabled by default now that merges are incremental).
    #[arg(long, default_value_t = false)]
    plateau_stop: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let ingest_cfg = IngestConfig {
        chunk_size: args.chunk_size,
        recursive: !args.no_recursive,
        follow_symlinks: false,
    };

    let mut trainer_cfg = TrainerConfig::default();
    trainer_cfg.target_vocab_size = args.vocab_size;
    trainer_cfg.min_frequency = args.min_frequency;
    trainer_cfg.allowed_token_lengths = args.allowed_lengths;
    trainer_cfg.plateau_frequency_floor = args.plateau_frequency;
    trainer_cfg.plateau_patience = args.plateau_patience;
    trainer_cfg.plateau_frequency_divisor = args.plateau_divisor;
    trainer_cfg.max_merge_iterations = args.max_merges.or(trainer_cfg.max_merge_iterations);
    trainer_cfg.plateau_stop_enabled = args.plateau_stop;

    let corpus = load_binary_corpus(&args.inputs, &ingest_cfg)?;
    let trainer = BytePairTrainer::new(trainer_cfg);
    let artefacts = trainer.train_from_sequences(&corpus)?;
    let output = artefacts.output;
    output.save_tokenizer(&args.output)?;

    eprintln!(
        "Saved tokenizer with {} base tokens and {} merges to {:?}",
        output.token_bytes.len(),
        output.merges.len(),
        args.output
    );
    eprintln!("Training metrics: {}", artefacts.metrics.summary());
    Ok(())
}
