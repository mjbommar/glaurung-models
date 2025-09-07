use crate::ingest::{collect_files, ChunkingMode, CorpusIter, IngestConfig};
use crate::train::{pad_vocab_to_power_of_two, train_tokenizer, TrainerConfig};
use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ModeArg {
    Complete,
    Fixed,
    Random,
}

#[derive(Parser, Debug)]
#[command(name = "bbpe")] 
#[command(about = "Binary BPE tokenizer trainer", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train a tokenizer and save tokenizer.json
    Train {
        /// Input paths (files or directories)
        #[arg(required = true)]
        input: Vec<PathBuf>,

        /// Chunking mode
        #[arg(long, value_enum, default_value_t = ModeArg::Fixed)]
        mode: ModeArg,

        /// Fixed chunk size in bytes
        #[arg(long, default_value_t = 4096)]
        fixed_bytes: usize,

        /// Random chunk size exponent min (2^min)
        #[arg(long, default_value_t = 3)]
        min_chunk_exp: u8,

        /// Random chunk size exponent max (2^max)
        #[arg(long, default_value_t = 14)]
        max_chunk_exp: u8,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Add <|start|> and <|end|> around files
        #[arg(long, default_value_t = true)]
        boundaries: bool,

        /// Vocab size target (BPE)
        #[arg(long, default_value_t = 32768)]
        vocab_size: usize,

        /// Minimum frequency for merges/pairs
        #[arg(long, default_value_t = 1024)]
        min_frequency: u64,

        /// Show training progress
        #[arg(long, default_value_t = true)]
        progress: bool,

        /// Attach template post-processor
        #[arg(long, default_value_t = true)]
        template: bool,

        /// Number of reserved special tokens to include initially
        #[arg(long, default_value_t = 128)]
        reserved: usize,

        /// Pad to next power of two
        #[arg(long, default_value_t = true)]
        pad_pow2: bool,

        /// Maximum length (in characters) for learned tokens during merges
        #[arg(long, default_value_t = 32)]
        max_token_length: usize,

        /// Output tokenizer.json path
        #[arg(long, default_value = "tokenizer.json")]
        output: PathBuf,
    },
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train {
            input,
            mode,
            fixed_bytes,
            min_chunk_exp,
            max_chunk_exp,
            seed,
            boundaries,
            vocab_size,
            min_frequency,
            progress,
            template,
            reserved,
            pad_pow2,
            max_token_length,
            output,
        } => {
            let ingest_cfg = IngestConfig {
                follow_symlinks: false,
                add_boundaries: boundaries,
                hidden: false,
                min_file_size: None,
                max_file_size: None,
            };
            let files = collect_files(&input, &ingest_cfg);
            if files.is_empty() {
                return Err(anyhow!("No files found under provided input paths"));
            }
            let chunking = match mode {
                ModeArg::Complete => ChunkingMode::Complete,
                ModeArg::Fixed => ChunkingMode::Fixed { size: fixed_bytes },
                ModeArg::Random => ChunkingMode::Random {
                    min_exp: min_chunk_exp,
                    max_exp: max_chunk_exp,
                    seed,
                },
            };
            let iter = CorpusIter::new(files, chunking, ingest_cfg);

            let trainer_cfg = TrainerConfig {
                vocab_size,
                min_frequency,
                show_progress: progress,
                with_template: template,
                reserved_count: reserved,
                pad_to_power_of_two: pad_pow2,
                max_token_length,
            };

            let mut tokenizer = train_tokenizer(iter, &trainer_cfg)?;
            if pad_pow2 {
                pad_vocab_to_power_of_two(&mut tokenizer)?;
            }
            // Save tokenizer.json
            let f = std::fs::File::create(&output)
                .with_context(|| format!("creating {:?}", output))?;
            serde_json::to_writer_pretty(f, &tokenizer)?;
            eprintln!("Saved tokenizer to {:?}", output);
        }
    }

    Ok(())
}
