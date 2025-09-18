use crate::ingest::{collect_files, ChunkingMode, CorpusIter, IngestConfig};
use crate::patterns::{PatternConfig, PatternType};
use crate::train::{pad_vocab_to_power_of_two, train_tokenizer, TrainerConfig};
use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

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

        /// Random chunk size exponent min (inclusive, uses 2^min)
        #[arg(long, default_value_t = 3)]
        min_chunk_exp: u8,

        /// Random chunk size exponent max (exclusive, uses 2^p for p < max)
        #[arg(long, default_value_t = 14)]
        max_chunk_exp: u8,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Random sampling rate for chunks (0.0..=1.0)
        /// A value of 0.1 keeps ~1/10 chunks; 1.0 keeps all
        #[arg(long, default_value_t = 1.0)]
        sample_rate: f64,

        /// Add <|start|> and <|end|> around files
        #[arg(long, default_value_t = true)]
        boundaries: bool,

        /// Enable high-entropy chunk filtering
        #[arg(long, default_value_t = true)]
        entropy_filter: bool,

        /// Shannon entropy cutoff (bits/byte) to drop chunks
        #[arg(long, default_value_t = 7.0)]
        entropy_cutoff: f64,

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

        /// Enable common opcode/pattern sequences
        #[arg(long, default_value_t = false)]
        enable_patterns: bool,

        /// Minimum pattern length as power of 2 (e.g., 2 means 2^2 = 4 bytes)
        #[arg(long, default_value_t = 2)]
        min_pattern_pow2: u8,

        /// Maximum pattern length as power of 2 (exclusive, e.g., 11 means up to 2^10 = 1024)
        #[arg(long, default_value_t = 11)]
        max_pattern_pow2: u8,

        /// Enable null byte patterns (0x00)
        #[arg(long, default_value_t = true)]
        pattern_null: bool,

        /// Enable x86 NOP patterns (0x90)
        #[arg(long, default_value_t = true)]
        pattern_nop_x86: bool,

        /// Enable x86 INT3 patterns (0xCC)
        #[arg(long, default_value_t = true)]
        pattern_int3: bool,

        /// Enable FF padding patterns (0xFF)
        #[arg(long, default_value_t = true)]
        pattern_ff: bool,

        /// Enable RISC NOP patterns (0x00000000)
        #[arg(long, default_value_t = false)]
        pattern_nop_risc: bool,

        /// Enable space patterns (0x20)
        #[arg(long, default_value_t = true)]
        pattern_space: bool,

        /// Enable dot patterns (0x2E)
        #[arg(long, default_value_t = false)]
        pattern_dot: bool,

        /// Enable slash patterns (0x2F)
        #[arg(long, default_value_t = false)]
        pattern_slash: bool,

        /// Enable 0x01 byte patterns
        #[arg(long, default_value_t = false)]
        pattern_one: bool,

        /// Enable ASCII '0' patterns (0x30)
        #[arg(long, default_value_t = false)]
        pattern_zero: bool,
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
            sample_rate,
            boundaries,
            entropy_filter,
            entropy_cutoff,
            vocab_size,
            min_frequency,
            progress,
            template,
            reserved,
            pad_pow2,
            max_token_length,
            output,
            enable_patterns,
            min_pattern_pow2,
            max_pattern_pow2,
            pattern_null,
            pattern_nop_x86,
            pattern_int3,
            pattern_ff,
            pattern_nop_risc,
            pattern_space,
            pattern_dot,
            pattern_slash,
            pattern_one,
            pattern_zero,
        } => {
            let ingest_cfg = IngestConfig {
                follow_symlinks: false,
                add_boundaries: boundaries,
                hidden: false,
                min_file_size: None,
                max_file_size: None,
                entropy_filter,
                entropy_cutoff,
                entropy_min_len: 16,
                // Clamp sample_rate to [0.0, 1.0]
                sample_rate: sample_rate.clamp(0.0, 1.0),
                seed,
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

            // Build pattern configuration
            let mut patterns = Vec::new();
            if pattern_null {
                patterns.push(PatternType::Null);
            }
            if pattern_nop_x86 {
                patterns.push(PatternType::NopX86);
            }
            if pattern_int3 {
                patterns.push(PatternType::Int3);
            }
            if pattern_ff {
                patterns.push(PatternType::FfPad);
            }
            if pattern_nop_risc {
                patterns.push(PatternType::NopRisc);
            }
            if pattern_space {
                patterns.push(PatternType::Space);
            }
            if pattern_dot {
                patterns.push(PatternType::Dot);
            }
            if pattern_slash {
                patterns.push(PatternType::Slash);
            }
            if pattern_one {
                patterns.push(PatternType::One);
            }
            if pattern_zero {
                patterns.push(PatternType::Zero);
            }
            
            let pattern_cfg = PatternConfig {
                enabled: enable_patterns,
                min_power: min_pattern_pow2,
                max_power: max_pattern_pow2,
                patterns,
            };
            
            let trainer_cfg = TrainerConfig {
                vocab_size,
                min_frequency,
                show_progress: progress,
                with_template: template,
                reserved_count: reserved,
                pad_to_power_of_two: pad_pow2,
                max_token_length,
            };

            let mut tokenizer = train_tokenizer(iter, &trainer_cfg, &pattern_cfg)?;
            if pad_pow2 {
                pad_vocab_to_power_of_two(&mut tokenizer)?;
            }
            // Save tokenizer.json
            let f = std::fs::File::create(&output)
                .with_context(|| format!("creating {output:?}"))?;
            serde_json::to_writer_pretty(f, &tokenizer)?;
            eprintln!("Saved tokenizer to {output:?}");
        }
    }

    Ok(())
}
