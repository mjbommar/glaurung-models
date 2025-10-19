use crate::config::TrainerConfig;
use crate::metrics::{IterationMetrics, TrainingMetrics, TrainingStopReason, current_rss_kb};
use crate::utils::{bytes_to_latin1_string, is_allowed_length};
use ahash::AHashMap;
use anyhow::{Result, anyhow};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, hash_map::Entry};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use tokenizers::tokenizer::AddedToken;

type TokenId = u32;
type Pair = (TokenId, TokenId);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct PairScore {
    frequency: usize,
    pair: Pair,
}

impl PairScore {
    fn new(pair: Pair, frequency: usize) -> Self {
        Self { frequency, pair }
    }
}

impl Ord for PairScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frequency
            .cmp(&other.frequency)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for PairScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Output artefacts produced by the byte-pair trainer.
#[derive(Debug, Clone)]
pub struct TrainingOutput {
    pub token_bytes: Vec<Vec<u8>>,
    pub merges: Vec<Pair>,
    pub config: TrainerConfig,
}

/// Combined artefacts and telemetry from a training run.
#[derive(Debug, Clone)]
pub struct TrainingArtifacts {
    pub output: TrainingOutput,
    pub metrics: TrainingMetrics,
}

impl TrainingOutput {
    /// Helper to convert tokens into their Latin-1 string representation.
    fn token_strings(&self) -> Vec<String> {
        self.token_bytes
            .iter()
            .map(|bytes| bytes_to_latin1_string(bytes))
            .collect()
    }

    /// Builds a Hugging Face `Tokenizer` in-memory from the trained artefacts.
    pub fn build_tokenizer(&self) -> Result<Tokenizer> {
        let token_strings = self.token_strings();
        let mut vocab = AHashMap::with_capacity(token_strings.len());
        for (idx, token) in token_strings.iter().enumerate() {
            vocab.insert(token.clone(), idx as u32);
        }

        let merges = self
            .merges
            .iter()
            .map(|(left, right)| {
                (
                    token_strings[*left as usize].clone(),
                    token_strings[*right as usize].clone(),
                )
            })
            .collect::<Vec<_>>();

        let builder = BPE::builder().vocab_and_merges(vocab, merges);
        let model = builder.build().map_err(|err| anyhow!(err))?;
        let mut tokenizer = Tokenizer::new(model);

        if !self.config.special_tokens.is_empty() {
            let added = self
                .config
                .special_tokens
                .iter()
                .map(|token| AddedToken::from(token.clone(), true))
                .collect::<Vec<_>>();
            tokenizer.add_special_tokens(&added);
        }
        Ok(tokenizer)
    }

    /// Persist the model to disk in Hugging Face JSON format.
    pub fn save_tokenizer<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let tokenizer = self.build_tokenizer()?;
        tokenizer
            .save(path.as_ref(), false)
            .map_err(|err| anyhow!(err))?;
        Ok(())
    }
}

/// Trainer implementation performing constrained BPE over raw byte sequences.
pub struct BytePairTrainer {
    cfg: TrainerConfig,
}

impl BytePairTrainer {
    pub fn new(cfg: TrainerConfig) -> Self {
        Self { cfg }
    }

    /// Executes training from an in-memory collection of byte sequences.
    pub fn train_from_sequences(&self, sequences: &[Vec<u8>]) -> Result<TrainingArtifacts> {
        if sequences.is_empty() {
            anyhow::bail!("Training requires at least one sequence");
        }
        if !self.cfg.allowed_token_lengths.iter().any(|&len| len == 1) {
            anyhow::bail!("Allowed token lengths must include 1 for base byte tokens");
        }
        let allowed_lengths = self.cfg.allowed_token_lengths.clone();
        let base_vocab = 256usize;
        let special_count = self.cfg.special_tokens.len();
        if self.cfg.target_vocab_size < base_vocab + special_count {
            anyhow::bail!(
                "target_vocab_size ({}) must be >= {} (base bytes + special tokens)",
                self.cfg.target_vocab_size,
                base_vocab + special_count
            );
        }
        let max_new_tokens = self.cfg.target_vocab_size - base_vocab - special_count;
        let mut working_sequences: Vec<Vec<TokenId>> = sequences
            .iter()
            .map(|seq| seq.iter().map(|&b| b as TokenId).collect())
            .collect();

        let mut token_bytes: Vec<Vec<u8>> = (0u32..256).map(|b| vec![b as u8]).collect();
        let mut token_lengths: Vec<usize> = vec![1; token_bytes.len()];
        let mut merges: Vec<Pair> = Vec::with_capacity(max_new_tokens);

        if self.cfg.plateau_frequency_divisor == 0 {
            anyhow::bail!("plateau_frequency_divisor must be > 0");
        }

        let mut iteration = 0usize;
        let training_start = Instant::now();
        let plateau_floor = self.cfg.plateau_frequency_floor.max(self.cfg.min_frequency);
        let plateau_stop_enabled = self.cfg.plateau_stop_enabled && self.cfg.plateau_patience > 0;
        let mut plateau_streak = 0usize;
        let mut initial_frequency: Option<usize> = None;
        let mut metrics = TrainingMetrics {
            iterations: Vec::with_capacity(max_new_tokens.min(16_384)),
            total_duration: Duration::ZERO,
            stop_reason: TrainingStopReason::TargetVocabReached,
        };

        let mut pair_counts =
            compute_pair_counts(&working_sequences, &token_lengths, &allowed_lengths);
        let mut heap = BinaryHeap::with_capacity(pair_counts.len().max(1));
        for (&pair, &count) in pair_counts.iter() {
            if count >= self.cfg.min_frequency {
                heap.push(PairScore::new(pair, count));
            }
        }

        while merges.len() < max_new_tokens {
            if let Some(max_iters) = self.cfg.max_merge_iterations {
                if iteration >= max_iters {
                    metrics.stop_reason = TrainingStopReason::MaxIterationsReached;
                    if self.cfg.show_progress {
                        eprintln!(
                            "[trainer] Stopping: reached max merge iterations {}",
                            max_iters
                        );
                    }
                    break;
                }
            }
            let iteration_start = Instant::now();
            let best_candidate = loop {
                match heap.pop() {
                    Some(score) => {
                        let current = pair_counts.get(&score.pair).copied().unwrap_or(0);
                        if current == 0 || current != score.frequency {
                            continue;
                        }
                        if current < self.cfg.min_frequency {
                            continue;
                        }
                        break Some((score.pair, current));
                    }
                    None => break None,
                }
            };
            let Some((best_pair, frequency)) = best_candidate else {
                metrics.stop_reason = TrainingStopReason::NoEligiblePairs;
                if self.cfg.show_progress {
                    eprintln!(
                        "[trainer] Stopping: no pairs meet min_frequency {}",
                        self.cfg.min_frequency
                    );
                }
                break;
            };
            let distinct_pairs = pair_counts.len();

            if initial_frequency.is_none() {
                initial_frequency = Some(frequency);
            }
            let freq_low = frequency <= plateau_floor
                || initial_frequency.map_or(false, |init| {
                    (frequency as u128) * (self.cfg.plateau_frequency_divisor as u128)
                        <= init as u128
                });

            if plateau_stop_enabled {
                if freq_low {
                    plateau_streak += 1;
                    if plateau_streak >= self.cfg.plateau_patience {
                        metrics.stop_reason = TrainingStopReason::PlateauReached;
                        if self.cfg.show_progress {
                            eprintln!(
                                "[trainer] Stopping: plateau reached (freq {} for {} iterations)",
                                frequency, plateau_streak
                            );
                        }
                        break;
                    }
                } else {
                    plateau_streak = 0;
                }
            }

            let combined_len =
                token_lengths[best_pair.0 as usize] + token_lengths[best_pair.1 as usize];
            let mut new_token = Vec::with_capacity(combined_len);
            new_token.extend_from_slice(&token_bytes[best_pair.0 as usize]);
            new_token.extend_from_slice(&token_bytes[best_pair.1 as usize]);
            let new_token_id = token_bytes.len() as TokenId;

            let total_merges = apply_merge(
                &mut working_sequences,
                best_pair,
                new_token_id,
                combined_len,
                &mut pair_counts,
                &mut heap,
                &token_lengths,
                &allowed_lengths,
            );
            if total_merges == 0 {
                // No merges applied despite frequency counts; stop to avoid infinite loop.
                metrics.stop_reason = TrainingStopReason::NoEligiblePairs;
                if self.cfg.show_progress {
                    eprintln!(
                        "[trainer] Stopping: candidate pair {:?} produced no merges",
                        best_pair
                    );
                }
                break;
            }

            token_bytes.push(new_token);
            token_lengths.push(combined_len);
            merges.push(best_pair);

            iteration += 1;
            if self.cfg.show_progress {
                let elapsed_iter = iteration_start.elapsed();
                let elapsed_total = training_start.elapsed();
                eprintln!(
                    "[trainer] iter {:>6} freq {:>8} merges {:>8} distinct_pairs {:>8} vocab {:>8} iter_time {:>6.2?} total_time {:>6.2?}",
                    iteration,
                    frequency,
                    total_merges,
                    distinct_pairs,
                    base_vocab + iteration,
                    elapsed_iter,
                    elapsed_total
                );
            }

            metrics.iterations.push(IterationMetrics {
                iteration,
                best_frequency: frequency,
                merges_applied: total_merges,
                distinct_pairs,
                elapsed_iteration: iteration_start.elapsed(),
                elapsed_total: training_start.elapsed(),
                rss_kb: current_rss_kb(),
            });
        }

        if self.cfg.show_progress {
            eprintln!(
                "[trainer] completed {} merges in {:.2?}; final vocab size {} (base 256 + {})",
                merges.len(),
                training_start.elapsed(),
                token_bytes.len(),
                merges.len()
            );
        }

        metrics.total_duration = training_start.elapsed();

        Ok(TrainingArtifacts {
            output: TrainingOutput {
                token_bytes,
                merges,
                config: self.cfg.clone(),
            },
            metrics,
        })
    }
}

fn compute_pair_counts(
    sequences: &[Vec<TokenId>],
    token_lengths: &[usize],
    allowed_lengths: &[usize],
) -> FxHashMap<Pair, usize> {
    sequences
        .par_iter()
        .map(|sequence| {
            let mut local = FxHashMap::default();
            if sequence.len() < 2 {
                return local;
            }
            let mut prev = sequence[0];
            for &current in &sequence[1..] {
                let combined_len = token_lengths[prev as usize] + token_lengths[current as usize];
                if is_allowed_length(combined_len, allowed_lengths) {
                    *local.entry((prev, current)).or_insert(0) += 1;
                }
                prev = current;
            }
            local
        })
        .reduce(FxHashMap::default, |mut acc, local| {
            for (pair, count) in local {
                *acc.entry(pair).or_insert(0) += count;
            }
            acc
        })
}

#[derive(Default)]
struct MergeAdjustments {
    deltas: FxHashMap<Pair, i64>,
    merges: usize,
}

fn accumulate_delta(
    deltas: &mut FxHashMap<Pair, i64>,
    pair: Pair,
    combined_len: usize,
    allowed_lengths: &[usize],
    delta: i64,
) {
    if delta == 0 {
        return;
    }
    if !is_allowed_length(combined_len, allowed_lengths) {
        return;
    }
    *deltas.entry(pair).or_insert(0) += delta;
}

fn apply_delta(
    pair_counts: &mut FxHashMap<Pair, usize>,
    heap: &mut BinaryHeap<PairScore>,
    pair: Pair,
    delta: i64,
) {
    if delta == 0 {
        return;
    }
    match delta.cmp(&0) {
        Ordering::Greater => {
            let amount = delta as usize;
            let count = pair_counts.entry(pair).or_insert(0);
            *count += amount;
            heap.push(PairScore::new(pair, *count));
        }
        Ordering::Less => {
            let amount = (-delta) as usize;
            if let Entry::Occupied(mut occupied) = pair_counts.entry(pair) {
                let current = *occupied.get();
                debug_assert!(
                    amount <= current,
                    "delta underflow for pair {:?}: {} > {}",
                    pair,
                    amount,
                    current
                );
                let new_value = current.saturating_sub(amount);
                if new_value == 0 {
                    occupied.remove();
                } else {
                    *occupied.get_mut() = new_value;
                    heap.push(PairScore::new(pair, new_value));
                }
            }
        }
        Ordering::Equal => {}
    }
}

#[inline]
fn token_length_with_new(
    token: TokenId,
    token_lengths: &[usize],
    new_token: TokenId,
    new_token_len: usize,
) -> usize {
    if token == new_token {
        new_token_len
    } else {
        token_lengths[token as usize]
    }
}

fn process_sequence(
    sequence: &mut Vec<TokenId>,
    pair: Pair,
    new_token: TokenId,
    new_token_len: usize,
    token_lengths: &[usize],
    allowed_lengths: &[usize],
) -> MergeAdjustments {
    let mut result = MergeAdjustments::default();
    if sequence.len() < 2 {
        return result;
    }

    let mut read = 0usize;
    let mut write = 0usize;
    let original_len = sequence.len();
    let left_len = token_lengths[pair.0 as usize];
    let right_len = token_lengths[pair.1 as usize];

    while read < original_len {
        if read + 1 < original_len && sequence[read] == pair.0 && sequence[read + 1] == pair.1 {
            let prev_token = if write > 0 {
                Some(sequence[write - 1])
            } else {
                None
            };
            let next_token = if read + 2 < original_len {
                Some(sequence[read + 2])
            } else {
                None
            };

            if let Some(prev) = prev_token {
                let prev_len = token_length_with_new(prev, token_lengths, new_token, new_token_len);
                let combined = prev_len + left_len;
                accumulate_delta(
                    &mut result.deltas,
                    (prev, pair.0),
                    combined,
                    allowed_lengths,
                    -1,
                );
            }
            accumulate_delta(
                &mut result.deltas,
                pair,
                left_len + right_len,
                allowed_lengths,
                -1,
            );
            if let Some(next) = next_token {
                let next_len = token_length_with_new(next, token_lengths, new_token, new_token_len);
                let combined = right_len + next_len;
                accumulate_delta(
                    &mut result.deltas,
                    (pair.1, next),
                    combined,
                    allowed_lengths,
                    -1,
                );
            }

            sequence[write] = new_token;
            write += 1;
            read += 2;
            result.merges += 1;

            if let Some(prev) = prev_token {
                let prev_len = token_length_with_new(prev, token_lengths, new_token, new_token_len);
                let combined = prev_len + new_token_len;
                accumulate_delta(
                    &mut result.deltas,
                    (prev, new_token),
                    combined,
                    allowed_lengths,
                    1,
                );
            }
            if let Some(next) = next_token {
                let next_len = token_length_with_new(next, token_lengths, new_token, new_token_len);
                let combined = new_token_len + next_len;
                accumulate_delta(
                    &mut result.deltas,
                    (new_token, next),
                    combined,
                    allowed_lengths,
                    1,
                );
            }
        } else {
            if write != read {
                sequence[write] = sequence[read];
            }
            write += 1;
            read += 1;
        }
    }

    sequence.truncate(write);
    result
}

fn apply_merge(
    sequences: &mut [Vec<TokenId>],
    pair: Pair,
    new_token: TokenId,
    new_token_len: usize,
    pair_counts: &mut FxHashMap<Pair, usize>,
    heap: &mut BinaryHeap<PairScore>,
    token_lengths: &[usize],
    allowed_lengths: &[usize],
) -> usize {
    let aggregate = sequences
        .par_iter_mut()
        .map(|sequence| {
            process_sequence(
                sequence,
                pair,
                new_token,
                new_token_len,
                token_lengths,
                allowed_lengths,
            )
        })
        .reduce(
            || MergeAdjustments::default(),
            |mut acc, mut local| {
                acc.merges += local.merges;
                for (pair_key, delta) in local.deltas.drain() {
                    *acc.deltas.entry(pair_key).or_insert(0) += delta;
                }
                acc
            },
        );

    for (pair_key, delta) in aggregate.deltas {
        apply_delta(pair_counts, heap, pair_key, delta);
    }

    aggregate.merges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trainer_produces_merges() -> Result<()> {
        let sequences = vec![
            vec![0x10, 0x20, 0x10, 0x20, 0x10, 0x20],
            vec![0x10, 0x20, 0x30, 0x40],
            vec![0x10, 0x20, 0x10, 0x20],
        ];
        let mut cfg = TrainerConfig::default();
        cfg.target_vocab_size = 272; // 256 base + 7 specials + 9 merges
        cfg.min_frequency = 2;
        cfg.show_progress = false;
        let trainer = BytePairTrainer::new(cfg.clone());

        let artefacts = trainer.train_from_sequences(&sequences)?;
        let output = artefacts.output;
        assert!(!artefacts.metrics.iterations.is_empty());
        assert!(!output.merges.is_empty());
        assert!(output.token_bytes.len() >= 257);

        let tokenizer = output.build_tokenizer()?;
        assert_eq!(tokenizer.get_vocab_size(false), output.token_bytes.len());

        Ok(())
    }

    #[test]
    fn tokenizer_round_trip() -> Result<()> {
        let sequences = vec![vec![0xAA, 0xBB, 0xAA, 0xBB], vec![0xAA, 0xBB, 0xCC, 0xDD]];
        let mut cfg = TrainerConfig::default();
        cfg.target_vocab_size = 264;
        cfg.min_frequency = 2;
        cfg.show_progress = false;
        let trainer = BytePairTrainer::new(cfg.clone());
        let artefacts = trainer.train_from_sequences(&sequences)?;
        let output = artefacts.output;

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tokenizer.json");
        output.save_tokenizer(&path)?;

        let loaded = Tokenizer::from_file(path).map_err(|err| anyhow!(err))?;
        assert_eq!(loaded.get_vocab_size(false), output.token_bytes.len());
        Ok(())
    }
}
