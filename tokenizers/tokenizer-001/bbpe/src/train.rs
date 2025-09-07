use crate::ingest::CorpusIter;
use crate::post::{build_template_processor, SpecialTokenSet};
use crate::util::{is_power_of_two, next_power_of_two};
use anyhow::{anyhow, Result};
use std::collections::HashSet;
use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::tokenizer::AddedToken;
use tokenizers::Tokenizer;
use tokenizers::{Model, Trainer};

#[derive(Clone, Debug)]
pub struct TrainerConfig {
    pub vocab_size: usize,         // default: 32768
    pub min_frequency: u64,        // default: 1024
    pub show_progress: bool,
    pub with_template: bool,       // add post-processor
    pub reserved_count: usize,     // default: 128
    pub pad_to_power_of_two: bool, // default: true
    pub max_token_length: usize,   // default: 32
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_768,
            min_frequency: 1024,
            show_progress: true,
            with_template: true,
            reserved_count: 128,
            pad_to_power_of_two: true,
            max_token_length: 32,
        }
    }
}

fn initial_alphabet_256() -> HashSet<char> {
    (0u16..=255)
        .map(|b| char::from_u32(b as u32).unwrap())
        .collect::<HashSet<char>>()
}

fn make_special_tokens(base: &SpecialTokenSet, reserved: usize) -> Vec<AddedToken> {
    let mut tokens = vec![
        AddedToken::from(base.start, true),
        AddedToken::from(base.end, true),
        AddedToken::from(base.sep, true),
        AddedToken::from(base.cls, true),
        AddedToken::from(base.pad, true),
        AddedToken::from(base.mask, true),
        AddedToken::from(base.unk, true),
    ];
    for i in 0..reserved {
        tokens.push(AddedToken::from(format!("<|reserved:{i}|>"), true));
    }
    tokens
}

pub fn train_tokenizer(
    iter: CorpusIter,
    trainer_cfg: &TrainerConfig,
) -> Result<Tokenizer> {
    // Trainer setup
    // Special tokens (base + reserved)
    let specials = SpecialTokenSet::default();
    let special_tokens = make_special_tokens(&specials, trainer_cfg.reserved_count);

    // Compute the target vocabulary size for the trainer.
    // Semantics: `trainer_cfg.vocab_size` is the total target size (including specials).
    // If requested, pad this target to the next power-of-two so the model ends up aligned.
    let mut target_vocab_size = trainer_cfg.vocab_size;
    if trainer_cfg.pad_to_power_of_two {
        target_vocab_size = next_power_of_two(target_vocab_size);
    }

    let mut builder = BpeTrainer::builder()
        .vocab_size(target_vocab_size)
        .min_frequency(trainer_cfg.min_frequency)
        .show_progress(trainer_cfg.show_progress)
        .initial_alphabet(initial_alphabet_256())
        .max_token_length(Some(trainer_cfg.max_token_length))
        .special_tokens(special_tokens.clone());
    let trainer: BpeTrainer = builder.build();
    let mut trainer = trainer;

    // Feed the iterator: we treat each yielded sequence as a single "word" for BPE training
    // Identity process: one string per yielded sequence.
    trainer
        .feed(iter, |s| Ok(vec![s.to_owned()]))
        .map_err(|e| anyhow!(e))?;

    // Train the model
    let mut bpe = BPE::default();
    let _returned_specials = trainer.train(&mut bpe).map_err(|e| anyhow!(e))?;

    // Assemble tokenizer
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.add_special_tokens(&special_tokens);

    if trainer_cfg.with_template {
        let proc = build_template_processor(&tokenizer, &specials)?;
        tokenizer.with_post_processor(Some(proc));
    }

    Ok(tokenizer)
}

// Ensure the final vocabulary size (model + added) is padded to the next power of two by
// appending more reserved tokens if needed.
pub fn pad_vocab_to_power_of_two(tokenizer: &mut Tokenizer) -> Result<()> {
    // Count only genuinely new added tokens (with IDs beyond the model's vocab).
    let model_size = tokenizer.get_model().get_vocab_size();
    let added_decoder = tokenizer
        .get_added_vocabulary()
        .get_added_tokens_decoder();
    let extra_added = added_decoder.keys().filter(|&&id| id as usize >= model_size).count();

    let total_unique = model_size + extra_added;
    if is_power_of_two(total_unique) {
        return Ok(());
    }
    let target = next_power_of_two(total_unique);

    // Find the next available reserved index label
    let mut i = 0usize;
    while tokenizer
        .token_to_id(&format!("<|reserved:{i}|>"))
        .is_some()
    {
        i += 1;
    }
    let need = target - total_unique;
    let mut to_add = Vec::with_capacity(need);
    for j in 0..need {
        to_add.push(AddedToken::from(format!("<|reserved:{}|>", i + j), true));
    }
    tokenizer.add_special_tokens(&to_add);
    Ok(())
}
