use anyhow::{anyhow, Result};
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub struct SpecialTokenSet {
    pub start: &'static str,
    pub end: &'static str,
    pub sep: &'static str,
    pub cls: &'static str,
    pub pad: &'static str,
    pub mask: &'static str,
    pub unk: &'static str,
}

impl Default for SpecialTokenSet {
    fn default() -> Self {
        Self {
            start: "<|start|>",
            end: "<|end|>",
            sep: "<|sep|>",
            cls: "<|cls|>",
            pad: "<|pad|>",
            mask: "<|mask|>",
            unk: "<|unk|>",
        }
    }
}

// Build a TemplateProcessing post-processor that uses the special tokens present in the tokenizer.
// Single: <|start|> $A <|end|>
// Pair:   <|start|> $A <|sep|> $B <|end|>
pub fn build_template_processor(
    tokenizer: &Tokenizer,
    specials: &SpecialTokenSet,
) -> Result<PostProcessorWrapper> {
    let id = |tok: &str| tokenizer.token_to_id(tok).ok_or_else(|| anyhow!(
        "Token '{}' missing from tokenizer vocab (model or added)", tok
    ));

    let special_tokens = vec![
        (specials.cls.to_string(), id(specials.cls)?),
        (specials.sep.to_string(), id(specials.sep)?),
        (specials.start.to_string(), id(specials.start)?),
        (specials.end.to_string(), id(specials.end)?),
        (specials.pad.to_string(), id(specials.pad)?),
        (specials.mask.to_string(), id(specials.mask)?),
        (specials.unk.to_string(), id(specials.unk)?),
    ];

    let template = TemplateProcessing::builder()
        .try_single(format!("{} $A {}", specials.start, specials.end))
        .map_err(|e| anyhow!(e))?
        .try_pair(format!(
            "{} $A {} $B {}",
            specials.start, specials.sep, specials.end
        ))
        .map_err(|e| anyhow!(e))?
        .special_tokens(special_tokens)
        .build()
        .map_err(|e| anyhow!(e))?;

    Ok(template.into())
}
