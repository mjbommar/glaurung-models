use crate::mapping::bytes_to_latin1_string;
use std::collections::HashMap;
use tokenizers::AddedToken;

/// Common binary patterns that appear frequently in executables
#[derive(Clone, Debug)]
pub enum PatternType {
    Null,     // 0x00 - padding, uninitialized memory
    NopX86,   // 0x90 - x86/x64 NOP instruction
    Int3,     // 0xCC - x86 breakpoint/software interrupt
    FfPad,    // 0xFF - flash erase, uninitialized EEPROM
    NopRisc,  // 0x00000000 - 4-byte NOP for RISC architectures (MIPS, RISC-V)
    Space,    // 0x20 - spaces in string sections
    Dot,      // 0x2E - periods (common in paths/strings)
    Slash,    // 0x2F - forward slash (paths)
    One,      // 0x01 - common in data sections
    Zero,     // 0x30 - ASCII '0' character
}

impl PatternType {
    /// Get the byte(s) that make up this pattern
    fn bytes(&self) -> Vec<u8> {
        match self {
            PatternType::Null => vec![0x00],
            PatternType::NopX86 => vec![0x90],
            PatternType::Int3 => vec![0xCC],
            PatternType::FfPad => vec![0xFF],
            PatternType::NopRisc => vec![0x00, 0x00, 0x00, 0x00],
            PatternType::Space => vec![0x20],
            PatternType::Dot => vec![0x2E],
            PatternType::Slash => vec![0x2F],
            PatternType::One => vec![0x01],
            PatternType::Zero => vec![0x30],
        }
    }
    
    /// Get the token name for this pattern at a given length
    fn token_name(&self, length: usize) -> String {
        match self {
            PatternType::Null => format!("<|null_{}|>", length),
            PatternType::NopX86 => format!("<|nop_x86_{}|>", length),
            PatternType::Int3 => format!("<|int3_{}|>", length),
            PatternType::FfPad => format!("<|ff_{}|>", length),
            PatternType::NopRisc => format!("<|nop_risc_{}|>", length / 4), // count of 4-byte NOPs
            PatternType::Space => format!("<|space_{}|>", length),
            PatternType::Dot => format!("<|dot_{}|>", length),
            PatternType::Slash => format!("<|slash_{}|>", length),
            PatternType::One => format!("<|one_{}|>", length),
            PatternType::Zero => format!("<|zero_{}|>", length),
        }
    }
    
    /// Check if this pattern type can have the given length
    fn valid_length(&self, length: usize) -> bool {
        match self {
            PatternType::NopRisc => length % 4 == 0, // Must be multiple of 4
            _ => true, // Single-byte patterns work with any length
        }
    }
}

/// Configuration for pattern sequence generation
#[derive(Clone, Debug)]
pub struct PatternConfig {
    pub enabled: bool,
    pub min_power: u8,  // Minimum power of 2 (e.g., 2 means 2^2 = 4 bytes)
    pub max_power: u8,  // Maximum power of 2 (exclusive)
    pub patterns: Vec<PatternType>,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_power: 2,   // Start at 4 bytes
            max_power: 11,  // Up to 1024 bytes (2^10)
            patterns: vec![
                PatternType::Null,
                PatternType::NopX86,
                PatternType::Int3,
                PatternType::FfPad,
            ],
        }
    }
}

/// Generate pattern tokens based on configuration
pub fn generate_pattern_tokens(config: &PatternConfig) -> Vec<AddedToken> {
    if !config.enabled {
        return Vec::new();
    }
    
    let mut tokens = Vec::new();
    
    for pattern in &config.patterns {
        for power in config.min_power..config.max_power {
            let length = 2_usize.pow(power as u32);
            
            // Skip invalid lengths for this pattern type
            if !pattern.valid_length(length) {
                continue;
            }
            
            // Create the token name
            let token_name = pattern.token_name(length);
            
            // Add as a regular token (not special)
            tokens.push(AddedToken::from(token_name, false));
        }
    }
    
    // Add RISC NOP patterns if enabled (4-byte aligned)
    if config.patterns.iter().any(|p| matches!(p, PatternType::NopRisc)) {
        // Generate for multiples of 4 bytes within the power range
        for power in config.min_power..config.max_power {
            let length = 2_usize.pow(power as u32);
            if length >= 4 && length % 4 == 0 {
                let token_name = PatternType::NopRisc.token_name(length);
                tokens.push(AddedToken::from(token_name, false));
            }
        }
    }
    
    tokens
}

/// Generate the actual pattern strings for the tokenizer vocabulary
/// These will be added to the tokenizer's vocabulary after BPE training
pub fn generate_pattern_strings(config: &PatternConfig) -> HashMap<String, String> {
    if !config.enabled {
        return HashMap::new();
    }
    
    let mut patterns = HashMap::new();
    
    for pattern in &config.patterns {
        let base_bytes = pattern.bytes();
        
        for power in config.min_power..config.max_power {
            let length = 2_usize.pow(power as u32);
            
            // Skip invalid lengths
            if !pattern.valid_length(length) {
                continue;
            }
            
            // Generate the repeated pattern
            let mut pattern_bytes = Vec::with_capacity(length);
            let repeat_count = length / base_bytes.len();
            
            for _ in 0..repeat_count {
                pattern_bytes.extend_from_slice(&base_bytes);
            }
            
            // Convert to latin-1 string
            let pattern_string = bytes_to_latin1_string(&pattern_bytes);
            let token_name = pattern.token_name(length);
            
            patterns.insert(token_name, pattern_string);
        }
    }
    
    patterns
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pattern_generation() {
        let config = PatternConfig {
            enabled: true,
            min_power: 2,  // Start at 4
            max_power: 4,  // Up to 8 (exclusive)
            patterns: vec![PatternType::Null, PatternType::NopX86],
        };
        
        let tokens = generate_pattern_tokens(&config);
        assert_eq!(tokens.len(), 4); // null_4, null_8, nop_x86_4, nop_x86_8
        
        let patterns = generate_pattern_strings(&config);
        assert_eq!(patterns.len(), 4);
        
        // Check null_4 is 4 null bytes
        let null_4 = patterns.get("<|null_4|>").unwrap();
        assert_eq!(null_4.len(), 4);
        assert_eq!(null_4.as_bytes(), &[0, 0, 0, 0]);
        
        // Check nop_x86_8 is 8 NOP bytes
        let nop_8 = patterns.get("<|nop_x86_8|>").unwrap();
        assert_eq!(nop_8.len(), 8);
        assert_eq!(nop_8.as_bytes(), &[0x90; 8]);
    }
    
    #[test]
    fn test_risc_nop_alignment() {
        let config = PatternConfig {
            enabled: true,
            min_power: 2,  // Start at 4
            max_power: 5,  // Up to 16 (exclusive)
            patterns: vec![PatternType::NopRisc],
        };
        
        let patterns = generate_pattern_strings(&config);
        
        // Should have 4, 8, 16 (all multiples of 4)
        assert!(patterns.contains_key("<|nop_risc_1|>")); // 4 bytes = 1 RISC NOP
        assert!(patterns.contains_key("<|nop_risc_2|>")); // 8 bytes = 2 RISC NOPs
        assert!(patterns.contains_key("<|nop_risc_4|>")); // 16 bytes = 4 RISC NOPs
    }
}