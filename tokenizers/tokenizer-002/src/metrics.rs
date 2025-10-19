use std::fs;
use std::time::Duration;

/// Per-iteration telemetry captured during training.
#[derive(Debug, Clone)]
pub struct IterationMetrics {
    pub iteration: usize,
    pub best_frequency: usize,
    pub merges_applied: usize,
    pub distinct_pairs: usize,
    pub elapsed_iteration: Duration,
    pub elapsed_total: Duration,
    pub rss_kb: usize,
}

/// Reason the trainer stopped executing merges.
#[derive(Debug, Clone)]
pub enum TrainingStopReason {
    TargetVocabReached,
    NoEligiblePairs,
    PlateauReached,
    MaxIterationsReached,
}

/// Aggregate metrics for a full training run.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub iterations: Vec<IterationMetrics>,
    pub total_duration: Duration,
    pub stop_reason: TrainingStopReason,
}

impl TrainingMetrics {
    pub fn iteration_count(&self) -> usize {
        self.iterations.len()
    }

    pub fn summary(&self) -> String {
        let (last_freq, last_distinct) = self
            .iterations
            .last()
            .map(|iter| (iter.best_frequency, iter.distinct_pairs))
            .unwrap_or((0, 0));
        format!(
            "iterations={}, stop={:?}, total={:.2?}, last_freq={}, remaining_pairs={}",
            self.iteration_count(),
            self.stop_reason,
            self.total_duration,
            last_freq,
            last_distinct
        )
    }
}

/// Returns the current resident set size (KB) if available on Linux, otherwise 0.
pub fn current_rss_kb() -> usize {
    if let Ok(statm) = fs::read_to_string("/proc/self/statm") {
        if let Some(pages_str) = statm.split_whitespace().nth(1) {
            if let Ok(pages) = pages_str.parse::<usize>() {
                return pages * page_size_kb();
            }
        }
    }
    0
}

fn page_size_kb() -> usize {
    static mut PAGE_SIZE_KB: usize = 0;
    unsafe {
        if PAGE_SIZE_KB == 0 {
            PAGE_SIZE_KB = (libc::sysconf(libc::_SC_PAGESIZE) as usize) / 1024;
        }
        PAGE_SIZE_KB
    }
}
