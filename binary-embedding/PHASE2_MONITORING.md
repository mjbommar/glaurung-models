# Phase 2 Quick Monitoring Guide

## Quick Commands

### Check Current Step
```bash
tail -1 micro_phase2_contrastive_*/mlm_training_log.jsonl | python3 -c "import json, sys; d=json.loads(sys.stdin.read()); print(f'Step: {d[\"step\"]:,}')"
```

### Get Latest Assessment
```bash
grep '"event": "assessment"' micro_phase2_contrastive_*/mlm_training_log.jsonl | tail -1 | python3 -c "
import json, sys
a = json.loads(sys.stdin.read())
step = a['step']
avg = a['assessment']['assessment/average_score'] * 100
embed = a['assessment']['assessment/embedding_quality_score'] * 100
pattern = a['assessment']['assessment/pattern_learning_score'] * 100
header = a['assessment']['assessment/file_header_score'] * 100
context = a['assessment']['assessment/context_understanding_score'] * 100

print(f'Step {step//1000}k:')
print(f'  Overall:  {avg:5.1f}% (baseline: 38.3%)')
print(f'  Embedding: {embed:5.1f}% (baseline: 24.9%) ← PRIMARY GOAL')
print(f'  Pattern:  {pattern:5.1f}% (baseline: 63.3%)')
print(f'  Header:   {header:5.1f}% (baseline: 25.0%)')
print(f'  Context:  {context:5.1f}% (baseline: 40.0%)')
"
```

### Check All Assessments
```bash
grep '"event": "assessment"' micro_phase2_contrastive_*/mlm_training_log.jsonl | python3 -c "
import json, sys

print('Phase 2 Assessment Progression:')
print('=' * 80)
print('Step    Avg     Embed   Pattern Header  Context   vs Phase1')
print('=' * 80)

for line in sys.stdin:
    a = json.loads(line)
    step = a['step']
    avg = a['assessment']['assessment/average_score'] * 100
    embed = a['assessment']['assessment/embedding_quality_score'] * 100
    pattern = a['assessment']['assessment/pattern_learning_score'] * 100
    header = a['assessment']['assessment/file_header_score'] * 100
    context = a['assessment']['assessment/context_understanding_score'] * 100

    # Compare to Phase 1 baseline (38.3% avg)
    delta = avg - 38.3
    trend = '↗' if delta > 0 else '↘' if delta < 0 else '→'

    print(f'{step//1000:3d}k:  {avg:5.1f}%  {embed:5.1f}%  {pattern:5.1f}%  {header:5.1f}%  {context:5.1f}%   {trend} {delta:+.1f}%')
"
```

### Check Loss Trends
```bash
grep '"event": "train_step"' micro_phase2_contrastive_*/mlm_training_log.jsonl | python3 -c "
import json, sys

print('Loss Trends (every 10k steps):')
print('=' * 70)
print('Step    MLM Loss  View Loss  SameFile Loss  LR')
print('=' * 70)

for line in sys.stdin:
    d = json.loads(line)
    step = d['step']

    if step % 10000 == 0:
        mlm = d['metrics']['mlm_loss']
        view = d['metrics']['view_loss']
        sf = d['metrics']['same_file_loss']
        lr = d['optimizer']['learning_rate']

        print(f'{step//1000:3d}k:  {mlm:8.4f}  {view:9.4f}  {sf:13.4f}  {lr:.6f}')
"
```

## Success Indicators

### ✅ GOOD Signs (Keep Going)
```
Embedding quality:  25% → 28% → 32% → 35%  (IMPROVING)
Pattern learning:   63% → 62% → 61% → 60%  (STABLE)
Overall average:    38% → 40% → 42% → 44%  (IMPROVING)
View loss:          2.5 → 2.0 → 1.5 → 1.2  (DECREASING)
SameFile loss:      2.8 → 2.3 → 1.8 → 1.4  (DECREASING)
```

### ⚠️ WARNING Signs (Needs Adjustment)
```
Embedding quality:  25% → 26% → 26% → 26%  (STALLED)
  → Increase contrastive weights: 0.5 → 0.75

Pattern learning:   63% → 58% → 54% → 50%  (DROPPING TOO FAST)
  → Decrease contrastive weights: 0.5 → 0.3
  → Increase MLM weight: 1.0 → 1.5
```

### 💀 FAILURE Signs (Stop & Analyze)
```
Embedding quality:  25% → 23% → 21% → 19%  (GETTING WORSE)
  → Wrong checkpoint? Check init path
  → LR too high? Reduce to 5e-6

Pattern learning:   63% → 52% → 42% → 35%  (CATASTROPHIC FORGETTING)
  → Stop immediately
  → Contrastive weights too high
  → Start over with weights 0.3 instead of 0.5

All losses:         INCREASING across the board
  → Overfitting, LR too high
  → Stop and restart with LR 5e-6
```

## Milestone Checks

### @ 10k Steps (Expected: ~2.5 hours)
```bash
# Run this command
grep '"event": "assessment"' micro_phase2_contrastive_*/mlm_training_log.jsonl | grep '"step": 10' | python3 -c "
import json, sys
a = json.loads(sys.stdin.read())
embed = a['assessment']['assessment/embedding_quality_score'] * 100
pattern = a['assessment']['assessment/pattern_learning_score'] * 100

print(f'10k Checkpoint:')
print(f'  Embedding: {embed:.1f}% (target: 26-28%, baseline: 24.9%)')
print(f'  Pattern:   {pattern:.1f}% (target: 62-64%, baseline: 63.3%)')

if embed >= 26 and pattern >= 60:
    print('  ✅ ON TRACK - Continue')
elif embed < 26 and pattern >= 60:
    print('  ⚠️  Embedding not improving - Consider increasing contrastive weights')
elif embed >= 26 and pattern < 60:
    print('  ⚠️  Pattern dropping - Consider decreasing contrastive weights')
else:
    print('  💀 BOTH PROBLEMS - Check configuration and logs')
"
```

### @ 25k Steps (Expected: ~6 hours)
```bash
# Target: Embedding 29-32%, Pattern 60-62%, Overall 41-42%
grep '"event": "assessment"' micro_phase2_contrastive_*/mlm_training_log.jsonl | grep '"step": 25' | python3 -c "
import json, sys
a = json.loads(sys.stdin.read())
avg = a['assessment']['assessment/average_score'] * 100
embed = a['assessment']['assessment/embedding_quality_score'] * 100
pattern = a['assessment']['assessment/pattern_learning_score'] * 100

print(f'25k Checkpoint:')
print(f'  Overall:   {avg:.1f}% (target: 41-42%, baseline: 38.3%)')
print(f'  Embedding: {embed:.1f}% (target: 29-32%, baseline: 24.9%)')
print(f'  Pattern:   {pattern:.1f}% (target: 60-62%, baseline: 63.3%)')

if avg >= 41 and embed >= 29:
    print('  ✅ EXCELLENT - On track for 45%+ final')
elif avg >= 40:
    print('  ✅ GOOD - On track for 43-45% final')
elif avg >= 39:
    print('  ⚠️  Slight improvement only - May need adjustments')
else:
    print('  💀 Not improving - Something is wrong')
"
```

### @ 50k Steps (Expected: ~12 hours)
```bash
# Target: Embedding 33-35%, Pattern 59-61%, Overall 43-45%
# This is likely near peak
grep '"event": "assessment"' micro_phase2_contrastive_*/mlm_training_log.jsonl | grep '"step": 50' | python3 -c "
import json, sys
a = json.loads(sys.stdin.read())
avg = a['assessment']['assessment/average_score'] * 100
embed = a['assessment']['assessment/embedding_quality_score'] * 100

print(f'50k Checkpoint:')
print(f'  Overall:   {avg:.1f}%')
print(f'  Embedding: {embed:.1f}%')

if avg >= 44 and embed >= 33:
    print('  ✅ EXCELLENT SUCCESS - Target achieved!')
elif avg >= 42:
    print('  ✅ GOOD SUCCESS - Solid improvement')
elif avg >= 40:
    print('  ✅ MODERATE SUCCESS - Some improvement')
else:
    print('  💀 FAILURE - Phase 2 did not help')
"
```

## Decision Points

### @ 10k - Continue or Adjust?
- ✅ Embed improving (26-28%) → **Continue**
- ⚠️ Embed stalled (24-26%) → **Increase contrastive weights to 0.75**
- 💀 Embed worse (<24%) → **Stop, investigate**

### @ 25k - Stay course or optimize?
- ✅ Avg 41%+, Embed 29%+ → **Continue to end**
- ⚠️ Avg 39-41%, Embed 27-29% → **Continue but may need longer**
- 💀 Avg <39% → **Phase 2 not working, analyze why**

### @ 50k - Continue or stop?
- ✅ Avg 44%+, Embed 33%+ → **Continue to 75k (may improve further)**
- ⚠️ Avg 42-44% → **This might be peak, but continue to 75k**
- 💀 Scores declining from 40k → **Peak was at 40k, stop now**

### @ 75k - Which checkpoint to keep?
```bash
# Find best checkpoint by embedding quality
grep '"event": "assessment"' micro_phase2_contrastive_*/mlm_training_log.jsonl | python3 -c "
import json, sys

best_step = 0
best_embed = 0
best_avg = 0

for line in sys.stdin:
    a = json.loads(line)
    step = a['step']
    embed = a['assessment']['assessment/embedding_quality_score'] * 100
    avg = a['assessment']['assessment/average_score'] * 100

    if embed > best_embed:
        best_step = step
        best_embed = embed
        best_avg = avg

print(f'Best checkpoint: {best_step//1000}k')
print(f'  Embedding: {best_embed:.1f}%')
print(f'  Overall:   {best_avg:.1f}%')
print(f'')
print(f'Copy to final:')
print(f'  cp -r micro_phase2_contrastive_*/checkpoint-{best_step}/ models/micro-phase2-best/')
"
```

## Live Monitoring

### Watch Logs (Auto-refresh)
```bash
watch -n 10 "tail -1 micro_phase2_contrastive_*/mlm_training_log.jsonl | python3 -c \"import json, sys; d=json.loads(sys.stdin.read()); print(f'Step {d[\\\"step\\\"]}: loss={d[\\\"loss\\\"][\\\"current\\\"]:.3f}, lr={d[\\\"optimizer\\\"][\\\"learning_rate\\\"]:.6f}')\""
```

### WandB
```
Project: glaurung-micro-phase2
URL: https://wandb.ai/<your-username>/glaurung-micro-phase2
```

Key charts to watch:
1. `assessment/embedding_quality_score` - Should increase
2. `assessment/pattern_learning_score` - Should stay stable
3. `assessment/average_score` - Should increase
4. `train/view_contrastive_loss` - Should decrease
5. `train/same_file_contrastive_loss` - Should decrease
6. `train/mlm_loss` - Should stay relatively stable

## Emergency Stop

If things go wrong:
```bash
# Find the training process
ps aux | grep binary_embedding

# Kill it
kill <PID>

# Or Ctrl+C in the terminal
```

## Compare to Phase 1

### After Phase 2 Completes
```bash
echo "Phase 1 (v6 @ 150k):"
echo "  Overall:   38.3%"
echo "  Embedding: 24.9%"
echo "  Pattern:   63.3%"
echo ""
echo "Phase 2 (best checkpoint):"
grep '"event": "assessment"' micro_phase2_contrastive_*/mlm_training_log.jsonl | tail -5
```

### Success Thresholds
- **Minimum:** Overall 40%+, Embedding 30%+ (Phase 2 helped a bit)
- **Target:** Overall 44%+, Embedding 35%+ (Phase 2 worked well)
- **Excellent:** Overall 46%+, Embedding 38%+ (Phase 2 exceeded expectations)

## Next Steps After Phase 2

### If Successful (44%+ overall)
1. Archive Phase 2 run
2. Save best checkpoint to `models/micro-phase2-best/`
3. Consider Phase 3: Extended fine-tuning or different objectives

### If Moderate (40-44% overall)
1. Analyze what improved and what didn't
2. Consider Phase 2b with adjusted weights
3. May be good enough for deployment

### If Failed (<40% overall)
1. Analyze logs to understand why
2. Check if wrong checkpoint loaded
3. May need different approach (architectural changes?)
