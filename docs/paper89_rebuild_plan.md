# Paper 89 Rebuild Plan

Last update: 2026-06-14 13:37:20 +01:00

## Target Claim

Embodied policies should allocate perception and action attention by physical consequence rather than token salience. A useful attention budget must spend compute and sensing on contacts, hazards, occlusions, and action-critical regions that change the outcome, not merely on visually salient or linguistically prominent tokens.

## Hostile Prior-Work Pressure

The local hostile set includes embodied foundation models, robot foundation-model surveys, physical-risk control, embodied VLA systems, and work warning that attention weights often capture correlations rather than causal task structure. The v4 rebuild must not claim novelty from generic attention, token pruning, active perception, uncertainty, or compute adaptation. It must test whether physical consequence changes budget allocation and downstream robot behavior.

## Evidence To Build

Replace the shared probability scaffold with a deterministic local attention-budget benchmark. Each episode should generate candidate perceptual/action tokens with token salience, uncertainty, physical consequence, hazard potential, occlusion, compute cost, and whether the token is causally critical for the task. Methods allocate a fixed attention budget, then act with residual risk from missed critical tokens and wasted budget.

### Tasks

- cluttered pick-and-place with distractor objects and hidden support surfaces.
- mobile manipulation around fragile obstacles.
- tool-use sequencing with low-salience but high-consequence contact points.
- deformable cable routing with latent snag and occlusion.

### Splits

- `balanced_nominal`
- `visual_distractor_shift`
- `physical_consequence_shift`
- `tight_compute_budget`
- `combined_hard_shift`

### Methods

- `uniform_attention_budget`
- `token_salience_transformer`
- `uncertainty_attention`
- `active_perception_value`
- `compute_adaptive_policy`
- `risk_aware_attention`
- `embodied_consequence_budgeting` (proposed)
- `oracle_consequence_attention`

### Metrics

- task success.
- critical-event recall.
- wasted attention rate.
- compute cost.
- latency violation.
- safety violation.
- physical regret versus oracle.
- calibration error.

### Ablations

- full embodied-consequence budgeting.
- minus physical-consequence estimator.
- minus action-critical contact map.
- minus compute-latency constraint.
- minus safety/hazard term.
- token-salience-only budget.
- uncertainty-only budget.

### Stress Tests

- visual distractor load.
- occlusion.
- compute budget tightening.
- latency penalty.
- hidden physical consequence shift.
- combined stress.

### Terminal Gate

Mark `STRONG_REVISE` only if the proposed method beats the strongest non-oracle baseline on combined hard-shift task success, critical-event recall, wasted attention, and safety without excessive compute/latency, while ablations degrade the mechanism. Otherwise mark `KILL_ARCHIVE`.

Even a `STRONG_REVISE` outcome is not ICLR-main ready without real robot hardware, external benchmark validation, or an implemented foundation-model attention module.
