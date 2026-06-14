# 89 Embodied Attention Budgeting

Submission-hardening version: v4

Terminal decision: **KILL_ARCHIVE** for ICLR main conference.

This repository contains a reproducible local evidence audit for the research bet:

> Allocate perception and action attention by physical consequence rather than token salience.

The v4 rebuild replaces the template scaffold with a deterministic attention-budget benchmark over four manipulation tasks, five shifts, eight methods, ablations, stress sweeps, and negative cases.

## Why This Is Archived

- On the combined hard-shift split, `embodied_consequence_budgeting` reaches `0.69303 +/- 0.02890` task success.
- The strongest success baseline, `risk_aware_attention`, reaches `0.67602 +/- 0.02113`.
- The paired task-success difference is only `0.01701 +/- 0.04077`.
- `minus_action_critical_contact_map` improves task success to `0.70493`, contradicting the full mechanism.
- At maximum combined stress, `risk_aware_attention` beats the proposed method on task success.
- The evidence is local and synthetic, not hardware or accepted high-fidelity benchmark validation.

## Reproduce

```powershell
python src\run_experiment.py
```

The runner writes:

- `results/rollouts.csv`
- `results/raw_seed_metrics.csv`
- `results/metrics.csv`
- `results/pairwise_stats.csv`
- `results/ablation_rollouts.csv`
- `results/ablation_seed_metrics.csv`
- `results/ablation_metrics.csv`
- `results/stress_sweep_raw.csv`
- `results/stress_sweep.csv`
- `results/negative_cases.csv`
- `results/summary.txt`
- `figures/attention_budget_*.png`

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/89.pdf`
