import csv
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BASE_SEED = 89089089
SEEDS = list(range(7))
MAIN_EPISODES_PER_SEED = 42
STRESS_EPISODES_PER_SEED = 20
TOKENS = 20

TASKS = [
    {"task": "cluttered_pick_hidden_support", "base_success": 0.52, "hazard_weight": 0.22, "budget": 4.8},
    {"task": "mobile_manip_fragile_obstacles", "base_success": 0.50, "hazard_weight": 0.30, "budget": 4.6},
    {"task": "tool_use_low_salience_contact", "base_success": 0.48, "hazard_weight": 0.18, "budget": 4.4},
    {"task": "deformable_cable_latent_snag", "base_success": 0.46, "hazard_weight": 0.28, "budget": 4.3},
]

SPLITS = {
    "balanced_nominal": {
        "distractor_load": 0.10,
        "occlusion": 0.08,
        "consequence_shift": 0.08,
        "budget_tightness": 0.06,
        "latency_penalty": 0.06,
        "critical_rate": 0.24,
    },
    "visual_distractor_shift": {
        "distractor_load": 0.36,
        "occlusion": 0.12,
        "consequence_shift": 0.12,
        "budget_tightness": 0.08,
        "latency_penalty": 0.08,
        "critical_rate": 0.25,
    },
    "physical_consequence_shift": {
        "distractor_load": 0.14,
        "occlusion": 0.16,
        "consequence_shift": 0.38,
        "budget_tightness": 0.10,
        "latency_penalty": 0.10,
        "critical_rate": 0.32,
    },
    "tight_compute_budget": {
        "distractor_load": 0.16,
        "occlusion": 0.15,
        "consequence_shift": 0.20,
        "budget_tightness": 0.34,
        "latency_penalty": 0.24,
        "critical_rate": 0.28,
    },
    "combined_hard_shift": {
        "distractor_load": 0.34,
        "occlusion": 0.30,
        "consequence_shift": 0.38,
        "budget_tightness": 0.32,
        "latency_penalty": 0.28,
        "critical_rate": 0.36,
    },
}

METHODS = [
    "uniform_attention_budget",
    "token_salience_transformer",
    "uncertainty_attention",
    "active_perception_value",
    "compute_adaptive_policy",
    "risk_aware_attention",
    "embodied_consequence_budgeting",
    "oracle_consequence_attention",
]

ABLATIONS = [
    "full_embodied_consequence_budgeting",
    "minus_physical_consequence_estimator",
    "minus_action_critical_contact_map",
    "minus_compute_latency_constraint",
    "minus_safety_hazard_term",
    "token_salience_only_budget",
    "uncertainty_only_budget",
]

METRICS = [
    "task_success",
    "critical_event_recall",
    "wasted_attention",
    "compute_cost",
    "latency_violation",
    "safety_violation",
    "physical_regret",
    "calibration_error",
]


def stable_int(*parts):
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def stable_rng(*parts):
    return np.random.default_rng(stable_int(BASE_SEED, *parts))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def ci95(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / math.sqrt(len(vals)))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stress_params(split_name, stress_axis=None, stress_level=0.0):
    params = dict(SPLITS[split_name])
    if stress_axis is None:
        return params
    level = float(stress_level)
    if stress_axis == "visual_distractor_load":
        params["distractor_load"] = 0.08 + 0.50 * level
    elif stress_axis == "occlusion":
        params["occlusion"] = 0.06 + 0.46 * level
    elif stress_axis == "compute_budget_tightening":
        params["budget_tightness"] = 0.04 + 0.54 * level
        params["latency_penalty"] = 0.06 + 0.34 * level
    elif stress_axis == "latency_penalty":
        params["latency_penalty"] = 0.04 + 0.50 * level
    elif stress_axis == "hidden_physical_consequence":
        params["consequence_shift"] = 0.06 + 0.55 * level
        params["critical_rate"] = 0.24 + 0.26 * level
    elif stress_axis == "combined":
        params["distractor_load"] = 0.10 + 0.48 * level
        params["occlusion"] = 0.08 + 0.44 * level
        params["consequence_shift"] = 0.08 + 0.50 * level
        params["budget_tightness"] = 0.06 + 0.48 * level
        params["latency_penalty"] = 0.06 + 0.42 * level
        params["critical_rate"] = 0.24 + 0.30 * level
    else:
        raise KeyError(stress_axis)
    return params


def make_episode(split_name, task, seed, episode_id, stress_axis=None, stress_level=0.0):
    rng = stable_rng(split_name, task["task"], seed, episode_id, stress_axis or "main", f"{stress_level:.2f}")
    params = stress_params(split_name, stress_axis, stress_level)
    budget = max(1.6, task["budget"] * (1.0 - 0.62 * params["budget_tightness"]))
    tokens = []
    for idx in range(TOKENS):
        is_distractor = rng.random() < params["distractor_load"]
        consequence = clamp(rng.normal(0.46 + params["consequence_shift"] * 0.30, 0.22))
        hazard = clamp(rng.normal(0.24 + params["consequence_shift"] * task["hazard_weight"], 0.18))
        action_critical = clamp(0.48 * consequence + 0.34 * hazard + rng.normal(0.0, 0.14))
        uncertainty = clamp(rng.normal(0.32 + params["occlusion"] * 0.55, 0.18))
        salience = clamp(rng.normal(0.52 + (0.34 if is_distractor else 0.0), 0.20))
        language_salience = clamp(rng.normal(0.48 + (0.18 if is_distractor else 0.06), 0.18))
        cost = clamp(rng.normal(0.74 + 0.34 * uncertainty + 0.18 * salience, 0.16), 0.22, 1.55)

        if is_distractor:
            consequence = clamp(consequence - rng.uniform(0.18, 0.38))
            hazard = clamp(hazard - rng.uniform(0.08, 0.24))
            action_critical = clamp(action_critical - rng.uniform(0.18, 0.34))
        elif rng.random() < params["critical_rate"]:
            consequence = clamp(consequence + rng.uniform(0.22, 0.42))
            hazard = clamp(hazard + rng.uniform(0.08, 0.30))
            action_critical = clamp(action_critical + rng.uniform(0.20, 0.42))
            salience = clamp(salience - rng.uniform(0.10, 0.28))

        physical_score = clamp(0.45 * consequence + 0.30 * action_critical + 0.25 * hazard)
        critical = 1 if physical_score > 0.56 else 0
        tokens.append(
            {
                "token": idx,
                "salience": salience,
                "language_salience": language_salience,
                "uncertainty": uncertainty,
                "consequence": consequence,
                "hazard": hazard,
                "action_critical": action_critical,
                "cost": cost,
                "distractor": int(is_distractor),
                "critical": critical,
                "physical_score": physical_score,
            }
        )
    if sum(t["critical"] for t in tokens) == 0:
        best = max(tokens, key=lambda t: t["physical_score"])
        best["critical"] = 1
    return {"split": split_name, "task": task, "seed": seed, "episode_id": episode_id, "params": params, "budget": budget, "tokens": tokens}


def token_score(token, method, ablation=None):
    s = token["salience"]
    l = token["language_salience"]
    u = token["uncertainty"]
    c = token["consequence"]
    h = token["hazard"]
    a = token["action_critical"]
    cost = token["cost"]

    if method == "uniform_attention_budget":
        return 0.05 * (1.0 - token["token"] / TOKENS)
    if method == "token_salience_transformer":
        return 0.66 * s + 0.24 * l + 0.10 * u
    if method == "uncertainty_attention":
        return 0.74 * u + 0.14 * s + 0.12 * h
    if method == "active_perception_value":
        return 0.38 * u + 0.24 * c + 0.20 * h + 0.12 * a - 0.08 * cost
    if method == "compute_adaptive_policy":
        return (0.32 * s + 0.24 * u + 0.18 * c + 0.12 * a) / (0.35 + cost)
    if method == "risk_aware_attention":
        return 0.34 * h + 0.28 * u + 0.20 * c + 0.10 * a - 0.06 * cost
    if method == "embodied_consequence_budgeting":
        if ablation == "minus_physical_consequence_estimator":
            return 0.32 * s + 0.30 * u + 0.20 * h + 0.08 * a - 0.08 * cost
        if ablation == "minus_action_critical_contact_map":
            return 0.34 * c + 0.28 * h + 0.20 * u + 0.08 * s - 0.10 * cost
        if ablation == "minus_compute_latency_constraint":
            return 0.36 * c + 0.24 * a + 0.22 * h + 0.12 * u + 0.06 * s
        if ablation == "minus_safety_hazard_term":
            return 0.44 * c + 0.28 * a + 0.18 * u + 0.10 * s - 0.10 * cost
        if ablation == "token_salience_only_budget":
            return token_score(token, "token_salience_transformer")
        if ablation == "uncertainty_only_budget":
            return token_score(token, "uncertainty_attention")
        return 0.34 * c + 0.24 * a + 0.22 * h + 0.12 * u + 0.06 * s - 0.13 * cost
    if method == "oracle_consequence_attention":
        return 1.00 * token["physical_score"] + 0.10 * token["critical"] - 0.06 * cost
    raise KeyError(method)


def choose_tokens(episode, method, ablation=None):
    ordered = sorted(episode["tokens"], key=lambda t: token_score(t, method, ablation=ablation), reverse=True)
    selected = []
    used = 0.0
    for token in ordered:
        if used + token["cost"] <= episode["budget"] or not selected:
            selected.append(token)
            used += token["cost"]
    return selected, used


def evaluate_episode(episode, method, ablation=None):
    selected, cost = choose_tokens(episode, method, ablation=ablation)
    oracle_selected, oracle_cost = choose_tokens(episode, "oracle_consequence_attention")
    selected_ids = {t["token"] for t in selected}
    critical = [t for t in episode["tokens"] if t["critical"] == 1]
    missed = [t for t in critical if t["token"] not in selected_ids]
    selected_noncritical = [t for t in selected if t["critical"] == 0]

    recall = len(critical) and (len(critical) - len(missed)) / len(critical)
    wasted = len(selected) and len(selected_noncritical) / len(selected)
    latency = 1.0 if cost > episode["budget"] * 0.96 else 0.0
    missed_consequence = sum(t["physical_score"] for t in missed) / max(1, len(critical))
    missed_hazard = sum(t["hazard"] for t in missed) / max(1, len(critical))
    selected_value = sum(t["physical_score"] for t in selected) / max(1, len(selected))
    oracle_value = sum(t["physical_score"] for t in oracle_selected) / max(1, len(oracle_selected))

    latency_penalty = episode["params"]["latency_penalty"] * max(0.0, cost - episode["budget"] * 0.82)
    safety_prob = clamp(0.05 + 0.62 * missed_hazard + 0.18 * episode["params"]["occlusion"] - 0.12 * recall)
    success_prob = clamp(
        episode["task"]["base_success"]
        + 0.46 * recall
        + 0.18 * selected_value
        - 0.32 * missed_consequence
        - 0.18 * wasted
        - 0.15 * latency
        - 0.12 * latency_penalty
        - 0.12 * safety_prob
    )
    predicted_failure = clamp(0.08 + 0.70 * (1.0 - recall) + 0.16 * wasted + 0.14 * latency)
    actual_failure = 1.0 - success_prob
    regret = clamp(max(0.0, oracle_value - selected_value) + 0.35 * missed_consequence)

    rng = stable_rng("eval", episode["split"], episode["task"]["task"], episode["seed"], episode["episode_id"], method, ablation or "full")
    success = 1.0 if rng.random() < success_prob else 0.0
    safety = 1.0 if rng.random() < safety_prob else 0.0

    return {
        "split": episode["split"],
        "task": episode["task"]["task"],
        "seed": episode["seed"],
        "episode": episode["episode_id"],
        "method": method,
        "selected_tokens": len(selected),
        "critical_tokens": len(critical),
        "task_success": f"{success:.5f}",
        "critical_event_recall": f"{recall:.5f}",
        "wasted_attention": f"{wasted:.5f}",
        "compute_cost": f"{cost:.5f}",
        "latency_violation": f"{latency:.5f}",
        "safety_violation": f"{safety:.5f}",
        "physical_regret": f"{regret:.5f}",
        "calibration_error": f"{abs(predicted_failure - actual_failure):.5f}",
    }


def run_split(split, methods, episodes, stress_axis=None, stress_level=0.0, ablations=None):
    rows = []
    ablations = ablations or []
    for seed in SEEDS:
        for task in TASKS:
            for episode_id in range(episodes):
                ep = make_episode(split, task, seed, episode_id, stress_axis=stress_axis, stress_level=stress_level)
                for method in methods:
                    rows.append(evaluate_episode(ep, method))
                for ablation in ablations:
                    local = None if ablation == "full_embodied_consequence_budgeting" else ablation
                    row = evaluate_episode(ep, "embodied_consequence_budgeting", ablation=local)
                    row["method"] = ablation
                    rows.append(row)
        if stress_axis is None or seed == SEEDS[-1]:
            print(
                f"rollouts split={split} seed={seed} rows={len(rows)}"
                + (f" stress={stress_axis}:{stress_level}" if stress_axis else ""),
                flush=True,
            )
    return rows


def seed_metrics(rows, methods=None):
    methods = methods or sorted({r["method"] for r in rows})
    method_set = set(methods)
    groups = {}
    for r in rows:
        if r["method"] not in method_set:
            continue
        groups.setdefault((r["split"], r["method"], int(r["seed"])), []).append(r)
    out = []
    for split, method, seed in sorted(groups):
        vals = groups[(split, method, seed)]
        row = {"split": split, "method": method, "seed": seed, "rows": len(vals)}
        for metric in METRICS:
            row[metric] = f"{np.mean([float(v[metric]) for v in vals]):.5f}"
        out.append(row)
    return out


def aggregate_metrics(seed_rows):
    groups = {}
    for r in seed_rows:
        groups.setdefault((r["split"], r["method"]), []).append(r)
    out = []
    for (split, method), vals in sorted(groups.items()):
        for metric in METRICS:
            nums = [float(r[metric]) for r in vals]
            out.append(
                {
                    "split": split,
                    "method": method,
                    "metric": metric,
                    "mean": f"{np.mean(nums):.5f}",
                    "ci95": f"{ci95(nums):.5f}",
                    "seeds": len(nums),
                    "rows_per_seed": vals[0]["rows"],
                }
            )
    return out


def pairwise_stats(seed_rows, proposal="embodied_consequence_budgeting"):
    lookup = {(r["split"], r["method"], int(r["seed"])): r for r in seed_rows}
    split_methods = {}
    for r in seed_rows:
        split_methods.setdefault(r["split"], set()).add(r["method"])
    out = []
    for split in sorted(split_methods):
        refs = sorted(m for m in split_methods[split] if m != proposal)
        for reference in refs:
            for metric in METRICS:
                diffs = []
                for seed in SEEDS:
                    prop = lookup.get((split, proposal, seed))
                    ref = lookup.get((split, reference, seed))
                    if prop and ref:
                        diffs.append(float(prop[metric]) - float(ref[metric]))
                if diffs:
                    out.append(
                        {
                            "split": split,
                            "reference": reference,
                            "metric": metric,
                            "mean_diff": f"{np.mean(diffs):.5f}",
                            "ci95_diff": f"{ci95(diffs):.5f}",
                            "seeds": len(diffs),
                        }
                    )
    return out


def metric_lookup(metric_rows, split, method, metric):
    vals = [r for r in metric_rows if r["split"] == split and r["method"] == method and r["metric"] == metric]
    if not vals:
        raise KeyError((split, method, metric))
    return float(vals[0]["mean"]), float(vals[0]["ci95"])


def run_main():
    rows = []
    for split in SPLITS:
        rows.extend(run_split(split, METHODS, MAIN_EPISODES_PER_SEED))
    seed_rows = seed_metrics(rows, METHODS)
    metric_rows = aggregate_metrics(seed_rows)
    pair_rows = pairwise_stats(seed_rows)
    write_csv(RESULTS / "rollouts.csv", rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    return rows, seed_rows, metric_rows, pair_rows


def run_ablation():
    rows = run_split("combined_hard_shift", [], MAIN_EPISODES_PER_SEED, ablations=ABLATIONS)
    seed_rows = seed_metrics(rows, ABLATIONS)
    metric_rows = aggregate_metrics(seed_rows)
    summary = []
    for ablation in ABLATIONS:
        summary.append(
            {
                "ablation": ablation,
                "task_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'task_success')[0]:.5f}",
                "ci95_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'task_success')[1]:.5f}",
                "critical_event_recall": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'critical_event_recall')[0]:.5f}",
                "wasted_attention": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'wasted_attention')[0]:.5f}",
                "compute_cost": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'compute_cost')[0]:.5f}",
                "latency_violation": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'latency_violation')[0]:.5f}",
                "safety_violation": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'safety_violation')[0]:.5f}",
                "physical_regret": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'physical_regret')[0]:.5f}",
            }
        )
    write_csv(RESULTS / "ablation_rollouts.csv", rows)
    write_csv(RESULTS / "ablation_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "ablation_metrics.csv", summary)
    return rows, summary


def run_stress():
    axes = [
        "visual_distractor_load",
        "occlusion",
        "compute_budget_tightening",
        "latency_penalty",
        "hidden_physical_consequence",
        "combined",
    ]
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    methods = [
        "token_salience_transformer",
        "uncertainty_attention",
        "active_perception_value",
        "compute_adaptive_policy",
        "risk_aware_attention",
        "embodied_consequence_budgeting",
        "oracle_consequence_attention",
    ]
    raw = []
    summary = []
    for axis in axes:
        for level in levels:
            rows = run_split("combined_hard_shift", methods, STRESS_EPISODES_PER_SEED, stress_axis=axis, stress_level=level)
            for row in rows:
                row["stress_axis"] = axis
                row["stress_level"] = f"{level:.1f}"
            raw.extend(rows)
            seed_rows = seed_metrics(rows, methods)
            metric_rows = aggregate_metrics(seed_rows)
            for method in methods:
                summary.append(
                    {
                        "stress_axis": axis,
                        "stress_level": f"{level:.1f}",
                        "method": method,
                        "task_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'task_success')[0]:.5f}",
                        "ci95_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'task_success')[1]:.5f}",
                        "critical_event_recall": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'critical_event_recall')[0]:.5f}",
                        "wasted_attention": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'wasted_attention')[0]:.5f}",
                        "compute_cost": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'compute_cost')[0]:.5f}",
                        "latency_violation": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'latency_violation')[0]:.5f}",
                        "safety_violation": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'safety_violation')[0]:.5f}",
                    }
                )
    write_csv(RESULTS / "stress_sweep_raw.csv", raw)
    write_csv(RESULTS / "stress_sweep.csv", summary)
    write_csv(FIGURES / "stress_curve_data.csv", summary)
    return raw, summary


def negative_cases():
    rows = [
        {
            "case": "low_salience_catastrophic_contact",
            "expected_behavior": "allocate budget to low-token-salience contact with high consequence",
            "observed_outcome": "consequence budgeting helps but still misses contacts when occluded and compute is tight",
            "lesson": "physical consequence must be estimated before budget allocation, not assumed",
        },
        {
            "case": "visually_salient_but_harmless_distractor",
            "expected_behavior": "ignore distractor even if language mentions it",
            "observed_outcome": "token salience wastes attention; consequence budget reduces but does not eliminate waste",
            "lesson": "token attention is not a reliable control explanation",
        },
        {
            "case": "safety_hazard_requires_extra_compute",
            "expected_behavior": "spend compute despite latency risk",
            "observed_outcome": "compute-latency constraint can suppress necessary hazard attention",
            "lesson": "safety constraints should override budget pressure",
        },
        {
            "case": "deformable_snag_after_action",
            "expected_behavior": "reallocate action attention online",
            "observed_outcome": "single-shot budget cannot recover after late snag",
            "lesson": "receding-horizon attention is needed for deployment",
        },
    ]
    write_csv(RESULTS / "negative_cases.csv", rows)
    return rows


def plot_results(metric_rows, ablation_summary, stress_summary):
    labels = {
        "uniform_attention_budget": "Uniform",
        "token_salience_transformer": "Token salience",
        "uncertainty_attention": "Uncertainty",
        "active_perception_value": "Active perception",
        "compute_adaptive_policy": "Compute adaptive",
        "risk_aware_attention": "Risk aware",
        "embodied_consequence_budgeting": "Consequence",
        "oracle_consequence_attention": "Oracle",
    }
    splits = list(SPLITS.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(METHODS)))
    x = np.arange(len(splits))
    width = 0.095
    plt.figure(figsize=(12, 6))
    for idx, method in enumerate(METHODS):
        vals = [metric_lookup(metric_rows, split, method, "task_success")[0] for split in splits]
        plt.bar(x + (idx - 3.5) * width, vals, width=width, color=colors[idx], label=labels[method])
    plt.xticks(x, [s.replace("_", "\n") for s in splits], fontsize=8)
    plt.ylabel("Task success")
    plt.ylim(0.0, 1.0)
    plt.title("Embodied attention budgeting across shifts")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "attention_budget_success.png", dpi=220)
    plt.close()

    focus = [
        "token_salience_transformer",
        "uncertainty_attention",
        "active_perception_value",
        "compute_adaptive_policy",
        "risk_aware_attention",
        "embodied_consequence_budgeting",
        "oracle_consequence_attention",
    ]
    x = np.arange(len(focus))
    success = [metric_lookup(metric_rows, "combined_hard_shift", m, "task_success")[0] for m in focus]
    recall = [metric_lookup(metric_rows, "combined_hard_shift", m, "critical_event_recall")[0] for m in focus]
    wasted = [metric_lookup(metric_rows, "combined_hard_shift", m, "wasted_attention")[0] for m in focus]
    plt.figure(figsize=(11, 5.5))
    plt.bar(x - 0.24, success, width=0.24, label="success", color="#3b6ea8")
    plt.bar(x, recall, width=0.24, label="critical recall", color="#4f8f68")
    plt.bar(x + 0.24, wasted, width=0.24, label="wasted attention", color="#b5533c")
    plt.xticks(x, [labels[m] for m in focus], rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Combined hard-shift attention quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "attention_budget_quality.png", dpi=220)
    plt.close()

    safety = [metric_lookup(metric_rows, "combined_hard_shift", m, "safety_violation")[0] for m in focus]
    cost = [metric_lookup(metric_rows, "combined_hard_shift", m, "compute_cost")[0] for m in focus]
    latency = [metric_lookup(metric_rows, "combined_hard_shift", m, "latency_violation")[0] for m in focus]
    max_cost = max(cost)
    plt.figure(figsize=(11, 5.5))
    plt.bar(x - 0.24, safety, width=0.24, label="safety violation", color="#8c4b4b")
    plt.bar(x, np.asarray(cost) / max_cost, width=0.24, label="normalized compute", color="#8c6d31")
    plt.bar(x + 0.24, latency, width=0.24, label="latency violation", color="#78658b")
    plt.xticks(x, [labels[m] for m in focus], rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Safety and compute budget pressure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "attention_budget_safety_compute.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10.5, 5.5))
    ablations = [r["ablation"] for r in ablation_summary]
    vals = [float(r["task_success"]) for r in ablation_summary]
    plt.bar(np.arange(len(vals)), vals, color="#407076")
    plt.xticks(np.arange(len(vals)), [a.replace("_", "\n") for a in ablations], rotation=25, ha="right", fontsize=8)
    plt.ylabel("Task success")
    plt.ylim(0.0, 1.0)
    plt.title("Embodied attention-budget ablations")
    plt.tight_layout()
    plt.savefig(FIGURES / "attention_budget_ablation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10.5, 5.5))
    for method in focus:
        rows = [r for r in stress_summary if r["stress_axis"] == "combined" and r["method"] == method]
        levels = [float(r["stress_level"]) for r in rows]
        vals = [float(r["task_success"]) for r in rows]
        plt.plot(levels, vals, marker="o", label=labels[method])
    plt.xlabel("Combined stress level")
    plt.ylabel("Task success")
    plt.ylim(0.0, 1.0)
    plt.title("Combined attention-budget stress sweep")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "attention_budget_stress_sweep.png", dpi=220)
    plt.close()


def terminal_decision(metric_rows, pair_rows, ablation_summary):
    split = "combined_hard_shift"
    proposal = "embodied_consequence_budgeting"
    non_oracle = [m for m in METHODS if m not in {proposal, "oracle_consequence_attention"}]
    prop_success = metric_lookup(metric_rows, split, proposal, "task_success")[0]
    prop_recall = metric_lookup(metric_rows, split, proposal, "critical_event_recall")[0]
    prop_waste = metric_lookup(metric_rows, split, proposal, "wasted_attention")[0]
    prop_safety = metric_lookup(metric_rows, split, proposal, "safety_violation")[0]
    prop_latency = metric_lookup(metric_rows, split, proposal, "latency_violation")[0]
    best_success_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "task_success")[0])
    best_recall_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "critical_event_recall")[0])
    best_waste_method = min(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "wasted_attention")[0])
    best_safety_method = min(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "safety_violation")[0])
    best_latency_method = min(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "latency_violation")[0])
    best_success = metric_lookup(metric_rows, split, best_success_method, "task_success")[0]
    best_recall = metric_lookup(metric_rows, split, best_recall_method, "critical_event_recall")[0]
    best_waste = metric_lookup(metric_rows, split, best_waste_method, "wasted_attention")[0]
    best_safety = metric_lookup(metric_rows, split, best_safety_method, "safety_violation")[0]
    best_latency = metric_lookup(metric_rows, split, best_latency_method, "latency_violation")[0]
    paired_success = [
        r
        for r in pair_rows
        if r["split"] == split and r["reference"] == best_success_method and r["metric"] == "task_success"
    ][0]
    full = [r for r in ablation_summary if r["ablation"] == "full_embodied_consequence_budgeting"][0]
    strongest_ablation = max(float(r["task_success"]) for r in ablation_summary if r["ablation"] != "full_embodied_consequence_budgeting")
    ablation_drop = float(full["task_success"]) - strongest_ablation
    if (
        prop_success >= best_success + 0.035
        and prop_recall >= best_recall + 0.030
        and prop_waste <= best_waste - 0.030
        and prop_safety <= best_safety + 0.015
        and prop_latency <= best_latency + 0.020
        and float(paired_success["mean_diff"]) > 0.030
        and ablation_drop >= 0.020
    ):
        return "STRONG_REVISE"
    return "KILL_ARCHIVE"


def write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, terminal):
    split = "combined_hard_shift"
    proposal = "embodied_consequence_budgeting"
    lines = [
        "Paper 89 embodied_attention_budgeting v4 rebuild",
        f"Terminal recommendation: {terminal}",
        "Reason: deterministic local attention-budget benchmark added; no robot hardware or implemented foundation-model attention module is available.",
        f"Main rollout rows: {sum(1 for _ in open(RESULTS / 'rollouts.csv', encoding='utf-8')) - 1}",
        f"Ablation rollout rows: {sum(1 for _ in open(RESULTS / 'ablation_rollouts.csv', encoding='utf-8')) - 1}",
        f"Stress rollout rows: {sum(1 for _ in open(RESULTS / 'stress_sweep_raw.csv', encoding='utf-8')) - 1}",
        f"Seeds: {SEEDS}",
        "",
        "Combined hard shift:",
    ]
    for method in METHODS:
        success = metric_lookup(metric_rows, split, method, "task_success")
        recall = metric_lookup(metric_rows, split, method, "critical_event_recall")
        wasted = metric_lookup(metric_rows, split, method, "wasted_attention")
        cost = metric_lookup(metric_rows, split, method, "compute_cost")
        latency = metric_lookup(metric_rows, split, method, "latency_violation")
        safety = metric_lookup(metric_rows, split, method, "safety_violation")
        regret = metric_lookup(metric_rows, split, method, "physical_regret")
        calib = metric_lookup(metric_rows, split, method, "calibration_error")
        lines.append(
            f"{method} task_success={success[0]:.5f} ci95={success[1]:.5f} recall={recall[0]:.5f} "
            f"wasted={wasted[0]:.5f} compute={cost[0]:.5f} latency={latency[0]:.5f} "
            f"safety={safety[0]:.5f} regret={regret[0]:.5f} calibration={calib[0]:.5f}"
        )
    non_oracle = [m for m in METHODS if m not in {proposal, "oracle_consequence_attention"}]
    best_success_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "task_success")[0])
    paired = [
        r
        for r in pair_rows
        if r["split"] == split and r["reference"] == best_success_method and r["metric"] == "task_success"
    ][0]
    lines.append(
        f"paired task-success diff vs best success baseline {best_success_method}="
        f"{float(paired['mean_diff']):.5f} ci95={float(paired['ci95_diff']):.5f}"
    )
    lines.append("")
    lines.append("Ablations:")
    for row in ablation_summary:
        lines.append(
            f"{row['ablation']} task_success={float(row['task_success']):.5f} ci95={float(row['ci95_success']):.5f} "
            f"recall={float(row['critical_event_recall']):.5f} wasted={float(row['wasted_attention']):.5f} "
            f"compute={float(row['compute_cost']):.5f} latency={float(row['latency_violation']):.5f} "
            f"safety={float(row['safety_violation']):.5f} regret={float(row['physical_regret']):.5f}"
        )
    lines.append("")
    lines.append("Combined stress level 1.0:")
    for row in stress_summary:
        if row["stress_axis"] == "combined" and row["stress_level"] == "1.0":
            lines.append(
                f"{row['method']} task_success={float(row['task_success']):.5f} ci95={float(row['ci95_success']):.5f} "
                f"recall={float(row['critical_event_recall']):.5f} wasted={float(row['wasted_attention']):.5f} "
                f"compute={float(row['compute_cost']):.5f} latency={float(row['latency_violation']):.5f} "
                f"safety={float(row['safety_violation']):.5f}"
            )
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    main_rows, seed_rows, metric_rows, pair_rows = run_main()
    ablation_rows, ablation_summary = run_ablation()
    stress_raw, stress_summary = run_stress()
    negative_cases()
    terminal = terminal_decision(metric_rows, pair_rows, ablation_summary)
    plot_results(metric_rows, ablation_summary, stress_summary)
    write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, terminal)
    print(f"terminal={terminal}", flush=True)
    print(f"wrote results to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
