#!/usr/bin/env python3
"""
Re-evaluate CV results using only test molecules whose max Tanimoto similarity
to the training set is below a threshold.

This filters out "easy" molecules that are structurally similar to training data,
giving a more rigorous assessment of model generalization. Since predictions
already exist as CSV files, this is pure pandas/numpy post-processing.

Usage:
    python evals/library_cv/scripts/evaluate_similarity_filtered.py \
        --cv-results evals/library_cv/results/transformer/transformer_wdr91_... \
        --similarity-data evals/analysis/results/fold_similarity_.../similarity_data.csv \
        --threshold 0.4

    # For baseline models:
    python evals/library_cv/scripts/evaluate_similarity_filtered.py \
        --cv-results evals/library_cv/results/baseline/baselines_WDR91_... \
        --similarity-data evals/analysis/results/fold_similarity_.../similarity_data.csv \
        --threshold 0.4 \
        --model-type xgb
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path so we can import cv_utils
PROJECT_ROOT = Path(__file__).resolve().parents[3]

from delbert.data.cv_utils import aggregate_cv_metrics, calculate_ranking_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-evaluate CV results with similarity-based filtering"
    )
    parser.add_argument(
        "--cv-results",
        type=Path,
        required=True,
        help="Path to CV results directory (e.g. evals/library_cv/results/transformer/...)",
    )
    parser.add_argument(
        "--similarity-data",
        type=Path,
        required=True,
        help="Path to similarity_data.csv from compute_fold_similarity.py",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Max similarity threshold. Keep only molecules with max_similarity < threshold.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Baseline model type (e.g. 'xgb', 'lgbm'). If set, reads baseline format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory. Default: evals/library_cv/results/similarity_filtered/...",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="100,500",
        help="Comma-separated K values for precision@K/enrichment@K (default: 100,500)",
    )
    return parser.parse_args()


def discover_folds(cv_results_dir: Path, model_type: str | None) -> list[int]:
    """Discover which fold indices exist in the results directory."""
    if model_type:
        # Baseline: {model_type}/fold_N/predictions.csv
        model_dir = cv_results_dir / model_type
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Baseline model directory not found: {model_dir}"
            )
        fold_dirs = sorted(model_dir.glob("fold_*"))
    else:
        # Transformer: fold_N/test_predictions.csv
        fold_dirs = sorted(cv_results_dir.glob("fold_*"))

    fold_indices = []
    for d in fold_dirs:
        if d.is_dir() and d.name.startswith("fold_"):
            try:
                fold_idx = int(d.name.split("_")[1])
                fold_indices.append(fold_idx)
            except (ValueError, IndexError):
                continue

    if not fold_indices:
        raise FileNotFoundError(
            f"No fold directories found in {cv_results_dir}"
            + (f"/{model_type}" if model_type else "")
        )

    return sorted(fold_indices)


def load_fold_predictions(
    cv_results_dir: Path, fold_idx: int, model_type: str | None
) -> pd.DataFrame:
    """Load predictions for a single fold.

    Returns DataFrame with columns: compound_id, label, prob
    """
    if model_type:
        pred_path = cv_results_dir / model_type / f"fold_{fold_idx}" / "predictions.csv"
    else:
        pred_path = cv_results_dir / f"fold_{fold_idx}" / "test_predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    df = pd.read_csv(pred_path)

    # Normalize column names
    if model_type:
        # Baseline: compound_id, label, prediction
        required = {"compound_id", "label", "prediction"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {pred_path}: {missing}. Found: {list(df.columns)}"
            )
        df = df.rename(columns={"prediction": "prob"})
    else:
        # Transformer: compound_id, label, prob, fold
        required = {"compound_id", "label", "prob"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {pred_path}: {missing}. Found: {list(df.columns)}"
            )

    # Coerce compound_id to string for safe joining
    df["compound_id"] = df["compound_id"].astype(str)

    return df[["compound_id", "label", "prob"]]


def load_original_fold_results(cv_results_dir: Path, model_type: str | None) -> pd.DataFrame | None:
    """Load the original fold_results.csv for comparison."""
    if model_type:
        path = cv_results_dir / f"{model_type}_fold_results.csv"
    else:
        path = cv_results_dir / "fold_results.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


def main():
    args = parse_args()

    cv_results_dir = args.cv_results.resolve()
    similarity_path = args.similarity_data.resolve()
    threshold = args.threshold
    model_type = args.model_type
    ks = [int(k.strip()) for k in args.ks.split(",")]

    # Validate inputs
    if not cv_results_dir.is_dir():
        raise FileNotFoundError(f"CV results directory not found: {cv_results_dir}")
    if not similarity_path.is_file():
        raise FileNotFoundError(f"Similarity data not found: {similarity_path}")
    if threshold <= 0 or threshold > 1.0:
        raise ValueError(f"Threshold must be in (0, 1.0], got {threshold}")

    # Load similarity data
    print(f"Loading similarity data from: {similarity_path}")
    sim_df = pd.read_csv(similarity_path)
    required_sim_cols = {"fold", "compound_id", "max_similarity"}
    missing_sim = required_sim_cols - set(sim_df.columns)
    if missing_sim:
        raise ValueError(
            f"Missing columns in similarity data: {missing_sim}. Found: {list(sim_df.columns)}"
        )
    sim_df["compound_id"] = sim_df["compound_id"].astype(str)
    sim_df["fold"] = sim_df["fold"].astype(int)
    # Deduplicate: some similarity data has duplicate compound_ids per fold
    n_before = len(sim_df)
    sim_df = sim_df.drop_duplicates(subset=["fold", "compound_id"], keep="first")
    n_dupes = n_before - len(sim_df)
    if n_dupes > 0:
        print(f"  Removed {n_dupes} duplicate (fold, compound_id) rows from similarity data")
    print(f"  {len(sim_df)} rows, {sim_df['fold'].nunique()} folds")

    # Discover folds
    fold_indices = discover_folds(cv_results_dir, model_type)
    n_folds = len(fold_indices)
    print(f"\nFound {n_folds} folds in: {cv_results_dir}")
    if model_type:
        print(f"  Model type: {model_type}")
    print(f"  Similarity threshold: {threshold}")
    print(f"  K values: {ks}")

    # Set up output directory
    source_name = cv_results_dir.name
    if model_type:
        source_name = f"{source_name}_{model_type}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    thr_str = f"{threshold:.2f}".rstrip("0").rstrip(".")

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (
            PROJECT_ROOT
            / "evals"
            / "library_cv"
            / "results"
            / "similarity_filtered"
            / f"{source_name}_thr{thr_str}_{timestamp}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Process each fold
    fold_results = []
    fold_stats = []
    skipped_folds = []

    for fold_idx in fold_indices:
        print(f"\n--- Fold {fold_idx} ---")

        # Load predictions
        pred_df = load_fold_predictions(cv_results_dir, fold_idx, model_type)
        n_total = len(pred_df)

        # Get similarity data for this fold
        sim_fold = sim_df[sim_df["fold"] == fold_idx].copy()
        if len(sim_fold) == 0:
            raise ValueError(
                f"No similarity data found for fold {fold_idx}. "
                f"Available folds in similarity data: {sorted(sim_df['fold'].unique())}"
            )

        # Validate: all prediction compound_ids must exist in similarity data
        pred_ids = set(pred_df["compound_id"])
        sim_ids = set(sim_fold["compound_id"])
        missing_ids = pred_ids - sim_ids
        if missing_ids:
            n_missing = len(missing_ids)
            example_ids = list(missing_ids)[:5]
            raise ValueError(
                f"Fold {fold_idx}: {n_missing} compound_ids in predictions not found "
                f"in similarity data. Examples: {example_ids}. "
                f"Predictions have {len(pred_ids)} compounds, "
                f"similarity data has {len(sim_ids)} compounds for this fold."
            )

        # Join predictions with similarity
        merged = pred_df.merge(
            sim_fold[["compound_id", "max_similarity"]],
            on="compound_id",
            how="left",
        )

        # Filter: keep only molecules below threshold
        filtered = merged[merged["max_similarity"] < threshold].copy()
        n_filtered = len(filtered)
        n_removed = n_total - n_filtered
        n_positives = int(filtered["label"].sum())

        print(f"  Total: {n_total}, Kept: {n_filtered} ({n_filtered/n_total*100:.1f}%), Removed: {n_removed}")
        print(f"  Positives in filtered set: {n_positives}")

        stat = {
            "fold": fold_idx,
            "n_total": n_total,
            "n_filtered": n_filtered,
            "n_removed": n_removed,
            "pct_kept": round(n_filtered / n_total * 100, 2) if n_total > 0 else 0,
            "n_positives": n_positives,
            "n_negatives": n_filtered - n_positives,
        }
        fold_stats.append(stat)

        # Handle empty fold
        if n_filtered == 0:
            print(f"  WARNING: All molecules filtered out! Skipping this fold.")
            skipped_folds.append(fold_idx)
            continue

        if n_positives == 0:
            print(f"  WARNING: No positives remaining after filtering! Skipping this fold.")
            skipped_folds.append(fold_idx)
            continue

        # Check for adaptive K
        for k in ks:
            if n_filtered < k:
                print(f"  Note: Only {n_filtered} molecules (< K={k}), precision@{k}/enrichment@{k} will be skipped")

        # Compute metrics
        labels = filtered["label"].values.astype(int)
        probs = filtered["prob"].values.astype(float)
        metrics = calculate_ranking_metrics(labels, probs, ks=ks)

        print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"  PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}")
        if "bedroc" in metrics:
            print(f"  BEDROC: {metrics['bedroc']:.4f}")

        fold_results.append(metrics)

        # Save per-fold outputs
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save filtered predictions (same format as source)
        filtered[["compound_id", "label", "prob"]].to_csv(
            fold_dir / "test_predictions.csv", index=False
        )

        # Save metrics
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"AGGREGATION")
    print(f"{'='*60}")

    if skipped_folds:
        print(f"\nWARNING: {len(skipped_folds)} fold(s) skipped due to empty filtered set: {skipped_folds}")

    n_valid_folds = len(fold_results)
    if n_valid_folds == 0:
        raise RuntimeError(
            "All folds were empty after filtering. Try a higher threshold."
        )

    print(f"\nUsing {n_valid_folds}/{n_folds} folds for aggregation")

    aggregate = aggregate_cv_metrics(fold_results)

    # Save fold_results.csv (same format as library_cv)
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(output_dir / "fold_results.csv", index=False)

    # Save aggregate.yaml (same format as library_cv)
    aggregate_output = {
        "experiment": f"{source_name}_thr{thr_str}",
        "n_folds": n_valid_folds,
        "metrics": {},
    }
    for metric_name, stats in aggregate["metrics"].items():
        aggregate_output["metrics"][metric_name] = {
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "ci_95": [float(stats["ci_95_lower"]), float(stats["ci_95_upper"])],
        }

    with open(output_dir / "aggregate.yaml", "w") as f:
        yaml.dump(aggregate_output, f, default_flow_style=False)

    # Save config.yaml with metadata
    config = {
        "source_cv_results": str(cv_results_dir),
        "similarity_data": str(similarity_path),
        "threshold": threshold,
        "model_type": model_type,
        "ks": ks,
        "n_folds_total": n_folds,
        "n_folds_valid": n_valid_folds,
        "skipped_folds": skipped_folds,
        "fold_stats": fold_stats,
        "timestamp": timestamp,
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Print aggregate metrics
    print(f"\nFiltered metrics (threshold < {threshold}):")
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'95% CI':>25}")
    print("-" * 72)
    for metric_name in sorted(aggregate_output["metrics"]):
        m = aggregate_output["metrics"][metric_name]
        ci_str = f"[{m['ci_95'][0]:.4f}, {m['ci_95'][1]:.4f}]"
        print(f"  {metric_name:<23} {m['mean']:>10.4f} {m['std']:>10.4f} {ci_str:>25}")

    # Comparison with original results
    original_df = load_original_fold_results(cv_results_dir, model_type)
    if original_df is not None:
        print(f"\n{'='*60}")
        print(f"COMPARISON: Original vs Filtered (threshold < {threshold})")
        print(f"{'='*60}")

        # Compute original aggregate from fold_results
        original_records = original_df.to_dict("records")
        # Baseline fold_results has a 'fold' column, transformer doesn't
        for rec in original_records:
            rec.pop("fold", None)
            rec.pop("train_time", None)

        original_agg = aggregate_cv_metrics(original_records)

        # Compare metrics present in both
        compare_metrics = sorted(
            set(aggregate_output["metrics"].keys())
            & set(original_agg["metrics"].keys())
        )

        print(f"\n{'Metric':<25} {'Original':>10} {'Filtered':>10} {'Diff':>10}")
        print("-" * 57)
        for metric_name in compare_metrics:
            orig_mean = original_agg["metrics"][metric_name]["mean"]
            filt_mean = aggregate_output["metrics"][metric_name]["mean"]
            diff = filt_mean - orig_mean
            print(f"  {metric_name:<23} {orig_mean:>10.4f} {filt_mean:>10.4f} {diff:>+10.4f}")

        print(f"\n  Original folds: {len(original_records)}, Filtered folds: {n_valid_folds}")

    # Summary stats
    total_kept = sum(s["n_filtered"] for s in fold_stats)
    total_all = sum(s["n_total"] for s in fold_stats)
    print(f"\n  Total molecules: {total_all} -> {total_kept} ({total_kept/total_all*100:.1f}% kept)")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
