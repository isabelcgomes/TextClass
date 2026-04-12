"""
evaluate.py — Batch evaluation of the classifier against the labeled dataset.

Usage:
    python scripts/evaluate.py \
        --dataset data/classification_dataset.csv \
        --output  reports/evaluation_report.json \
        --api-url http://localhost:8000
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_text(text: str, api_url: str) -> dict:
    """Call the /classify endpoint and return the full response dict."""
    response = requests.post(
        f"{api_url}/classify",
        json={"text": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def run_batch(df: pd.DataFrame, api_url: str) -> pd.DataFrame:
    """Run the classifier on every row. Returns df with new columns."""
    results = []
    total = len(df)

    for i, row in df.iterrows():
        print(f"  [{i+1}/{total}] id={row['id']} ...", end=" ", flush=True)
        try:
            resp = classify_text(row["text"], api_url)
            results.append({
                "id": row["id"],
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": resp["predicted_label"],
                "is_valid_label": resp["is_valid_label"],
                "latency_ms": resp["latency_ms"],
                "error": None,
            })
            match = "✓" if resp["predicted_label"] == row["label"] else "✗"
            print(f'{match}  pred="{resp["predicted_label"]}"')
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            results.append({
                "id": row["id"],
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": None,
                "is_valid_label": False,
                "latency_ms": None,
                "error": str(exc),
            })

        # Polite rate-limiting: avoid hammering the API
        time.sleep(0.3)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results_df: pd.DataFrame) -> dict:
    valid = results_df[results_df["error"].isna() & results_df["predicted_label"].notna()]
    errors = results_df[results_df["error"].notna()]

    y_true = valid["true_label"].tolist()
    y_pred = valid["predicted_label"].tolist()
    labels = sorted(set(y_true))

    # --- Overall ---
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # --- Per-class ---
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_dict = {
        "labels": labels,
        "matrix": cm.tolist(),
    }

    # --- Confusion pairs (top misclassifications) ---
    confusion_pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i][j] > 0:
                confusion_pairs.append({
                    "true": true_label,
                    "predicted": pred_label,
                    "count": int(cm[i][j]),
                })
    confusion_pairs.sort(key=lambda x: -x["count"])

    # --- Invalid label rate ---
    invalid_label_count = int((~results_df["is_valid_label"]).sum())

    # --- Latency ---
    lat = valid["latency_ms"].dropna()
    latency_stats = {
        "mean_ms": round(lat.mean(), 1),
        "p50_ms": round(lat.quantile(0.50), 1),
        "p90_ms": round(lat.quantile(0.90), 1),
        "p99_ms": round(lat.quantile(0.99), 1),
        "max_ms": round(lat.max(), 1),
    }

    return {
        "summary": {
            "total_samples": len(results_df),
            "evaluated": len(valid),
            "errors": len(errors),
            "invalid_label_predictions": invalid_label_count,
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        },
        "per_class": report_dict,
        "confusion_matrix": cm_dict,
        "top_confusions": confusion_pairs[:10],
        "latency": latency_stats,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def print_report(metrics: dict) -> None:
    s = metrics["summary"]
    print("\n" + "=" * 65)
    print("  EVALUATION REPORT")
    print("=" * 65)
    print(f"  Samples evaluated : {s['evaluated']} / {s['total_samples']}")
    print(f"  Errors            : {s['errors']}")
    print(f"  Invalid labels    : {s['invalid_label_predictions']}")
    print(f"  Accuracy          : {s['accuracy']:.2%}")
    print(f"  Macro F1          : {s['macro_f1']:.2%}")
    print(f"  Weighted F1       : {s['weighted_f1']:.2%}")

    print("\n  LATENCY")
    lat = metrics["latency"]
    print(f"  Mean: {lat['mean_ms']}ms  |  P50: {lat['p50_ms']}ms  |  "
          f"P90: {lat['p90_ms']}ms  |  P99: {lat['p99_ms']}ms")

    print("\n  PER-CLASS F1")
    pc = metrics["per_class"]
    for label in metrics["confusion_matrix"]["labels"]:
        row = pc.get(label, {})
        f1 = row.get("f1-score", 0)
        support = row.get("support", 0)
        bar = "█" * int(f1 * 20)
        print(f"  {f1:.2f} {bar:<20}  [{support:2d}]  {label}")

    if metrics["top_confusions"]:
        print("\n  TOP MISCLASSIFICATIONS")
        for pair in metrics["top_confusions"][:5]:
            print(f"  [{pair['count']}x]  \"{pair['true']}\"")
            print(f"        → \"{pair['predicted']}\"")

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the classifier in batch.")
    parser.add_argument("--dataset", default="data/classification_dataset.csv")
    parser.add_argument("--output", default="reports/evaluation_report.json")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    print(f"\nLoading dataset from: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"Loaded {len(df)} rows.\n")

    print("Running classifier...")
    results_df = run_batch(df, args.api_url)

    metrics = compute_metrics(results_df)
    print_report(metrics)

    # Save structured report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "api_url": args.api_url,
        "dataset": args.dataset,
        "metrics": metrics,
        "predictions": results_df.to_dict(orient="records"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
