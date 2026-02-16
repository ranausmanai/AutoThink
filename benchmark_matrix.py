"""Reproducible multi-seed, multi-budget benchmark for AutoThink vs FLAML vs AutoGluon.

Outputs:
- benchmarks/benchmark_raw.csv
- benchmarks/benchmark_summary.csv
- benchmarks/benchmark_report.md
- benchmarks/pareto_by_budget.png

Usage:
    python benchmark_matrix.py
    python benchmark_matrix.py --budgets 10,30,60 --seeds 42,1337,2025
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split


TOOL_AUTOTHINK = "AutoThink V4"
TOOL_FLAML = "FLAML"
TOOL_AUTOGLUON = "AutoGluon"
TOOLS = [TOOL_AUTOTHINK, TOOL_FLAML, TOOL_AUTOGLUON]


@dataclass
class DatasetSpec:
    name: str
    target: str
    task: str  # "binary" | "regression"
    metric_name: str
    loader: Callable[[int], pd.DataFrame]


def load_heart_disease(seed: int) -> pd.DataFrame:
    df = pd.read_csv("playground-series-s6e2/train.csv")
    return df.sample(n=10000, random_state=seed).reset_index(drop=True)


def load_loan(seed: int) -> pd.DataFrame:
    df = pd.read_csv("playground-series-s5e11/train.csv")
    return df.sample(n=10000, random_state=seed).reset_index(drop=True)


def load_regression(_: int) -> pd.DataFrame:
    # Fixed synthetic dataset for reproducibility and speed.
    np.random.seed(12345)
    n = 5000
    df = pd.DataFrame(
        {
            "sqft": np.random.uniform(500, 5000, n),
            "bedrooms": np.random.choice([1, 2, 3, 4, 5], n),
            "age": np.random.uniform(0, 50, n),
            "location_score": np.random.uniform(1, 10, n),
            "garage": np.random.choice([0, 1], n),
            "neighborhood": np.random.choice(["A", "B", "C", "D"], n),
        }
    )
    df["price"] = (
        df["sqft"] * 150
        + df["bedrooms"] * 20000
        + df["location_score"] * 30000
        - df["age"] * 2000
        + df["garage"] * 15000
        + np.random.randn(n) * 30000
    )
    return df


DATASETS = [
    DatasetSpec(
        name="Heart Disease (binary, 10K)",
        target="Heart Disease",
        task="binary",
        metric_name="AUC",
        loader=load_heart_disease,
    ),
    DatasetSpec(
        name="Loan Repayment (binary, 10K)",
        target="loan_paid_back",
        task="binary",
        metric_name="AUC",
        loader=load_loan,
    ),
    DatasetSpec(
        name="House Price (regression, 5K)",
        target="price",
        task="regression",
        metric_name="RMSE",
        loader=load_regression,
    ),
]


def score_metric(y_true: pd.Series, y_pred: np.ndarray, task: str) -> float:
    if task == "binary":
        return float(roc_auc_score(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_autothink(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    target_col: str,
    time_budget: int,
) -> tuple[np.ndarray, float]:
    from autothink.core.autothink_v4 import AutoThinkV4

    train_df = x_train.copy()
    train_df[target_col] = y_train.values

    t0 = time.time()
    model = AutoThinkV4(time_budget=time_budget, verbose=False)
    model.fit(train_df, target_col)
    elapsed = time.time() - t0

    preds = model.predict(x_val)
    return np.asarray(preds), elapsed


def run_flaml(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    task: str,
    time_budget: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    from flaml import AutoML

    automl = AutoML()
    flaml_task = "classification" if task != "regression" else "regression"
    flaml_metric = "roc_auc" if task == "binary" else "rmse"

    t0 = time.time()
    automl.fit(
        x_train,
        y_train,
        task=flaml_task,
        metric=flaml_metric,
        time_budget=time_budget,
        verbose=0,
        seed=seed,
    )
    elapsed = time.time() - t0

    if task == "binary":
        preds = automl.predict_proba(x_val)[:, 1]
    else:
        preds = automl.predict(x_val)

    return np.asarray(preds), elapsed


def run_autogluon(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    target_col: str,
    task: str,
    time_budget: int,
) -> tuple[np.ndarray, float]:
    from autogluon.tabular import TabularPredictor

    train_df = x_train.copy()
    train_df[target_col] = y_train.values

    ag_task = "binary" if task == "binary" else "regression"
    ag_metric = "roc_auc" if task == "binary" else "root_mean_squared_error"

    t0 = time.time()
    predictor = TabularPredictor(
        label=target_col,
        problem_type=ag_task,
        eval_metric=ag_metric,
        verbosity=0,
    ).fit(
        train_df,
        time_limit=time_budget,
    )
    elapsed = time.time() - t0

    if task == "binary":
        proba = predictor.predict_proba(x_val)
        pos_col = proba.columns[-1]
        preds = proba[pos_col].values
    else:
        preds = predictor.predict(x_val).values

    return np.asarray(preds), elapsed


def is_better(task: str, s1: float, s2: float) -> bool:
    if task == "regression":
        return s1 < s2
    return s1 > s2


def ci95(series: pd.Series) -> float:
    vals = series.dropna().to_numpy(dtype=float)
    n = len(vals)
    if n <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / np.sqrt(n))


def safe_call(fn: Callable[[], tuple[np.ndarray, float]]) -> tuple[np.ndarray | None, float | None, str | None]:
    try:
        preds, elapsed = fn()
        return preds, elapsed, None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def build_report(raw: pd.DataFrame, summary: pd.DataFrame, out_path: Path, budgets: list[int], seeds: list[int]) -> None:
    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(f"Budgets: {budgets}")
    lines.append(f"Seeds: {seeds}")
    lines.append("")

    ok_runs = raw[raw["status"] == "ok"]
    failed = raw[raw["status"] != "ok"]

    lines.append("## Run Health")
    lines.append("")
    lines.append(f"- Total runs: {len(raw)}")
    lines.append(f"- Successful runs: {len(ok_runs)}")
    lines.append(f"- Failed runs: {len(failed)}")
    if not failed.empty:
        lines.append("- Failed combinations (dataset/tool/budget/seed):")
        for _, row in failed.iterrows():
            lines.append(
                f"  - {row['dataset']} | {row['tool']} | budget={row['budget_s']} | seed={row['seed']} | {row['error'][:160]}"
            )
    lines.append("")

    lines.append("## Summary (mean +- 95% CI)")
    lines.append("")
    for budget in budgets:
        lines.append(f"### Budget = {budget}s")
        lines.append("")
        for dataset in DATASETS:
            part = summary[(summary["budget_s"] == budget) & (summary["dataset"] == dataset.name)]
            if part.empty:
                continue
            lines.append(f"#### {dataset.name} ({dataset.metric_name})")
            lines.append("")
            lines.append("| Tool | Quality | Time |")
            lines.append("|------|---------|------|")
            for _, row in part.sort_values("tool").iterrows():
                if dataset.task == "regression":
                    qual = f"{row['metric_mean']:.2f} +- {row['metric_ci95']:.2f}"
                else:
                    qual = f"{row['metric_mean']:.5f} +- {row['metric_ci95']:.5f}"
                t = f"{row['time_mean_s']:.2f}s +- {row['time_ci95_s']:.2f}s"
                lines.append(f"| {row['tool']} | {qual} | {t} |")
            lines.append("")

    out_path.write_text("\n".join(lines))


def plot_pareto(summary: pd.DataFrame, out_path: Path, budgets: list[int]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.4 * len(budgets), 4.8), constrained_layout=True)

    if len(budgets) == 1:
        axes = [axes]

    palette = {
        TOOL_AUTOTHINK: "#0f766e",
        TOOL_FLAML: "#1d4ed8",
        TOOL_AUTOGLUON: "#b45309",
    }

    for idx, budget in enumerate(budgets):
        ax = axes[idx]
        part = summary[summary["budget_s"] == budget]

        for dataset in DATASETS:
            ds = part[part["dataset"] == dataset.name]
            if ds.empty:
                continue
            for _, row in ds.iterrows():
                marker = "o" if dataset.task == "binary" else "s"
                ax.errorbar(
                    row["time_mean_s"],
                    row["metric_mean"],
                    xerr=row["time_ci95_s"],
                    yerr=row["metric_ci95"],
                    fmt=marker,
                    color=palette.get(row["tool"], "#334155"),
                    alpha=0.9,
                    markersize=8,
                )
                short_ds = dataset.name.split(" (")[0]
                ax.annotate(
                    f"{row['tool'].replace('AutoThink V4', 'AutoThink')}\n{short_ds}",
                    (row["time_mean_s"], row["metric_mean"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                )

        ax.set_title(f"Budget {budget}s")
        ax.set_xlabel("Train time (s)")
        ax.set_ylabel("Metric (AUC up, RMSE down)")

    fig.suptitle("AutoML Trade-off: Time vs Quality (mean with 95% CI)", fontsize=13)
    fig.savefig(out_path, dpi=180)


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", default="10,30,60", help="Comma-separated time budgets in seconds")
    parser.add_argument("--seeds", default="42,1337,2025", help="Comma-separated seeds")
    parser.add_argument("--outdir", default="benchmarks", help="Output directory")
    args = parser.parse_args()

    budgets = parse_int_list(args.budgets)
    seeds = parse_int_list(args.seeds)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    total = len(DATASETS) * len(budgets) * len(seeds) * len(TOOLS)
    counter = 0
    print(f"Running {total} fits | datasets={len(DATASETS)} budgets={budgets} seeds={seeds}")

    for dataset in DATASETS:
        for budget in budgets:
            for seed in seeds:
                df = dataset.loader(seed)
                x = df.drop(columns=[dataset.target])
                y = df[dataset.target]

                stratify = y if dataset.task != "regression" else None
                x_train, x_val, y_train, y_val = train_test_split(
                    x,
                    y,
                    test_size=0.2,
                    random_state=seed,
                    stratify=stratify,
                )

                runners = [
                    (
                        TOOL_AUTOTHINK,
                        lambda: run_autothink(
                            x_train,
                            y_train,
                            x_val,
                            dataset.target,
                            budget,
                        ),
                    ),
                    (
                        TOOL_FLAML,
                        lambda: run_flaml(
                            x_train,
                            y_train,
                            x_val,
                            dataset.task,
                            budget,
                            seed,
                        ),
                    ),
                    (
                        TOOL_AUTOGLUON,
                        lambda: run_autogluon(
                            x_train,
                            y_train,
                            x_val,
                            dataset.target,
                            dataset.task,
                            budget,
                        ),
                    ),
                ]

                for tool_name, fn in runners:
                    counter += 1
                    print(
                        f"[{counter:03d}/{total}] {dataset.name} | budget={budget}s | seed={seed} | {tool_name}",
                        flush=True,
                    )
                    preds, elapsed, err = safe_call(fn)
                    if preds is None:
                        rows.append(
                            {
                                "dataset": dataset.name,
                                "task": dataset.task,
                                "metric": dataset.metric_name,
                                "budget_s": budget,
                                "seed": seed,
                                "tool": tool_name,
                                "status": "failed",
                                "score": np.nan,
                                "time_s": np.nan,
                                "error": err,
                            }
                        )
                        continue

                    s = score_metric(y_val, preds, dataset.task)
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "task": dataset.task,
                            "metric": dataset.metric_name,
                            "budget_s": budget,
                            "seed": seed,
                            "tool": tool_name,
                            "status": "ok",
                            "score": s,
                            "time_s": elapsed,
                            "error": "",
                        }
                    )

    raw = pd.DataFrame(rows)
    raw_path = outdir / "benchmark_raw.csv"
    raw.to_csv(raw_path, index=False)

    ok = raw[raw["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No successful benchmark runs; cannot produce summary.")

    grouped = (
        ok.groupby(["dataset", "task", "metric", "budget_s", "tool"], as_index=False)
        .agg(
            metric_mean=("score", "mean"),
            metric_std=("score", "std"),
            time_mean_s=("time_s", "mean"),
            time_std_s=("time_s", "std"),
            n=("score", "count"),
        )
    )
    grouped["metric_ci95"] = (
        ok.groupby(["dataset", "task", "metric", "budget_s", "tool"])["score"]
        .apply(ci95)
        .values
    )
    grouped["time_ci95_s"] = (
        ok.groupby(["dataset", "task", "metric", "budget_s", "tool"])["time_s"]
        .apply(ci95)
        .values
    )

    summary_path = outdir / "benchmark_summary.csv"
    grouped.to_csv(summary_path, index=False)

    report_path = outdir / "benchmark_report.md"
    build_report(raw, grouped, report_path, budgets, seeds)

    pareto_path = outdir / "pareto_by_budget.png"
    plot_pareto(grouped, pareto_path, budgets)

    print("\nDone.")
    print(f"- Raw: {raw_path}")
    print(f"- Summary: {summary_path}")
    print(f"- Report: {report_path}")
    print(f"- Plot: {pareto_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
