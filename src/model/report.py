"""Result tables, selection note and figures for the cross-validated comparison."""

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model.cv import METRIC_NAMES, MODEL_ORDER, PL_MODELS, CvResults, Selection, summarise_metrics
from src.model.data import StudyData

TOP_INDICATORS = 3


def git_commit(root: Path) -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def header_lines(study: StudyData, root: Path) -> list[str]:
    return [f"# sha256_t1={study.sha256_t1}", f"# sha256_t2={study.sha256_t2}", f"# git_commit={git_commit(root)}"]


def write_csv_with_header(df: pd.DataFrame, path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n" + df.to_csv(index=False))


def coefficient_summary(coefs: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = coefs[coefs["model"] == model]
    wide = sub.pivot(index="feature", columns="fold", values="beta")
    out = pd.DataFrame({
        "mean_beta": wide.mean(axis=1),
        "sd_across_folds": wide.std(axis=1, ddof=1),
        "same_sign_folds": wide.apply(lambda r: int((np.sign(r) == np.sign(r.mean())).sum()), axis=1),
    })
    out["abs_mean_beta"] = out["mean_beta"].abs()
    return out.join(wide.add_prefix("fold_")).sort_values("abs_mean_beta", ascending=False)


def model_table(summary: pd.DataFrame) -> pd.DataFrame:
    ll = summary[summary["metric"] == "heldout_ll"].set_index("model")
    ordered = [m for m in MODEL_ORDER if m in ll.index]
    return ll.loc[ordered].drop(columns="metric").sort_values("mean", ascending=False)


def markdown_table(df: pd.DataFrame, digits: int) -> str:
    cols = [df.index.name or "model"] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, (int, float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def write_selection(study: StudyData, root: Path, results: CvResults, summary: pd.DataFrame, selection: Selection, path: Path) -> None:
    ll = model_table(summary)
    ll.index.name = "model"
    other_metrics = summary[summary["metric"] != "heldout_ll"].pivot(index="model", columns="metric", values="mean")
    other_metrics = other_metrics.loc[[m for m in MODEL_ORDER if m in other_metrics.index], [m for m in METRIC_NAMES if m != "heldout_ll"]]
    other_metrics.index.name = "model"
    published_coefs = coefficient_summary(results.coefficients, selection.published)
    top = published_coefs.head(TOP_INDICATORS)
    lam = results.tuning[results.tuning["model"].isin(PL_MODELS)].pivot(index="fold", columns="model", values="lambda_or_alpha")
    diag = results.diagnostics
    text = [
        "# Model selection (spec rule 5.4)",
        "",
        *header_lines(study, root),
        "",
        "Primary metric: mean held-out Plackett-Luce log-likelihood per voted player over four leave-one-edition-out folds (higher is better). "
        "No 2026 output was used.",
        "",
        "## Held-out log-likelihood per voted player",
        "",
        markdown_table(ll, 3),
        "",
        "U0 is a uniform-utility reference (a model that knows nothing); any useful model must beat it.",
        "",
        "## Selection outcome",
        "",
        f"**Published model: {selection.published}.** {selection.reported_alongside} is reported alongside.",
        "",
        *[f"- {n}" for n in selection.notes],
        "",
        "## Secondary metrics (mean over folds)",
        "",
        markdown_table(other_metrics, 3),
        "",
        f"## Back-test coefficients of {selection.published} (provisional: per-fold fits, final fit and intervals come later)",
        "",
        f"Top {TOP_INDICATORS} by absolute mean weight: " + ", ".join(f"**{f}** ({r.mean_beta:+.2f}, same sign in {int(r.same_sign_folds)}/4 folds)" for f, r in top.iterrows()),
        "",
        markdown_table(published_coefs.rename_axis("feature"), 3),
        "",
        "## Selected penalty per fold (PL models)",
        "",
        markdown_table(lam.rename_axis("fold"), 3),
        "",
        "## Diagnostics",
        "",
        markdown_table(diag.set_index("fold")[["diagnostic", "value"]], 3) if len(diag) else "none",
        "",
        "The B2 random forest produces win probabilities that, before renormalisation, sum to the values above (a proper distribution sums to 1).",
    ]
    path.write_text("\n".join(text) + "\n")


def plot_heldout_ll(metrics: pd.DataFrame, path: Path) -> None:
    ll = metrics[metrics["metric"] == "heldout_ll"]
    models = [m for m in MODEL_ORDER if m in set(ll["model"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, model in enumerate(models):
        values = ll[ll["model"] == model].sort_values("fold")["value"].to_numpy()
        ax.scatter(values, [i] * len(values), color="grey", alpha=0.75, s=30, zorder=2)
        ax.scatter(values.mean(), i, s=110, color="tab:blue", edgecolor="k", zorder=3)
    uniform = ll[ll["model"] == "U0"]["value"].mean() if "U0" in models else None
    if uniform is not None:
        ax.axvline(uniform, color="red", linestyle="--", label="uniform reference (mean)")
        ax.legend(loc="lower left")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("held-out log-likelihood per voted player (higher is better)")
    ax.set_title("Leave-one-edition-out back-test (grey = each held-out edition, blue = mean)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def write_all(study: StudyData, root: Path, results: CvResults, selection: Selection) -> pd.DataFrame:
    summary = summarise_metrics(results.metrics)
    lines = header_lines(study, root)
    (root / "results").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    write_csv_with_header(results.metrics, root / "results" / "metrics_by_fold.csv", lines)
    write_csv_with_header(summary, root / "results" / "metrics_summary.csv", lines)
    write_csv_with_header(results.coefficients, root / "results" / "coefficients_by_fold.csv", lines)
    write_selection(study, root, results, summary, selection, root / "results" / "selection.md")
    plot_heldout_ll(results.metrics, root / "figures" / "heldout_ll_by_model.png")
    return summary
