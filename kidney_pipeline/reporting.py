import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import COSINE_K_VALUES, SVM_NU_VALUES


def _config_names() -> List[str]:
    configs = [f"cosine_k{k}" for k in COSINE_K_VALUES]
    configs += [f"svm_nu{nu:.2f}" for nu in SVM_NU_VALUES]
    return configs


def save_patient_csv(
    output_dir: Path,
    patient_id: str,
    image_paths: List[Path],
    frame_nums: List[int],
    scores: Dict[str, List],
    predictions: Dict[str, List[bool]],
    true_kidney_frames: set,
) -> None:
    configs = _config_names()
    rows = []
    n = len(image_paths)
    # Rank by cosine_k10 as the primary ordering for display
    primary_scores = scores["cosine_k10"]
    order = sorted(range(n), key=lambda i: primary_scores[i], reverse=True)

    for rank, i in enumerate(order, start=1):
        row = {
            "patient_id": patient_id,
            "image_path": str(image_paths[i]),
            "frame_num": frame_nums[i],
            "rank_by_cosine_k10": rank,
            "is_true_kidney": frame_nums[i] in true_kidney_frames,
        }
        for cfg in configs:
            row[cfg] = scores[cfg][i]
        for pred_key, pred_list in predictions.items():
            row[pred_key] = pred_list[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_rankings.csv", index=False)


def copy_top_candidates(
    output_dir: Path,
    image_paths: List[Path],
    scores: List[float],
    n_top: int = 5,
) -> None:
    dest = output_dir / "top5_candidates"
    dest.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(image_paths)), key=lambda i: scores[i], reverse=True)
    for rank, i in enumerate(order[:n_top], start=1):
        src = image_paths[i]
        shutil.copy(src, dest / f"rank_{rank}_{src.name}")


def save_extractor_metrics(
    output_dir: Path,
    metrics_per_config: Dict[str, dict],
) -> None:
    rows = []
    for cfg, m in metrics_per_config.items():
        if cfg.startswith("cosine"):
            method = "cosine"
            param = cfg.split("_k")[1]
        else:
            method = "svm"
            param = cfg.split("_nu")[1]
        rows.append({
            "scoring_method": method,
            "param": param,
            "config": cfg,
            "top1_acc": round(m["top1_acc"], 4),
            "top3_acc": round(m["top3_acc"], 4),
            "top5_acc": round(m["top5_acc"], 4),
            "mrr": round(m["mrr"], 4),
        })
    df = pd.DataFrame(rows).sort_values("mrr", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "metrics.csv", index=False)


def generate_final_report(
    results_root: Path,
    extractor_names: List[str],
) -> None:
    all_rows = []
    for name in extractor_names:
        metrics_path = results_root / f"results_{name}" / "metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        df.insert(0, "extractor", name)
        all_rows.append(df)

    if not all_rows:
        print("No metrics found; skipping final report.")
        return

    combined = pd.concat(all_rows, ignore_index=True).sort_values("mrr", ascending=False)
    out_dir = results_root / "results_final_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "full_comparison_table.csv", index=False)

    # HTML report with color formatting
    _write_html_report(combined, out_dir / "comparison_table.html")

    # Best configuration
    best = combined.iloc[0]
    summary = (
        f"Best Configuration\n"
        f"==================\n"
        f"Extractor     : {best['extractor']}\n"
        f"Method        : {best['scoring_method']}\n"
        f"Parameter     : {best['param']}\n"
        f"Config key    : {best['config']}\n"
        f"Top-1 Accuracy: {best['top1_acc']:.4f}\n"
        f"Top-3 Accuracy: {best['top3_acc']:.4f}\n"
        f"Top-5 Accuracy: {best['top5_acc']:.4f}\n"
        f"MRR           : {best['mrr']:.4f}\n"
    )
    (out_dir / "best_configuration.txt").write_text(summary)
    print("\n" + summary)


def _write_html_report(df: pd.DataFrame, path: Path) -> None:
    metric_cols = ["top1_acc", "top3_acc", "top5_acc", "mrr"]

    def _color_scale(series: pd.Series) -> list:
        mn, mx = series.min(), series.max()
        styles = []
        for v in series:
            if mx == mn:
                ratio = 0.5
            else:
                ratio = (v - mn) / (mx - mn)
            # green (high) → red (low)
            r = int(255 * (1 - ratio))
            g = int(200 * ratio)
            styles.append(f"background-color: rgb({r},{g},80); color: black;")
        return styles

    html_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        is_best = i == 0
        style = 'style="font-weight:bold; border: 2px solid #333;"' if is_best else ""
        cells = (
            f"<td {style}>{row['extractor']}</td>"
            f"<td {style}>{row['scoring_method']}</td>"
            f"<td {style}>{row['param']}</td>"
        )
        for col in metric_cols:
            cells += f"<td {style}>{row[col]:.4f}</td>"
        html_rows.append(f"<tr>{cells}</tr>")

    # Build per-column color styles
    col_styles: dict = {}
    for col in metric_cols:
        col_styles[col] = _color_scale(df[col])

    # Rebuild with colors
    html_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        is_best = i == 0
        bstyle = "font-weight:bold;" if is_best else ""
        cells = (
            f'<td style="{bstyle}">{row["extractor"]}</td>'
            f'<td style="{bstyle}">{row["scoring_method"]}</td>'
            f'<td style="{bstyle}">{row["param"]}</td>'
        )
        for col in metric_cols:
            cstyle = col_styles[col][i] + bstyle
            cells += f'<td style="{cstyle}">{row[col]:.4f}</td>'
        row_style = ' style="outline: 2px solid #1a1;"' if is_best else ""
        html_rows.append(f"<tr{row_style}>{cells}</tr>")

    headers = "".join(
        f"<th>{c}</th>"
        for c in ["extractor", "scoring_method", "param"] + metric_cols
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Kidney Frame Retrieval — Experiment Results</title>
<style>
  body {{ font-family: Arial, sans-serif; padding: 20px; }}
  h1 {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th {{ background: #444; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #ddd; }}
  tr:hover {{ filter: brightness(0.95); }}
</style>
</head>
<body>
<h1>Kidney Frame Retrieval — Experiment Results</h1>
<p>Sorted by MRR (descending). Best row outlined in green.</p>
<table>
<thead><tr>{headers}</tr></thead>
<tbody>
{"".join(html_rows)}
</tbody>
</table>
</body>
</html>"""
    path.write_text(html)
    print(f"  HTML report saved: {path}")
