"""
Ensemble Democracy for Kidney Frame Selection
==============================================
Selects top-N models by MRR, applies MRR-weighted voting with a high threshold
(penalising false positives heavily).  Outputs an HTML report with per-frame
vote breakdown, model contributions, and per-patient precision/recall.

Usage:
    python run_ensemble.py                   # default: top-10 models, threshold 0.60
    python run_ensemble.py --top-n 7 --threshold 0.65
    python run_ensemble.py --top-n 42        # all models
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from kidney_pipeline.config import GROUND_TRUTH, ROOT_DIR

OUT_DIR = ROOT_DIR / "results_final_comparison"
METRICS_CSV = OUT_DIR / "full_comparison_table.csv"

# Short labels for display
_EXT_SHORT = {
    "resnet50":        "R50",
    "efficientnet_b0": "EN0",
    "densenet121":     "DN1",
    "convnext_tiny":   "CNT",
    "dinov2_vits14":   "DV2",
    "ultrasam":        "SAM",
}
_CFG_SHORT = {
    "cosine_k5":   "C5",
    "cosine_k10":  "C10",
    "cosine_k20":  "C20",
    "svm_nu0.01":  "S1",
    "svm_nu0.05":  "S5",
    "svm_nu0.10":  "S10",
    "svm_nu0.20":  "S20",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_top_models(n: int) -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV).sort_values("mrr", ascending=False).head(n)
    return df.reset_index(drop=True)


def load_patient_votes(patient_id: str, models: pd.DataFrame) -> pd.DataFrame | None:
    """
    Returns a DataFrame indexed by frame_num with one boolean column per model.
    Model columns are named '{extractor}|{config}'.
    """
    frames_data = {}
    frame_nums = None

    for _, m in models.iterrows():
        ext, cfg = m["extractor"], m["config"]
        key = f"{ext}|{cfg}"
        csv_path = ROOT_DIR / f"results_{ext}" / "patients" / patient_id / "all_rankings.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        pred_col = f"predicted_{cfg}"
        if pred_col not in df.columns:
            continue
        mapping = dict(zip(df["frame_num"], df[pred_col].astype(bool)))
        if frame_nums is None:
            frame_nums = sorted(mapping.keys())
        frames_data[key] = [mapping.get(f, False) for f in frame_nums]

    if frame_nums is None:
        return None
    return pd.DataFrame(frames_data, index=frame_nums)


def weighted_vote(votes_df: pd.DataFrame, models: pd.DataFrame, threshold: float):
    """
    Returns (vote_fraction, predicted) Series indexed by frame_num.
    vote_fraction = sum(mrr * vote) / sum(mrr) per frame.
    predicted = vote_fraction >= threshold.
    """
    mrr_weights = {}
    for _, m in models.iterrows():
        key = f"{m['extractor']}|{m['config']}"
        if key in votes_df.columns:
            mrr_weights[key] = m["mrr"]

    total_w = sum(mrr_weights.values())
    if total_w == 0:
        return pd.Series(0.0, index=votes_df.index), pd.Series(False, index=votes_df.index)

    frac = pd.Series(0.0, index=votes_df.index)
    for key, w in mrr_weights.items():
        frac += votes_df[key].astype(float) * w
    frac /= total_w

    return frac, frac >= threshold


# ── metrics ──────────────────────────────────────────────────────────────────

def patient_metrics(predicted_frames: set, true_frames: set, all_frames: set) -> dict:
    tp = len(predicted_frames & true_frames)
    fp = len(predicted_frames - true_frames)
    fn = len(true_frames - predicted_frames)
    tn = len((all_frames - predicted_frames) - true_frames)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


# ── HTML generation ───────────────────────────────────────────────────────────

def _model_header_cells(models: pd.DataFrame) -> str:
    cells = ""
    for _, m in models.iterrows():
        short = f"{_EXT_SHORT.get(m['extractor'], m['extractor'][:3])}/{_CFG_SHORT.get(m['config'], m['config'][:4])}"
        mrr = m["mrr"]
        bg = "#3498db" if m["scoring_method"] == "cosine" else "#8e44ad"
        cells += (
            f'<th style="background:{bg};color:white;font-size:10px;'
            f'writing-mode:vertical-rl;transform:rotate(180deg);padding:6px 3px;'
            f'min-width:24px;" title="{m["extractor"]} / {m["config"]} MRR={mrr:.4f}">'
            f'{short}</th>'
        )
    return cells


def _vote_bar(frac: float, threshold: float) -> str:
    pct = int(frac * 100)
    bar_color = "#27ae60" if frac >= threshold else "#e67e22" if frac >= threshold * 0.7 else "#bdc3c7"
    thr_pct = int(threshold * 100)
    return (
        f'<div style="position:relative;width:64px;height:14px;background:#ecf0f1;border-radius:3px;">'
        f'<div style="width:{pct}%;height:100%;background:{bar_color};border-radius:3px;"></div>'
        f'<div style="position:absolute;left:{thr_pct}%;top:0;width:2px;height:100%;background:#e74c3c;"></div>'
        f'<span style="position:absolute;top:0;left:2px;font-size:9px;line-height:14px;color:#333;">{pct}%</span>'
        f'</div>'
    )


def build_html(
    models: pd.DataFrame,
    threshold: float,
    all_results: list,
    summary_rows: list,
) -> str:

    model_th = _model_header_cells(models)

    # --- model legend strip ---
    legend_items = []
    for _, m in models.iterrows():
        short = f"{_EXT_SHORT.get(m['extractor'], m['extractor'])}/{_CFG_SHORT.get(m['config'], m['config'])}"
        bg = "#3498db" if m["scoring_method"] == "cosine" else "#8e44ad"
        legend_items.append(
            f'<span style="display:inline-block;margin:2px 6px;padding:2px 7px;'
            f'background:{bg};color:white;border-radius:3px;font-size:11px;" '
            f'title="{m["extractor"]} / {m["config"]}">{short} <b style="opacity:.8">{m["mrr"]:.3f}</b></span>'
        )
    legend_strip = "".join(legend_items)

    # --- summary table ---
    sum_rows_html = ""
    for sr in summary_rows:
        prec_color = "#27ae60" if sr["precision"] >= 0.8 else "#e67e22" if sr["precision"] >= 0.5 else "#e74c3c"
        rec_color  = "#27ae60" if sr["recall"]    >= 0.8 else "#e67e22" if sr["recall"]    >= 0.5 else "#e74c3c"
        sum_rows_html += (
            f'<tr>'
            f'<td style="font-weight:bold">{sr["patient"]}</td>'
            f'<td style="color:#27ae60">{sr["true_frames"]}</td>'
            f'<td style="color:#2c3e50">{sr["predicted_frames"]}</td>'
            f'<td style="color:#27ae60">{sr["tp"]}</td>'
            f'<td style="color:#e74c3c">{sr["fp"]}</td>'
            f'<td style="color:#e67e22">{sr["fn"]}</td>'
            f'<td style="color:{prec_color};font-weight:bold">{sr["precision"]:.0%}</td>'
            f'<td style="color:{rec_color};font-weight:bold">{sr["recall"]:.0%}</td>'
            f'<td style="font-weight:bold">{sr["f1"]:.0%}</td>'
            f'</tr>'
        )

    # aggregate
    n = len(summary_rows)
    avg_prec = np.mean([r["precision"] for r in summary_rows])
    avg_rec  = np.mean([r["recall"]    for r in summary_rows])
    avg_f1   = np.mean([r["f1"]        for r in summary_rows])
    tot_tp   = sum(r["tp"] for r in summary_rows)
    tot_fp   = sum(r["fp"] for r in summary_rows)
    tot_fn   = sum(r["fn"] for r in summary_rows)
    sum_rows_html += (
        f'<tr style="font-weight:bold;background:#2c3e50;color:white;">'
        f'<td>MEAN</td><td>—</td><td>—</td>'
        f'<td>{tot_tp}</td><td>{tot_fp}</td><td>{tot_fn}</td>'
        f'<td>{avg_prec:.0%}</td><td>{avg_rec:.0%}</td><td>{avg_f1:.0%}</td>'
        f'</tr>'
    )

    # --- per-patient detail sections ---
    details_html = ""
    for res in all_results:
        pid = res["patient_id"]
        pid_label = f"Patient {int(pid.replace('patient',''))}"
        true_frames = res["true_frames"]
        predicted = res["predicted"]
        votes_df = res["votes_df"]
        fracs = res["fracs"]

        rows_html = ""
        for frame_num in sorted(votes_df.index):
            frac = fracs[frame_num]
            is_pred = frame_num in predicted
            is_true = frame_num in true_frames

            if is_true and is_pred:
                row_bg, status = "#d5f5e3", "✓ TP"
            elif is_pred and not is_true:
                row_bg, status = "#fdecea", "✗ FP"
            elif is_true and not is_pred:
                row_bg, status = "#fef9e7", "△ FN"
            else:
                row_bg, status = "#f9f9f9", "TN"

            vote_cells = ""
            for _, m in models.iterrows():
                key = f"{m['extractor']}|{m['config']}"
                voted = bool(votes_df.at[frame_num, key]) if key in votes_df.columns else False
                v_bg = "#27ae60" if voted else "#ecf0f1"
                v_txt = "✓" if voted else "·"
                vote_cells += (
                    f'<td style="text-align:center;background:{v_bg};color:{"white" if voted else "#aaa"};'
                    f'font-size:11px;padding:3px;">{v_txt}</td>'
                )

            rows_html += (
                f'<tr style="background:{row_bg};">'
                f'<td style="text-align:center;font-weight:bold;">Frame {frame_num}</td>'
                f'<td style="text-align:center;">{_vote_bar(frac, threshold)}</td>'
                f'<td style="text-align:center;font-weight:bold;font-size:13px;">'
                f'{"🟢 YES" if is_pred else "⬜ no"}</td>'
                f'<td style="text-align:center;">{"✓ Kidney" if is_true else "—"}</td>'
                f'<td style="text-align:center;font-size:12px;">{status}</td>'
                f'{vote_cells}'
                f'</tr>'
            )

        pred_str = ", ".join(str(f) for f in sorted(predicted)) or "—"
        true_str = ", ".join(str(f) for f in sorted(true_frames))
        m_res    = res["metrics"]
        tp_str   = f'TP={m_res["tp"]} FP={m_res["fp"]} FN={m_res["fn"]}'
        prec_str = f'Prec={m_res["precision"]:.0%} Rec={m_res["recall"]:.0%} F1={m_res["f1"]:.0%}'

        details_html += f"""
<details style="margin-bottom:12px;border:1px solid #bdc3c7;border-radius:6px;overflow:hidden;">
  <summary style="cursor:pointer;padding:10px 14px;background:#2c3e50;color:white;font-size:14px;
    list-style:none;display:flex;justify-content:space-between;align-items:center;">
    <span><b>{pid_label}</b> &nbsp; True: <span style="color:#2ecc71">{true_str}</span>
      &nbsp; Predicted: <span style="color:#f39c12">{pred_str}</span></span>
    <span style="font-size:12px;opacity:.8">{tp_str} &nbsp;|&nbsp; {prec_str}</span>
  </summary>
  <div style="overflow-x:auto;">
  <table style="border-collapse:collapse;width:100%;font-size:12px;">
    <thead>
      <tr style="background:#34495e;color:white;">
        <th style="padding:6px 10px;">Frame</th>
        <th style="padding:6px 10px;">Vote%</th>
        <th style="padding:6px 10px;">Predicted</th>
        <th style="padding:6px 10px;">True label</th>
        <th style="padding:6px 10px;">Status</th>
        {model_th}
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  </div>
</details>
"""

    threshold_pct = int(threshold * 100)
    n_models = len(models)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Ensemble Democracy — Kidney Frame Selection</title>
<style>
  body {{ font-family: Arial, sans-serif; padding: 24px; background: #f4f6f8; color: #2c3e50; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  h2 {{ font-size: 15px; margin: 20px 0 8px; color: #2c3e50; }}
  .card {{ background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px;
           box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; }}
  th {{ background: #34495e; color: white; text-align: center; }}
  td {{ text-align: center; }}
  details summary::-webkit-details-marker {{ display: none; }}
</style>
</head>
<body>

<h1>🗳️ Ensemble Democracy — Kidney Frame Selection</h1>
<p style="color:#7f8c8d;margin-top:0;">
  Top <b>{n_models} models</b> by MRR · Weighted vote threshold: <b>{threshold_pct}%</b>
  (high threshold penalises false positives)
</p>

<div class="card">
  <b>Voters ({n_models} models, ordered by MRR):</b><br><br>
  {legend_strip}
  <br><br>
  <span style="font-size:12px;color:#666;">
    Vote bar: green = predicted kidney (≥{threshold_pct}%), red line = threshold.
    Blue label = cosine similarity &nbsp;|&nbsp; Purple label = One-Class SVM
  </span>
</div>

<div class="card">
<h2 style="margin-top:0;">Summary — All 20 Patients</h2>
<table>
  <thead>
    <tr>
      <th>Patient</th><th>True kidney frames</th><th>Predicted frames</th>
      <th>TP</th><th>FP</th><th>FN</th>
      <th>Precision</th><th>Recall</th><th>F1</th>
    </tr>
  </thead>
  <tbody>{sum_rows_html}</tbody>
</table>
</div>

<div class="card">
<h2 style="margin-top:0;">Per-Patient Vote Breakdown</h2>
<p style="font-size:12px;color:#666;">
  ✓ = model voted kidney &nbsp; · = not kidney
  &nbsp;|&nbsp; ✓ TP = true positive, ✗ FP = false positive,
  △ FN = missed kidney, TN = correct non-kidney
</p>
{details_html}
</div>

</body>
</html>"""
    return html


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top models (by MRR) to include in the vote")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Minimum weighted-vote fraction to classify as kidney (0–1). "
                             "Higher = fewer false positives")
    args = parser.parse_args()

    models = load_top_models(args.top_n)
    threshold = args.threshold

    print(f"Ensemble: top {len(models)} models, threshold={threshold:.0%}")
    print(models[["extractor", "config", "mrr"]].to_string(index=False))
    print()

    patients = [f"patient{str(i).zfill(3)}" for i in range(1, 21)]
    all_results = []
    summary_rows = []

    for pid in patients:
        true_frames = set(GROUND_TRUTH.get(pid, []))
        votes_df = load_patient_votes(pid, models)
        if votes_df is None:
            print(f"  {pid}: no data, skipping")
            continue

        fracs, predicted_series = weighted_vote(votes_df, models, threshold)
        predicted = set(votes_df.index[predicted_series].tolist())
        all_frames = set(votes_df.index.tolist())

        m = patient_metrics(predicted, true_frames, all_frames)
        pid_label = f"Patient {int(pid.replace('patient',''))}"
        summary_rows.append({
            "patient": pid_label,
            "true_frames": ", ".join(str(f) for f in sorted(true_frames)),
            "predicted_frames": ", ".join(str(f) for f in sorted(predicted)) or "—",
            **m,
        })
        all_results.append({
            "patient_id": pid,
            "true_frames": true_frames,
            "predicted": predicted,
            "votes_df": votes_df,
            "fracs": fracs,
            "metrics": m,
        })

        p_str = ", ".join(str(f) for f in sorted(predicted)) or "—"
        print(f"  {pid_label}  true={sorted(true_frames)}  pred={p_str}  "
              f"P={m['precision']:.0%} R={m['recall']:.0%} F1={m['f1']:.0%}  "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']}")

    # --- HTML ---
    html = build_html(models, threshold, all_results, summary_rows)
    out_html = OUT_DIR / "ensemble_predictions.html"
    out_html.write_text(html)
    print(f"\nSaved HTML → {out_html}")

    # --- CSV summary ---
    csv_rows = []
    for res in all_results:
        pid  = res["patient_id"]
        fracs = res["fracs"]
        votes_df = res["votes_df"]
        true_frames = res["true_frames"]
        predicted   = res["predicted"]
        for frame_num in sorted(votes_df.index):
            vote_pct = fracs[frame_num]
            per_model = {
                f"{m['extractor']}|{m['config']}":
                    bool(votes_df.at[frame_num, f"{m['extractor']}|{m['config']}"])
                    if f"{m['extractor']}|{m['config']}" in votes_df.columns else None
                for _, m in models.iterrows()
            }
            is_pred = frame_num in predicted
            is_true = frame_num in true_frames
            status = ("TP" if is_pred and is_true else
                      "FP" if is_pred and not is_true else
                      "FN" if not is_pred and is_true else "TN")
            csv_rows.append({
                "patient_id": pid,
                "frame_num": frame_num,
                "vote_fraction": round(vote_pct, 4),
                "predicted_kidney": is_pred,
                "true_kidney": is_true,
                "status": status,
                **{f"voted_{k.replace('|','_')}": v for k, v in per_model.items()},
            })
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / "ensemble_frame_votes.csv", index=False)
    print(f"Saved CSV  → {OUT_DIR / 'ensemble_frame_votes.csv'}")

    # --- overall ---
    avg_p  = np.mean([r["precision"] for r in summary_rows])
    avg_r  = np.mean([r["recall"]    for r in summary_rows])
    avg_f1 = np.mean([r["f1"]        for r in summary_rows])
    tot_tp = sum(r["tp"] for r in summary_rows)
    tot_fp = sum(r["fp"] for r in summary_rows)
    tot_fn = sum(r["fn"] for r in summary_rows)
    print(f"\nOverall — TP={tot_tp} FP={tot_fp} FN={tot_fn}")
    print(f"Mean Precision={avg_p:.0%}  Recall={avg_r:.0%}  F1={avg_f1:.0%}")


if __name__ == "__main__":
    main()
