"""
Generate a per-patient comparison table:
  rows = patients
  columns = true kidney frames + predicted frames from every extractor × scoring config
"""
from pathlib import Path

import pandas as pd

from kidney_pipeline.config import EXTRACTOR_NAMES, GROUND_TRUTH, ROOT_DIR

PRED_CONFIGS = [
    ("cosine", "cosine_k5"),
    ("cosine", "cosine_k10"),
    ("cosine", "cosine_k20"),
    ("svm",    "svm_nu0.01"),
    ("svm",    "svm_nu0.05"),
    ("svm",    "svm_nu0.10"),
    ("svm",    "svm_nu0.20"),
]

EXTRACTOR_LABELS = {
    "resnet50":       "ResNet50",
    "efficientnet_b0":"EffNet-B0",
    "densenet121":    "DenseNet121",
    "convnext_tiny":  "ConvNeXt-T",
    "dinov2_vits14":  "DINOv2-S",
    "ultrasam":       "UltraSAM",
}

CONFIG_LABELS = {
    "cosine_k5":   "Cos k=5",
    "cosine_k10":  "Cos k=10",
    "cosine_k20":  "Cos k=20",
    "svm_nu0.01":  "SVM 0.01",
    "svm_nu0.05":  "SVM 0.05",
    "svm_nu0.10":  "SVM 0.10",
    "svm_nu0.20":  "SVM 0.20",
}


def _get_predicted(csv_path: Path, cfg: str) -> list:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    col = f"predicted_{cfg}"
    if col not in df.columns:
        return None
    return sorted(df[df[col] == True]["frame_num"].tolist())


def _cell_class(predicted: list | None, true_set: set) -> str:
    if predicted is None:
        return "na"
    if not predicted:
        return "miss"  # no frames selected
    hits = [f for f in predicted if f in true_set]
    if len(hits) == len(true_set) and len(predicted) == len(true_set):
        return "perfect"  # exact match
    if hits:
        return "partial"  # at least one hit
    return "wrong"  # selected frames but none are kidney


def build_table():
    patients = [f"patient{str(i).zfill(3)}" for i in range(1, 21)]

    # --- CSV export (flat) ---
    csv_rows = []
    for pid in patients:
        true_frames = sorted(GROUND_TRUTH.get(pid, []))
        row = {
            "patient": pid,
            "true_kidney_frames": " | ".join(str(f) for f in true_frames),
        }
        true_set = set(true_frames)
        for ext in EXTRACTOR_NAMES:
            csv_path = ROOT_DIR / f"results_{ext}" / "patients" / pid / "all_rankings.csv"
            for _, cfg in PRED_CONFIGS:
                predicted = _get_predicted(csv_path, cfg)
                key = f"{EXTRACTOR_LABELS[ext]} / {CONFIG_LABELS[cfg]}"
                if predicted is None:
                    row[key] = "N/A"
                elif not predicted:
                    row[key] = "—"
                else:
                    row[key] = " | ".join(str(f) for f in predicted)
        csv_rows.append(row)

    csv_df = pd.DataFrame(csv_rows)
    out_dir = ROOT_DIR / "results_final_comparison"
    csv_df.to_csv(out_dir / "per_patient_predictions.csv", index=False)
    print(f"Saved per_patient_predictions.csv")

    # --- HTML export ---
    _write_html(patients, out_dir / "per_patient_predictions.html")
    print(f"Saved per_patient_predictions.html")


def _write_html(patients: list, out_path: Path):
    lines = []

    # Header row — two-level: extractor / config
    # Build column groups
    col_groups = []  # (ext, cfg, label_ext, label_cfg)
    for ext in EXTRACTOR_NAMES:
        for _, cfg in PRED_CONFIGS:
            col_groups.append((ext, cfg, EXTRACTOR_LABELS[ext], CONFIG_LABELS[cfg]))

    # Extractor header spans
    ext_spans: list[tuple[str, int]] = []
    for ext in EXTRACTOR_NAMES:
        ext_spans.append((EXTRACTOR_LABELS[ext], len(PRED_CONFIGS)))

    header1 = (
        '<th rowspan="2" style="background:#2c3e50;color:white;">Patient</th>'
        '<th rowspan="2" style="background:#27ae60;color:white;">True Kidney<br>Frame #s</th>'
    )
    for label, span in ext_spans:
        header1 += f'<th colspan="{span}" style="background:#2980b9;color:white;">{label}</th>'

    header2 = ""
    for ext, cfg, _, cfg_label in col_groups:
        bg = "#3498db" if cfg.startswith("cosine") else "#8e44ad"
        header2 += f'<th style="background:{bg};color:white;font-size:11px;">{cfg_label}</th>'

    # Data rows
    data_rows = []
    for pid in patients:
        true_frames = sorted(GROUND_TRUTH.get(pid, []))
        true_set = set(true_frames)
        true_str = ", ".join(str(f) for f in true_frames)
        pid_label = f"Patient {int(pid.replace('patient',''))}"

        cells = (
            f'<td style="font-weight:bold;background:#ecf0f1;">{pid_label}</td>'
            f'<td style="background:#d5f5e3;text-align:center;font-weight:bold;">{true_str}</td>'
        )

        for ext, cfg, _, _ in col_groups:
            csv_path = ROOT_DIR / f"results_{ext}" / "patients" / pid / "all_rankings.csv"
            predicted = _get_predicted(csv_path, cfg)
            status = _cell_class(predicted, true_set)

            if predicted is None:
                disp = "N/A"
            elif not predicted:
                disp = "—"
            else:
                parts = []
                for f in predicted:
                    if f in true_set:
                        parts.append(f'<span style="color:#27ae60;font-weight:bold;">{f}</span>')
                    else:
                        parts.append(f'<span style="color:#e74c3c;">{f}</span>')
                disp = ", ".join(parts)

            bg_map = {
                "perfect": "#d5f5e3",
                "partial": "#fef9e7",
                "wrong":   "#fdecea",
                "miss":    "#f5f5f5",
                "na":      "#eeeeee",
            }
            bg = bg_map.get(status, "#ffffff")
            cells += f'<td style="background:{bg};text-align:center;font-size:12px;">{disp}</td>'

        data_rows.append(f"<tr>{cells}</tr>")

    # Legend
    legend = """
    <div style="margin-top:20px;font-size:13px;">
      <b>Legend:</b>
      <span style="margin-left:10px;padding:2px 8px;background:#d5f5e3;border-radius:3px;">Perfect match</span>
      <span style="margin-left:10px;padding:2px 8px;background:#fef9e7;border-radius:3px;">Partial match</span>
      <span style="margin-left:10px;padding:2px 8px;background:#fdecea;border-radius:3px;">All wrong</span>
      <span style="margin-left:10px;padding:2px 8px;background:#f5f5f5;border-radius:3px;">None predicted</span>
      &nbsp; Frame numbers: <span style="color:#27ae60;font-weight:bold;">green = correct kidney</span>,
      <span style="color:#e74c3c;">red = false positive</span>
    </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Per-Patient Kidney Frame Predictions</title>
<style>
  body {{ font-family: Arial, sans-serif; padding: 20px; font-size: 13px; }}
  h1 {{ color: #2c3e50; font-size: 18px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #bdc3c7; padding: 5px 8px; white-space: nowrap; }}
  tr:hover {{ filter: brightness(0.97); }}
  th {{ text-align: center; }}
</style>
</head>
<body>
<h1>Per-Patient Kidney Frame Predictions — All Extractors &amp; Scoring Methods</h1>
<p style="color:#666;">
  Blue columns = cosine similarity (k=5/10/20) &nbsp;|&nbsp;
  Purple columns = One-Class SVM (nu=0.01/0.05/0.10/0.20)
</p>
<div style="overflow-x:auto;">
<table>
<thead>
  <tr>{header1}</tr>
  <tr>{header2}</tr>
</thead>
<tbody>
{"".join(data_rows)}
</tbody>
</table>
</div>
{legend}
</body>
</html>"""

    out_path.write_text(html)


if __name__ == "__main__":
    build_table()
