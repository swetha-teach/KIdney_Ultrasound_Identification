from typing import Dict, List, Set

import numpy as np

from .config import COSINE_K_VALUES, SVM_NU_VALUES


def _scoring_configs() -> List[str]:
    configs = [f"cosine_k{k}" for k in COSINE_K_VALUES]
    configs += [f"svm_nu{nu:.2f}" for nu in SVM_NU_VALUES]
    return configs


def evaluate_ranking(
    ranked_frames: List[int],
    true_kidney_frames: Set[int],
    n_total: int,
) -> dict:
    """
    ranked_frames: frame numbers in ranked order (best first).
    Returns first_rank (1-based, or n_total+1 if no kidney found), and top-k booleans.
    """
    first_rank = n_total + 1
    for rank, frame in enumerate(ranked_frames, start=1):
        if frame in true_kidney_frames:
            first_rank = rank
            break
    return {
        "first_rank": first_rank,
        "in_top1": first_rank <= 1,
        "in_top3": first_rank <= 3,
        "in_top5": first_rank <= 5,
    }


def aggregate_metrics(per_patient: List[dict]) -> dict:
    n = len(per_patient)
    if n == 0:
        return {"top1_acc": 0, "top3_acc": 0, "top5_acc": 0, "mrr": 0}
    top1 = sum(p["in_top1"] for p in per_patient) / n
    top3 = sum(p["in_top3"] for p in per_patient) / n
    top5 = sum(p["in_top5"] for p in per_patient) / n
    mrr = float(np.mean([1.0 / p["first_rank"] for p in per_patient]))
    return {"top1_acc": top1, "top3_acc": top3, "top5_acc": top5, "mrr": mrr}


def evaluate_all_configs(
    patient_results: List[dict],
) -> Dict[str, dict]:
    """
    patient_results: list of per-patient dicts, each containing
        {patient_id, ranked_frames_<config>: List[int], true_frames: Set[int], n_images: int}
    Returns a dict keyed by config name with aggregated metrics.
    """
    configs = _scoring_configs()
    config_patient_evals: Dict[str, List[dict]] = {c: [] for c in configs}

    for pr in patient_results:
        true_frames = pr["true_frames"]
        n = pr["n_images"]
        for cfg in configs:
            ranked = pr[f"ranked_frames_{cfg}"]
            ev = evaluate_ranking(ranked, true_frames, n)
            ev["patient_id"] = pr["patient_id"]
            config_patient_evals[cfg].append(ev)

    return {cfg: aggregate_metrics(evals) for cfg, evals in config_patient_evals.items()}
