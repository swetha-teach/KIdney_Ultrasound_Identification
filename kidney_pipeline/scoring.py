from typing import Dict, List, Tuple

import numpy as np
from sklearn.svm import OneClassSVM

from .config import COSINE_K_VALUES, GAP_MIN_THRESHOLD, SVM_NU_VALUES


def cosine_topk_score(query: np.ndarray, bank: np.ndarray, k: int) -> float:
    sims = bank @ query  # (N,) — both L2-normalized
    k_eff = min(k, len(sims))
    return float(np.partition(sims, -k_eff)[-k_eff:].mean())


def train_ocsvm(bank: np.ndarray, nu: float) -> OneClassSVM:
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    ocsvm.fit(bank)
    return ocsvm


def svm_score(query: np.ndarray, ocsvm: OneClassSVM) -> Tuple[float, bool]:
    q = query.reshape(1, -1)
    score = float(ocsvm.decision_function(q)[0])
    is_inlier = ocsvm.predict(q)[0] == 1
    return score, is_inlier


def gap_threshold(scores_desc: List[float], min_gap: float = GAP_MIN_THRESHOLD) -> int:
    """Return the number of images to keep based on the largest score drop."""
    n = len(scores_desc)
    if n <= 1:
        return 1
    arr = np.array(scores_desc)
    gaps = arr[:-1] - arr[1:]  # positive = drop between consecutive sorted scores
    idx = int(np.argmax(gaps))
    if gaps[idx] >= min_gap:
        return idx + 1
    # Fallback: keep images with score above the mean
    mean_score = float(arr.mean())
    cutoff = next((i for i, s in enumerate(scores_desc) if s < mean_score), n)
    return max(1, cutoff)


def score_all_images(
    patient_embeddings: np.ndarray,
    reference_bank: np.ndarray,
    ocsvms: Dict[float, OneClassSVM],
) -> Dict[str, List]:
    """
    Score every patient image against the reference bank with all configurations.
    Returns a dict of score lists (one value per image), keyed by config name.
    """
    n = len(patient_embeddings)
    scores: Dict[str, List] = {
        f"cosine_k{k}": [] for k in COSINE_K_VALUES
    }
    for nu in SVM_NU_VALUES:
        scores[f"svm_nu{nu:.2f}"] = []
        scores[f"svm_nu{nu:.2f}_is_inlier"] = []

    for i in range(n):
        q = patient_embeddings[i]
        for k in COSINE_K_VALUES:
            scores[f"cosine_k{k}"].append(cosine_topk_score(q, reference_bank, k))
        for nu, ocsvm in ocsvms.items():
            sc, inlier = svm_score(q, ocsvm)
            scores[f"svm_nu{nu:.2f}"].append(sc)
            scores[f"svm_nu{nu:.2f}_is_inlier"].append(inlier)

    return scores


def compute_dynamic_predictions(scores: Dict[str, List]) -> Dict[str, List[bool]]:
    """
    For each scoring config, produce a boolean list indicating which images
    are dynamically predicted as kidney.
    """
    predictions: Dict[str, List[bool]] = {}
    n = len(next(iter(scores.values())))

    for k in COSINE_K_VALUES:
        key = f"cosine_k{k}"
        vals = scores[key]
        ranked_idx = np.argsort(vals)[::-1].tolist()
        ranked_scores = [vals[i] for i in ranked_idx]
        keep = gap_threshold(ranked_scores)
        keep_set = set(ranked_idx[:keep])
        predictions[f"predicted_{key}"] = [i in keep_set for i in range(n)]

    for nu in SVM_NU_VALUES:
        key = f"svm_nu{nu:.2f}"
        predictions[f"predicted_{key}"] = scores[f"{key}_is_inlier"]

    return predictions
