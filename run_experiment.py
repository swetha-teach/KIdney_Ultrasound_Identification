"""
Kidney Frame Retrieval Experiment
==================================
Compares 6 feature extractors × 7 scoring configurations (3 cosine k-values + 4 SVM nu-values)
for identifying kidney ultrasound frames in mixed-organ patient folders.

Usage:
    python run_experiment.py                          # all extractors
    python run_experiment.py --extractors resnet50    # single extractor
    python run_experiment.py --skip-cache             # force re-extract embeddings
    python run_experiment.py --no-copy-images         # skip copying top-5 images
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from kidney_pipeline.config import (
    COSINE_K_VALUES,
    EXTRACTOR_NAMES,
    GROUND_TRUTH,
    PATIENTS_DIR,
    REFERENCE_DIR,
    ROOT_DIR,
    SVM_NU_VALUES,
    frame_number,
    get_image_paths,
)
from kidney_pipeline.evaluation import evaluate_all_configs
from kidney_pipeline.extractors import load_extractor
from kidney_pipeline.reporting import (
    copy_top_candidates,
    generate_final_report,
    save_extractor_metrics,
    save_patient_csv,
)
from kidney_pipeline.scoring import (
    compute_dynamic_predictions,
    score_all_images,
    train_ocsvm,
)


def run_extractor(
    extractor_name: str,
    skip_cache: bool,
    copy_images: bool,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Extractor: {extractor_name}")
    print(f"{'=' * 60}")

    extractor = load_extractor(extractor_name)
    results_dir = ROOT_DIR / f"results_{extractor_name}"
    embed_dir = results_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    # ── Reference bank ────────────────────────────────────────────
    ref_cache = embed_dir / "reference.npy"
    if skip_cache and ref_cache.exists():
        ref_cache.unlink()

    ref_paths = get_image_paths(REFERENCE_DIR)
    print(f"\nReference images: {len(ref_paths)}")
    reference_bank = extractor.build_reference_bank(ref_paths, ref_cache)
    print(f"  Reference bank shape: {reference_bank.shape}")

    # ── Train all OCSVM models on reference bank ──────────────────
    print("\nTraining One-Class SVMs...")
    ocsvms: Dict[float, object] = {}
    for nu in SVM_NU_VALUES:
        print(f"  nu={nu:.2f} ... ", end="", flush=True)
        ocsvms[nu] = train_ocsvm(reference_bank, nu)
        print("done")

    # ── Process each patient ──────────────────────────────────────
    patient_dirs = sorted([p for p in PATIENTS_DIR.iterdir() if p.is_dir()])
    print(f"\nProcessing {len(patient_dirs)} patients...")

    patient_results: List[dict] = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        image_paths = get_image_paths(patient_dir)
        if not image_paths:
            continue

        frame_nums = [frame_number(p) for p in image_paths]
        true_frames = set(GROUND_TRUTH.get(patient_id, []))
        n = len(image_paths)

        print(f"\n  {patient_id}: {n} images | true kidney frames: {sorted(true_frames)}")

        # Load or compute patient embeddings
        pat_cache = embed_dir / f"{patient_id}.npy"
        if skip_cache and pat_cache.exists():
            pat_cache.unlink()
        patient_embeddings = extractor.build_patient_embeddings(image_paths, pat_cache)

        # Score all images against all configs
        scores = score_all_images(patient_embeddings, reference_bank, ocsvms)

        # Dynamic threshold predictions
        predictions = compute_dynamic_predictions(scores)

        # Save per-patient CSV
        patient_out = results_dir / "patients" / patient_id
        save_patient_csv(
            patient_out,
            patient_id,
            image_paths,
            frame_nums,
            scores,
            predictions,
            true_frames,
        )

        # Copy top-5 candidate images (by cosine_k10)
        if copy_images:
            copy_top_candidates(patient_out, image_paths, scores["cosine_k10"], n_top=5)

        # Build ranked frame lists per config for evaluation
        pr: dict = {
            "patient_id": patient_id,
            "true_frames": true_frames,
            "n_images": n,
        }
        configs = [f"cosine_k{k}" for k in COSINE_K_VALUES]
        configs += [f"svm_nu{nu:.2f}" for nu in SVM_NU_VALUES]
        for cfg in configs:
            cfg_scores = scores[cfg]
            order = np.argsort(cfg_scores)[::-1].tolist()
            pr[f"ranked_frames_{cfg}"] = [frame_nums[i] for i in order]

        patient_results.append(pr)

    # ── Evaluate across all patients ──────────────────────────────
    print(f"\nEvaluating metrics for {extractor_name}...")
    metrics = evaluate_all_configs(patient_results)
    save_extractor_metrics(results_dir, metrics)

    print(f"\n  Results saved to: {results_dir}")
    print(f"\n  Metrics summary:")
    for cfg, m in sorted(metrics.items(), key=lambda x: -x[1]["mrr"]):
        print(
            f"    {cfg:<20}  top1={m['top1_acc']:.3f}  "
            f"top3={m['top3_acc']:.3f}  top5={m['top5_acc']:.3f}  MRR={m['mrr']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Kidney frame retrieval experiment")
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=EXTRACTOR_NAMES,
        choices=EXTRACTOR_NAMES,
        metavar="NAME",
        help=f"Extractors to run (default: all). Choices: {EXTRACTOR_NAMES}",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Force re-extraction of embeddings even if cache exists",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Skip copying top-5 candidate images (faster)",
    )
    args = parser.parse_args()

    copy_images = not args.no_copy_images

    print(f"Extractors to run: {args.extractors}")
    print(f"Patients dir     : {PATIENTS_DIR}")
    print(f"Reference dir    : {REFERENCE_DIR}")

    for name in args.extractors:
        run_extractor(name, skip_cache=args.skip_cache, copy_images=copy_images)

    print(f"\n{'=' * 60}")
    print("  Generating final comparison report...")
    print(f"{'=' * 60}")
    generate_final_report(ROOT_DIR, args.extractors)
    print("\nDone.")


if __name__ == "__main__":
    main()
