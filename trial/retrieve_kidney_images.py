from pathlib import Path
import shutil

import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity


REFERENCE_DIR = Path("reference_kidney")
PATIENTS_DIR = Path("patient_folders")
OUTPUT_DIR = Path("retrieved_kidney_outputs")

TOP_K = 3
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_image_paths(folder: Path):
    return sorted([
        p for p in folder.rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    ])


def load_feature_extractor():
    """
    Uses ImageNet-pretrained ResNet50 as a generic feature extractor.
    We remove the final classification layer and keep the 2048-dimensional embedding.
    """

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Remove final fully connected classifier
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    preprocess = weights.transforms()

    return feature_extractor, preprocess


def extract_embedding(image_path: Path, model, preprocess):
    """
    Converts image to RGB, applies preprocessing, extracts normalized embedding.
    """

    image = Image.open(image_path).convert("RGB")
    x = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model(x)

    feat = feat.flatten(start_dim=1)
    feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)

    return feat.cpu()


def build_reference_bank(model, preprocess):
    reference_paths = get_image_paths(REFERENCE_DIR)

    if len(reference_paths) == 0:
        raise FileNotFoundError(f"No reference kidney images found in {REFERENCE_DIR}")

    embeddings = []

    print(f"Extracting embeddings from {len(reference_paths)} kidney reference images...")

    for path in tqdm(reference_paths):
        emb = extract_embedding(path, model, preprocess)
        embeddings.append(emb)

    reference_embeddings = torch.cat(embeddings, dim=0)

    return reference_paths, reference_embeddings


def score_image_against_reference(image_path, model, preprocess, reference_embeddings):
    query_embedding = extract_embedding(image_path, model, preprocess)

    sims = cosine_similarity(
        query_embedding.numpy(),
        reference_embeddings.numpy()
    )[0]

    # Different possible scores
    max_similarity = float(sims.max())
    mean_top5_similarity = float(sorted(sims, reverse=True)[:5][0]) if len(sims) < 5 else float(sum(sorted(sims, reverse=True)[:5]) / 5)

    return max_similarity, mean_top5_similarity


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Using device: {DEVICE}")

    model, preprocess = load_feature_extractor()
    reference_paths, reference_embeddings = build_reference_bank(model, preprocess)

    all_rows = []

    patient_dirs = sorted([
        p for p in PATIENTS_DIR.iterdir()
        if p.is_dir()
    ])

    if len(patient_dirs) == 0:
        raise FileNotFoundError(f"No patient folders found in {PATIENTS_DIR}")

    for patient_dir in patient_dirs:
        image_paths = get_image_paths(patient_dir)

        if len(image_paths) == 0:
            print(f"No images found for {patient_dir.name}")
            continue

        patient_rows = []

        print(f"\nProcessing {patient_dir.name} with {len(image_paths)} images...")

        for img_path in tqdm(image_paths):
            max_sim, top5_sim = score_image_against_reference(
                img_path,
                model,
                preprocess,
                reference_embeddings
            )

            row = {
                "patient_id": patient_dir.name,
                "image_path": str(img_path),
                "max_similarity_to_kidney_reference": max_sim,
                "top5_mean_similarity_to_kidney_reference": top5_sim,
            }

            patient_rows.append(row)
            all_rows.append(row)

        patient_df = pd.DataFrame(patient_rows)

        # Rank images by similarity to kidney references
        patient_df = patient_df.sort_values(
            by="max_similarity_to_kidney_reference",
            ascending=False
        )

        patient_output_dir = OUTPUT_DIR / patient_dir.name
        patient_output_dir.mkdir(exist_ok=True)

        # Save top-k kidney candidates
        top_candidates = patient_df.head(TOP_K)

        for rank, (_, row) in enumerate(top_candidates.iterrows(), start=1):
            src = Path(row["image_path"])
            dst = patient_output_dir / f"rank_{rank}_{src.name}"
            shutil.copy(src, dst)

        patient_df.to_csv(patient_output_dir / "ranking.csv", index=False)

        best = patient_df.iloc[0]
        print("Best candidate:")
        print(best["image_path"])
        print("similarity:", best["max_similarity_to_kidney_reference"])

    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(OUTPUT_DIR / "all_patient_rankings.csv", index=False)

    print("\nDone.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()