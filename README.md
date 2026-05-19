# KIdney_Ultrasound_Identification
Testing dffferent methods to differentiate kidney form other organs in abdominnal utlrasound images


# Kidney Ultrasound Image Identification

This project identifies kidney ultrasound images from patient-wise folders containing ultrasound images of multiple organs.

## Goal

Given a folder of images for each patient, predict which image(s) correspond to the kidney.

```text
patient001/
├── frame1.png
├── frame2.png
├── frame3.png
└── ...
```

The output is a ranked list of likely kidney images, along with final predictions.

## Current Approach

We use pretrained feature extractors to convert each ultrasound image into an embedding vector. These embeddings are scored using kidney-reference images.

The current methods include:

- **Top-k mean cosine similarity** against a bank of confirmed kidney ultrasound images.
- **One-Class SVM** trained only on confirmed kidney image embeddings.

For cosine similarity, different values of `k` are tested:

```text
k = 5, 10, 20
```

For One-Class SVM, different `nu` values are tested:

```text
nu = 0.01, 0.05, 0.10, 0.20
```

## Evaluation

The methods are evaluated on 20 labelled patient folders where the true kidney frame numbers are known.

Metrics used:

- Top-1 accuracy
- Top-3 accuracy
- Top-5 accuracy
- Mean Reciprocal Rank
- Precision
- Recall
- F1-score
- False positives per patient

## Ensemble Method

We are building an ensemble on top of the best-performing feature extractor and scoring-method combinations.

The ensemble combines the strongest individual models and produces a final kidney-likeness score for each image.

Since selecting a non-kidney image is more harmful than missing one kidney image, the ensemble is designed to penalize **false positives much more strongly than false negatives**.

```text
false_positive_penalty >> false_negative_penalty
```

The final prediction rule is conservative:

```text
predict kidney only when multiple strong methods agree
```

## Output

For each patient, the pipeline saves:

```text
outputs/
├── patient001/
│   ├── predicted_kidney/
│   └── ranking.csv
├── patient002/
│   ├── predicted_kidney/
│   └── ranking.csv
└── all_patient_predictions.csv
```

Each ranking file contains:

- Patient ID
- Image path
- Frame number
- Individual model scores
- Ensemble score
- Final prediction
- Rank
- True/false positive status, if ground truth is available

## Summary

This project uses feature extraction, kidney-reference similarity, One-Class SVM, and a conservative ensemble to identify kidney ultrasound images from mixed-organ patient folders. The main priority of the ensemble is to reduce false positives while still retrieving likely kidney images.