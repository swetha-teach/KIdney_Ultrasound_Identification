---
name: project-ultrasam-note
description: UltraSAM extractor currently uses SAM-ViT-B base; MedSAM fine-tuned weights need retry
metadata:
  type: project
---

Current `ultrasam` extractor uses Meta's SAM ViT-B base weights (sam_vit_b_01ec64.pth) downloaded from dl.fbaipublicfiles.com.

**Why:** The SansuiHan/medical_models HuggingFace repo (which hosts medsam_vit_b.pth) uses the XET protocol which stalled on this network. The CAMMA UltraSam uses MMDetection format incompatible with segment_anything.

**How to apply:** After the main experiment completes, retry MedSAM by:
1. Download medsam_vit_b.pth via a different method (e.g., direct browser download or gdown from the official Google Drive: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN)
2. Place it at ~/.cache/ultrasam/medsam_vit_b.pth
3. Update extractors.py to use that filename
4. Run: python run_experiment.py --extractors ultrasam --skip-cache
