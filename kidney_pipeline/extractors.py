import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + 1e-8)


class BaseExtractor(ABC):
    name: str
    embed_dim: int

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        pass

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> np.ndarray:
        pass

    def extract_embedding(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        x = self._preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self._forward(x)
        emb = emb.flatten()
        return _l2_normalize(emb.reshape(1, -1)).flatten()

    def build_reference_bank(self, paths: List[Path], cache_path: Path) -> np.ndarray:
        if cache_path.exists():
            print(f"  Loading cached reference embeddings from {cache_path.name}")
            return np.load(cache_path)
        print(f"  Extracting {len(paths)} reference embeddings with {self.name}...")
        embeddings = []
        for p in tqdm(paths, desc=f"  {self.name} reference"):
            embeddings.append(self.extract_embedding(p))
        bank = np.stack(embeddings, axis=0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, bank)
        return bank

    def build_patient_embeddings(self, paths: List[Path], cache_path: Path) -> np.ndarray:
        if cache_path.exists():
            return np.load(cache_path)
        embeddings = []
        for p in paths:
            embeddings.append(self.extract_embedding(p))
        bank = np.stack(embeddings, axis=0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, bank)
        return bank


# ── ResNet50 ──────────────────────────────────────────────────────────────────

class ResNet50Extractor(BaseExtractor):
    name = "resnet50"
    embed_dim = 2048

    def _load_model(self):
        from torchvision import models
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        self._model = nn.Sequential(*list(model.children())[:-1]).eval().to(DEVICE)
        self._transforms = weights.transforms()

    def __init__(self):
        self._load_model()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._transforms(image)

    def _forward(self, x: torch.Tensor) -> np.ndarray:
        return self._model(x).squeeze(-1).squeeze(-1).cpu().numpy()


# ── EfficientNet-B0 ───────────────────────────────────────────────────────────

class EfficientNetB0Extractor(BaseExtractor):
    name = "efficientnet_b0"
    embed_dim = 1280

    def __init__(self):
        self._load_model()

    def _load_model(self):
        import timm
        self._model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).eval().to(DEVICE)
        cfg = timm.data.resolve_model_data_config(self._model)
        self._transforms = timm.data.create_transform(**cfg, is_training=False)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._transforms(image)

    def _forward(self, x: torch.Tensor) -> np.ndarray:
        return self._model(x).cpu().numpy()


# ── DenseNet121 ───────────────────────────────────────────────────────────────

class DenseNet121Extractor(BaseExtractor):
    name = "densenet121"
    embed_dim = 1024

    def __init__(self):
        self._load_model()

    def _load_model(self):
        from torchvision import models, transforms
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        self._features = model.features.eval().to(DEVICE)
        self._pool = nn.AdaptiveAvgPool2d((1, 1))
        self._transforms = weights.transforms()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._transforms(image)

    def _forward(self, x: torch.Tensor) -> np.ndarray:
        feat = self._features(x)
        feat = self._pool(feat)
        return feat.squeeze(-1).squeeze(-1).cpu().numpy()


# ── ConvNeXt-Tiny ─────────────────────────────────────────────────────────────

class ConvNeXtTinyExtractor(BaseExtractor):
    name = "convnext_tiny"
    embed_dim = 768

    def __init__(self):
        self._load_model()

    def _load_model(self):
        from torchvision import models
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        # Remove classifier (Sequential: 0=flatten, 1=linear)
        model.classifier = nn.Identity()
        self._model = model.eval().to(DEVICE)
        self._transforms = weights.transforms()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._transforms(image)

    def _forward(self, x: torch.Tensor) -> np.ndarray:
        return self._model(x).cpu().numpy()


# ── DINOv2 ViT-S/14 ──────────────────────────────────────────────────────────

class DINOv2ViTSExtractor(BaseExtractor):
    name = "dinov2_vits14"
    embed_dim = 384

    def __init__(self):
        self._load_model()

    def _load_model(self):
        from torchvision import transforms
        self._model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
        ).eval().to(DEVICE)
        self._transforms = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._transforms(image)

    def _forward(self, x: torch.Tensor) -> np.ndarray:
        return self._model(x).cpu().numpy()


# ── UltraSAM encoder ─────────────────────────────────────────────────────────

ULTRASAM_CACHE = Path.home() / ".cache" / "ultrasam"
# SAM ViT-B from Meta (official CDN) — base architecture used by all SAM medical variants.
# Will retry MedSAM fine-tuned weights once a direct download path is confirmed.
_SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
_SAM_VIT_B_FILENAME = "sam_vit_b_01ec64.pth"


def _download_ultrasam_weights() -> Path:
    ULTRASAM_CACHE.mkdir(parents=True, exist_ok=True)
    dest = ULTRASAM_CACHE / _SAM_VIT_B_FILENAME
    if dest.exists():
        print(f"  Using cached SAM ViT-B weights: {dest}")
        return dest
    print(f"  Downloading SAM ViT-B weights (~374 MB) from Meta CDN...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_done = downloaded // 1024 // 1024
            mb_total = total_size // 1024 // 1024
            print(f"\r  {pct}% ({mb_done} MB / {mb_total} MB)", end="", flush=True)

    urllib.request.urlretrieve(_SAM_VIT_B_URL, dest, _progress)
    print()
    return dest


class UltraSAMExtractor(BaseExtractor):
    name = "ultrasam"
    embed_dim = 256

    def __init__(self):
        self._load_model()

    def _load_model(self):
        from segment_anything import sam_model_registry
        checkpoint = _download_ultrasam_weights()
        sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        sam.eval().to(DEVICE)
        self._encoder = sam.image_encoder
        self._pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(DEVICE)
        self._pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(DEVICE)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms
        img = transforms.Resize((1024, 1024))(image)
        x = transforms.ToTensor()(img) * 255.0
        x = (x - self._pixel_mean.cpu()) / self._pixel_std.cpu()
        return x

    def _forward(self, x: torch.Tensor) -> np.ndarray:
        # Encoder output: (1, 256, 64, 64) — mean-pool spatial dims → (1, 256)
        feat = self._encoder(x)
        feat = feat.mean(dim=(-2, -1))
        return feat.cpu().numpy()


# ── Registry ──────────────────────────────────────────────────────────────────

EXTRACTOR_REGISTRY = {
    "resnet50": ResNet50Extractor,
    "efficientnet_b0": EfficientNetB0Extractor,
    "densenet121": DenseNet121Extractor,
    "convnext_tiny": ConvNeXtTinyExtractor,
    "dinov2_vits14": DINOv2ViTSExtractor,
    "ultrasam": UltraSAMExtractor,
}


def load_extractor(name: str) -> BaseExtractor:
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor: {name}. Choose from {list(EXTRACTOR_REGISTRY)}")
    print(f"\nLoading extractor: {name}")
    return EXTRACTOR_REGISTRY[name]()
