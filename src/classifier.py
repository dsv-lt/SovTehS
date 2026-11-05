"""Нейросетевые классификаторы стиля (Стадия 1).

Модуль предоставляет единый интерфейс ``StyleClassifier`` и две реализации:
1. ``BaselineClassifier`` — PyTorch MLP поверх char TF-IDF (быстрый CPU/GPU инференс);
2. ``LoraClassifier`` — трансформер ``ruBert-base`` с дообученным LoRA-адаптером.
"""

from __future__ import annotations

import gc
import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import joblib
import numpy as np
import torch
from torch import nn

from src.config import Settings
from src.preprocessing import chunk_text

logger = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    """Полносвязная сеть для классификации стиля по вектору TF-IDF."""

    def __init__(self, in_features: int = 50000, num_classes: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Prediction:
    author: str
    confidence: float
    top_k: list[tuple[str, float]]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class StyleClassifier(ABC):
    """Общий интерфейс классификаторов."""

    name: str = "unknown"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.classes: list[str] = []
        self._lock = threading.Lock()

    @abstractmethod
    def _logits(self, chunks: list[str]) -> torch.Tensor:
        """Сырые логиты [n_chunks, n_classes]."""

    def predict(self, text: str, k: int = 3) -> Prediction:
        chunks = chunk_text(text, chunk_chars=self.settings.chunk_chars)
        if not chunks:
            raise ValueError("Пустой текст")
        chunks = chunks[: self.settings.max_chunks]
        with self._lock, torch.inference_mode():
            logits = self._logits(chunks).float()
            probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
        top = np.argsort(-probs)[: max(1, k)]
        top_k = [(self.classes[i], float(probs[i])) for i in top]
        return Prediction(author=top_k[0][0], confidence=top_k[0][1], top_k=top_k)

    def close(self) -> None:
        """Освобождает память модели (RAM/GPU)."""
        for attr in ("model", "vectorizer", "tokenizer"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class BaselineClassifier(StyleClassifier):
    """Классификатор на базе char TF-IDF + PyTorch MLP."""

    name = "pytorch-mlp-char-tfidf"

    def __init__(self, settings: Settings, vectorizer: object | None = None):
        super().__init__(settings)
        if vectorizer is None:
            vectorizer = joblib.load(settings.vectorizer_path)
        self.vectorizer = vectorizer
        ckpt = torch.load(settings.baseline_path, map_location=self.device, weights_only=True)
        self.classes = list(ckpt.get("classes", []))
        self.model = TextClassifier(
            in_features=ckpt.get("in_features", 50000),
            num_classes=ckpt.get("num_classes", len(self.classes)),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    def _logits(self, chunks: list[str]) -> torch.Tensor:
        mat = self.vectorizer.transform(chunks)
        tensor = torch.from_numpy(mat.toarray()).float().to(self.device)
        return self.model(tensor)


class LoraClassifier(StyleClassifier):
    """Классификатор на базе ruBert-base + LoRA адаптер."""

    name = "rubert-base-lora"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads((settings.lora_dir / "labels.json").read_text(encoding="utf-8"))
        self.classes = list(meta["classes"])
        self.max_length = int(meta.get("max_length", 256))
        base_path = (
            settings.base_model_dir if settings.base_model_dir.exists() else meta["base_model"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(settings.lora_dir)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_path,
            num_labels=len(self.classes),
            id2label=dict(enumerate(self.classes)),
            label2id={c: i for i, c in enumerate(self.classes)},
        )
        # Слияние LoRA-весов с базовыми: нет накладных расходов адаптера на инференсе
        self.model = PeftModel.from_pretrained(base, settings.lora_dir).merge_and_unload()
        self.model.to(self.device).eval()
        logger.info(
            "LoRA classifier loaded from %s (base: %s) on %s",
            settings.lora_dir,
            base_path,
            self.device,
        )

    def _logits(self, chunks: list[str]) -> torch.Tensor:
        batch_size = 8
        all_logits: list[torch.Tensor] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(**enc).logits
            else:
                out = self.model(**enc).logits
            all_logits.append(out)
        return torch.cat(all_logits, dim=0) if len(all_logits) > 1 else all_logits[0]


def load_classifier(settings: Settings, vectorizer: object | None = None) -> StyleClassifier:
    if (
        settings.classifier_backend == "lora"
        and (settings.lora_dir / "adapter_config.json").exists()
    ):
        return LoraClassifier(settings)
    if settings.classifier_backend == "lora":
        logger.warning(
            "LoRA adapter not found at %s, falling back to PyTorch baseline", settings.lora_dir
        )
    return BaselineClassifier(settings, vectorizer=vectorizer)
