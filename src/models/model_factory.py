# src/models/model_factory.py

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------- Util path & state_dict ----------

def _resolve_checkpoint_path(path: Optional[str]) -> Optional[str]:
    """Kembalikan path absolut jika file ada; jika None atau tidak ada, kembalikan None."""
    if not path:
        return None
    abs_path = os.path.abspath(path)
    return abs_path if os.path.isfile(abs_path) else None


def _unwrap_state_dict(state):
    """
    Normalisasi berbagai format checkpoint:
    - {'state_dict': {...}}  -> {...}
    - {'model': {...}}       -> {...}
    - checkpoint dari wrapper InputResizeWrapper
      yang punya prefix 'model.' di semua key.
    """
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]

    # Tangani kasus wrapper: semua key diawali 'model.'
    if isinstance(state, dict):
        keys = list(state.keys())
        if keys and all(k.startswith("model.") for k in keys):
            stripped = {}
            for k, v in state.items():
                new_k = k[len("model."):]  # buang prefix 'model.'
                stripped[new_k] = v
            state = stripped

    return state



def _load_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor], logger: Optional[logging.Logger] = None) -> Tuple[list, list, list]:
    """
    Muat state_dict ke model dengan toleransi:
    - Drop kunci yang shape-nya tidak cocok (umumnya head).
    - strict=False agar missing/unexpected tidak memblokir.
    Return: (dropped_keys, missing_keys, unexpected_keys)
    """
    model_sd = model.state_dict()
    dropped = []
    state = dict(state_dict)  # salin supaya aman dimodifikasi

    # Drop hanya kunci yang shape-nya berbeda
    for k, v in list(state.items()):
        if k in model_sd and model_sd[k].shape != v.shape:
            dropped.append(k)
            state.pop(k)

    missing, unexpected = model.load_state_dict(state, strict=False)

    if logger:
        def _preview(keys: list, n: int = 12) -> str:
            if not keys:
                return ""
            head = keys[:n]
            suffix = " ..." if len(keys) > n else ""
            return ", ".join(head) + suffix

        logger.info(
            "Loaded checkpoint with strict=False (audit: allows partial load; "
            "commonly used when num_classes/head differs from checkpoint)."
        )
        logger.info(f"Load audit summary: dropped={len(dropped)} | missing={len(missing)} | unexpected={len(unexpected)}")
        if dropped:
            logger.info(
                f"Dropped {len(dropped)} key(s) due to shape mismatch (usually classifier head). "
                f"Example: {_preview(dropped)}"
            )
        if missing:
            logger.info(
                f"Missing keys after load (present in model, not in checkpoint): {len(missing)}. "
                f"Example: {_preview(list(missing))}"
            )
        if unexpected:
            logger.info(
                f"Unexpected keys after load (present in checkpoint, not in model): {len(unexpected)}. "
                f"Example: {_preview(list(unexpected))}"
            )

    return dropped, missing, unexpected


# ---------- Adaptor input khusus ViT ----------

class InputResizeWrapper(nn.Module):
    """
    Membungkus model agar input di-resize ke ukuran yang diharapkan backbone ViT.
    Tidak mengubah pipeline CoWePS lain: dataset/augmentasi/ConvNeXt dibiarkan apa adanya.
    """
    def __init__(self, model: nn.Module, target_size: int):
        super().__init__()
        self.model = model
        self.target_size = int(target_size)

    def forward(self, x):
        # x: [B, C, H, W]
        h, w = x.shape[-2], x.shape[-1]
        if h != self.target_size or w != self.target_size:
            # Bicubic cocok untuk ViT; align_corners=False aman
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bicubic', align_corners=False)
        return self.model(x)


class ModelFactory:
    """
    Pusat pembuatan model. Tujuan: sederhana, prediktabel, dan offline-friendly.
    """

    @staticmethod
    def _maybe_wrap_for_vit(model: nn.Module, model_name: str, logger: Optional[logging.Logger] = None) -> nn.Module:
        """
        Pasang adaptor input untuk ViT yang membutuhkan img_size tertentu.
        """
        name = (model_name or "").lower()

        # CLIP ViT varian 224 (timm: vit_base_patch16_clip_224.*)
        if "vit" in name and "clip" in name and "224" in name:
            if logger: logger.info("Applying InputResizeWrapper(target_size=224) for CLIP ViT.")
            return InputResizeWrapper(model, target_size=224)

        # DINOv2 ViT-B/14 (timm varian .lvd142m) — img_size 518
        if "vit_base_patch14_dinov2" in name or ("dinov2" in name and "patch14" in name):
            if logger: logger.info("Applying InputResizeWrapper(target_size=518) for DINOv2 ViT.")
            return InputResizeWrapper(model, target_size=518)

        # Model lain (ConvNeXt, ResNet, dst.) tidak perlu adaptor
        return model

    @staticmethod
    def create_model(
        model_name: str,
        num_classes: int,
        checkpoint_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> nn.Module:
        """
        Bangun model via timm dan muat bobot lokal jika tersedia.
        - Jika checkpoint_path ada → pretrained=False, load manual (strict=False).
        - Jika checkpoint_path None → pretrained=True (menggunakan hub).
        """
        ckpt = _resolve_checkpoint_path(checkpoint_path)

        if logger:
            logger.info("=" * 80)
            logger.info("MODEL FACTORY v2.4")
            logger.info("=" * 80)
            logger.info(f"Model           : {model_name}")
            logger.info(f"Num classes     : {num_classes}")
            logger.info(f"Local weights   : {ckpt if ckpt else 'None (using timm hub if available)'}")

        if ckpt:
            # OFFLINE PATH: bangun arsitektur tanpa bobot hub, lalu muat checkpoint lokal
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

            # Muat file
            state = torch.load(ckpt, map_location="cpu")
            state = _unwrap_state_dict(state)

            # Toleran terhadap mismatch head / key lain
            _load_flexible(model, state, logger=logger)

            if logger:
                logger.info(f"Local checkpoint loaded from: {ckpt}")
        else:
            # ONLINE/HUB PATH: tidak ada bobot lokal → pakai pretrained resmi
            # CATATAN: ini menyentuh internet jika belum tercache.
            # Jika checkpoint lokal tidak ada → hentikan dengan error yang jelas.
            raise RuntimeError(
                f"Local checkpoint not found for '{model_name}'. "
                f"Offline policy active: provide a valid local 'checkpoint_path' in config['model']['local_weights_path']."
            )

        # Pasang adaptor input hanya bila diperlukan (ViT tertentu)
        model = ModelFactory._maybe_wrap_for_vit(model, model_name, logger=logger)
        return model


def create_model_from_config(config: Dict, logger: Optional[logging.Logger] = None) -> nn.Module:
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    model_name = model_cfg.get("model_name", "convnext_base.fb_in22k")
    num_classes = int(model_cfg.get("num_classes", 5))
    checkpoint_path = model_cfg.get("local_weights_path", None)
    model_kwargs = model_cfg.get("model_kwargs", {})  # <--- NEW

    if logger:
        logger.info(f"[create_model_from_config] model_name={model_name}, num_classes={num_classes}, "
                    f"local_weights_path={checkpoint_path}, model_kwargs={model_kwargs}")

    ckpt = _resolve_checkpoint_path(checkpoint_path)
    if ckpt:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, **model_kwargs)
        state = _unwrap_state_dict(torch.load(ckpt, map_location="cpu"))
        _load_flexible(model, state, logger=logger)
    else:
        # OFFLINE-ONLY POLICY (China server): dilarang download.
        raise RuntimeError(
            f"Local checkpoint not found for '{model_name}'. "
            f"Offline policy active: set 'model.local_weights_path' to an existing .pth/.safetensors."
        )

    model = ModelFactory._maybe_wrap_for_vit(model, model_name, logger=logger)
    return model

# End of src/models/model_factory.py