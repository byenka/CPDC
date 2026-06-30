#!/usr/bin/env python3
"""CoDeC Phase 3 (target): Dual-head inference scoring + AE quality gate.

This module implements the *true* CoDeC dual-paradigm logic:
- Teacher-D (ConvNeXt supervised) and Teacher-R (DINOv2 representative) are scored separately.
- Disagreement is computed via JS divergence (optionally KL).
- Agreement is derived from JS and used for tiering.
- Autoencoder provides Q-score as physical gate.

Design goals
- Surgical: does not modify legacy modules.
- Reuses stable helpers from `src.inference.inference_scoring` where possible.
- Emits a single scores CSV + tier_A/B/C manifests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_processing import DRDataset

from src.models.model_factory import create_model_from_config

# Reuse proven loaders/calibration helpers from legacy v2.5 module.
from src.inference.inference_scoring import (  # noqa: E402
    UNet,
    _apply_temperature,
    _resolve_manifest_for_mode,
    find_q_threshold_roc,
    get_reconstruction_error,
)

from src.inference.codec_dualhead_metrics import (
    agreement_from_js,
    js_divergence,
    kl_divergence,
    probs_entropy_margin,
)
from src.inference.codec_dualhead_tiering import assign_tiers_dualhead
from src.inference.codec_export_tiers import export_tier_manifests_codec


def _load_base(base_config_path: str) -> Dict[str, Any]:
    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if "paths" not in cfg:
        raise ValueError(f"Missing required 'paths' in base config: {base_config_path}")
    return cfg


def _load_single_model_and_temperature(
    model_config_path: str,
    base_cfg: Dict[str, Any],
    device: torch.device,
    logger=None,
) -> Tuple[torch.nn.Module, float]:
    """Load one model + its temperature scaler for inference.

    This is intentionally separate from legacy `load_ensemble_a`, because CoDeC Phase 3
    teachers are *fixed weights* referenced by `model.local_weights_path` (not student
    checkpoints under output_dir/checkpoint_name).
    """

    log = (logger.info if logger else print)

    with open(model_config_path, "r") as f:
        mcfg = yaml.safe_load(f) or {}

    # Inject global base paths so any downstream helpers can resolve properly.
    mcfg["paths"] = base_cfg.get("paths", {})

    # Prefer a trained checkpoint when present (needed for OOF fold configs).
    # Fallback: use fixed weights via model.local_weights_path (teacher configs).
    model_cfg = (mcfg.get("model", {}) or {})
    out_dir = model_cfg.get("output_dir", None)
    ckpt_name = model_cfg.get("checkpoint_name", None)

    trained_ckpt = None
    if out_dir and ckpt_name:
        cand = os.path.join(str(out_dir), str(ckpt_name))
        if os.path.exists(cand):
            trained_ckpt = cand

    if trained_ckpt:
        mcfg_for_load = dict(mcfg)
        mcfg_for_load["model"] = dict(model_cfg)
        mcfg_for_load["model"]["local_weights_path"] = trained_ckpt
        log(f"   ✓ Using trained checkpoint for inference: {trained_ckpt}")
        model = create_model_from_config(mcfg_for_load, logger=logger)
    else:
        model = create_model_from_config(mcfg, logger=logger)
    model = model.to(device).eval()

    T_path = (mcfg.get("model", {}) or {}).get("calibration_path", None)
    if not T_path or not os.path.exists(T_path):
        raise FileNotFoundError(f"Temperature file not found: {T_path}")

    T_data = torch.load(T_path, map_location="cpu")
    T_optimal = T_data["temperature"] if isinstance(T_data, dict) else T_data
    T_optimal = float(T_optimal)
    log(f"   ✓ Temperature loaded: T = {T_optimal:.4f} ({T_path})")

    return model, T_optimal


def _resolve_output_path(base_cfg: Dict[str, Any], *, mode: str, out_csv: Optional[str]) -> str:
    if out_csv:
        out = str(out_csv)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        return out

    save_csv = base_cfg.get("inference", {}).get("save_scores_csv", None)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        return save_csv if str(save_csv).endswith(".csv") else os.path.join(save_csv, f"full_inference_results_{mode}.csv")

    scores_dir = base_cfg.get("paths", {}).get("scores_dir", "data/scores")
    os.makedirs(scores_dir, exist_ok=True)
    return os.path.join(scores_dir, f"full_inference_results_{mode}.csv")


def _find_oof_pcols(df: pd.DataFrame, *, source: str, suffix: str) -> Tuple[list[str], int]:
    mid_tag = f"_{source}" if source else ""
    want_tail = f"{mid_tag}{suffix}"
    cols: list[str] = []
    for c in df.columns:
        if not isinstance(c, str) or not c.startswith("p"):
            continue
        if not c.endswith(want_tail):
            continue
        mid = c[1 : -len(want_tail)]
        if mid.isdigit():
            cols.append(c)
    cols = sorted(cols, key=lambda x: int(x[1 : -len(want_tail)]))
    return cols, len(cols)


def _prepare_oof_lookup(oof_path: str) -> Tuple[Dict[str, Dict[str, Optional[np.ndarray]]], int]:
    oof_file = Path(oof_path)
    if not oof_file.exists():
        raise FileNotFoundError(f"OOF scores file not found: {oof_file}")

    df = pd.read_csv(oof_file)
    if "image_path" not in df.columns:
        raise ValueError(f"OOF scores file missing 'image_path' column: {oof_file}")

    p_D, k_D = _find_oof_pcols(df, source="D", suffix="")
    p_R, k_R = _find_oof_pcols(df, source="R", suffix="")
    if not p_D or not p_R:
        raise ValueError(
            f"OOF scores file missing required probability columns p*_D / p*_R: {oof_file}"
        )

    p_D_b, _ = _find_oof_pcols(df, source="D", suffix="_before")
    p_R_b, _ = _find_oof_pcols(df, source="R", suffix="_before")

    # Normalize and keep first occurrence per image_path for lookup speed.
    df = df.copy()
    df["image_path"] = df["image_path"].astype(str)
    df = df.drop_duplicates(subset=["image_path"], keep="first")
    lookup: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for _, row in df.iterrows():
        img = str(row["image_path"])
        entry: Dict[str, Optional[np.ndarray]] = {
            "D": pd.to_numeric(row[p_D], errors="coerce").to_numpy(dtype=np.float64),
            "R": pd.to_numeric(row[p_R], errors="coerce").to_numpy(dtype=np.float64),
            "D_before": pd.to_numeric(row[p_D_b], errors="coerce").to_numpy(dtype=np.float64) if p_D_b else None,
            "R_before": pd.to_numeric(row[p_R_b], errors="coerce").to_numpy(dtype=np.float64) if p_R_b else None,
        }
        lookup[img] = entry

    return lookup, min(k_D, k_R)


def run_full_inference_dualhead(
    *,
    base_config_path: str,
    convnext_config_path: str,
    dinov2_config_path: str,
    ae_config_path: Optional[str],
    mode: str = "master",
    out_csv: Optional[str] = None,
    compute_kl: bool = False,
    scores_oof_path: Optional[str] = None,
    use_oof_for_disagreement: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Run CoDeC Phase 3 with true dual-head scoring.

    Writes one scores CSV and tier manifests.

    Notes
    - For backward compatibility with downstream tooling, this also emits averaged
      probabilities in columns `p0..pK` (these correspond to p_avg after calibration).
        - When `scores_oof_path` is provided, label confidences are overridden by OOF
            probabilities (and optionally disagreement metrics) while retaining in-sample
            values in *_insample columns.
    """

    log = (logger.info if logger else print)

    base_cfg = _load_base(base_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # Load teachers (fixed weights + calibration).
    log("\n" + "=" * 80)
    log("LOADING CODEC TEACHERS (INFERENCE-ONLY)")
    log("=" * 80)
    log(f"1) Teacher-D config: {convnext_config_path}")
    teacher_D, T_D = _load_single_model_and_temperature(convnext_config_path, base_cfg, device, logger)
    log(f"2) Teacher-R config: {dinov2_config_path}")
    teacher_R, T_R = _load_single_model_and_temperature(dinov2_config_path, base_cfg, device, logger)

    # Load AE (optional)
    ae_enabled = False
    ae_model = None
    Q_threshold = None
    q_err_min = None
    q_err_max = None

    skip_q = bool(base_cfg.get("inference", {}).get("skip_qscore", False))
    if ae_config_path is not None and not skip_q:
        with open(ae_config_path, "r") as f:
            ae_cfg = yaml.safe_load(f) or {}
        ae_cfg["paths"] = base_cfg["paths"]

        ae_enabled = True
        ae_model = UNet(n_channels=3, n_classes=3, bilinear=True).to(device).eval()
        ae_checkpoint = os.path.join(ae_cfg["model"]["output_dir"], ae_cfg["model"]["checkpoint_name"])
        if not os.path.exists(ae_checkpoint):
            raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_checkpoint}")
        ae_model.load_state_dict(torch.load(ae_checkpoint, map_location=device))
        log(f"✓ Autoencoder loaded: {ae_checkpoint}")

        Q_threshold, q_err_min, q_err_max = find_q_threshold_roc(ae_model, base_cfg, device, logger)
    else:
        log("✓ AE disabled (skip_qscore=True or no ae_config_path)")

    # Prepare loader
    manifest_path = _resolve_manifest_for_mode(base_cfg, mode, logger)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found for mode '{mode}': {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    main_dataset = DRDataset(manifest_path, base_cfg, mode="val")
    bs = int(base_cfg.get("inference", {}).get("batch_size", 8))
    main_loader = DataLoader(main_dataset, batch_size=bs, shuffle=False, num_workers=4)

    log(f"✓ Mode: {mode}")
    log(f"✓ Manifest: {manifest_path}")
    log(f"✓ Main dataset loaded: {len(main_dataset)} images")
    log(f"✓ Batch size: {main_loader.batch_size}")

    # Disagreement alpha
    alpha = float((base_cfg.get("codec_dualhead", {}) or {}).get("disagreement", {}).get("alpha", 8.0) or 8.0)

    # Optional OOF lookup (used to override confidences/disagreement)
    oof_lookup: Optional[Dict[str, Dict[str, Optional[np.ndarray]]]] = None
    oof_num_classes: Optional[int] = None
    oof_class_mismatch_warned = False
    if scores_oof_path:
        try:
            oof_lookup, oof_num_classes = _prepare_oof_lookup(scores_oof_path)
            log(f"✓ OOF scores loaded: {scores_oof_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load OOF scores from {scores_oof_path}: {e}")

    results = []
    idx_global = 0

    teacher_D.eval()
    teacher_R.eval()

    with torch.no_grad():
        for batch in tqdm(main_loader, desc="CoDeC Dual-Head Inference"):
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    images, labels = batch
                elif len(batch) >= 3:
                    images, labels = batch[0], batch[1]
                else:
                    raise ValueError(f"Unexpected batch length from DataLoader: {len(batch)}")
            elif isinstance(batch, dict):
                images = batch.get("image") or batch.get("images")
                labels = batch.get("label") or batch.get("labels")
                if images is None or labels is None:
                    raise ValueError("Batch dict must contain 'image(s)' and 'label(s)' keys.")
            else:
                raise ValueError(f"Unexpected batch type from DataLoader: {type(batch)}")

            images = images.to(device)
            B = images.shape[0]

            # Teacher-D
            logits_D_raw = teacher_D(images)
            logits_D_scaled = _apply_temperature(logits_D_raw, T_D)
            probs_D_before = torch.softmax(logits_D_raw, dim=1)
            probs_D = torch.softmax(logits_D_scaled, dim=1)

            # Teacher-R
            logits_R_raw = teacher_R(images)
            logits_R_scaled = _apply_temperature(logits_R_raw, T_R)
            probs_R_before = torch.softmax(logits_R_raw, dim=1)
            probs_R = torch.softmax(logits_R_scaled, dim=1)

            # Average (for backward-compatible columns p0..pK)
            probs_avg_before = 0.5 * (probs_D_before + probs_R_before)
            probs_avg = 0.5 * (probs_D + probs_R)

            # Per-teacher metrics
            C_D_before, Pred_D_before = probs_D_before.max(dim=1)
            ent_D_before, mar_D_before = probs_entropy_margin(probs_D_before)
            C_D, Pred_D = probs_D.max(dim=1)
            ent_D, mar_D = probs_entropy_margin(probs_D)

            C_R_before, Pred_R_before = probs_R_before.max(dim=1)
            ent_R_before, mar_R_before = probs_entropy_margin(probs_R_before)
            C_R, Pred_R = probs_R.max(dim=1)
            ent_R, mar_R = probs_entropy_margin(probs_R)

            # Avg metrics
            C_avg_before, Pred_avg_before = probs_avg_before.max(dim=1)
            ent_avg_before, mar_avg_before = probs_entropy_margin(probs_avg_before)
            C_avg, Pred_avg = probs_avg.max(dim=1)
            ent_avg, mar_avg = probs_entropy_margin(probs_avg)

            # Disagreement metrics (after calibration)
            js = js_divergence(probs_D, probs_R)
            agreement = agreement_from_js(js, alpha=alpha)
            top1_agree = (Pred_D == Pred_R).to(torch.int32)

            if compute_kl:
                kl_DR = kl_divergence(probs_D, probs_R)
                kl_RD = kl_divergence(probs_R, probs_D)
            else:
                kl_DR = None
                kl_RD = None

            # AE quality gate
            if ae_enabled and ae_model is not None:
                raw_error = get_reconstruction_error(ae_model, images, device)
                denom = (float(q_err_max) - float(q_err_min)) if (q_err_max is not None and q_err_min is not None and float(q_err_max) > float(q_err_min)) else 1.0
                q_cont = (float(q_err_max) - raw_error) / denom if q_err_max is not None else (0.0 - raw_error)
                q_cont = np.clip(q_cont, 0.0, 1.0)
                Q_bin = (raw_error < float(Q_threshold)).astype(np.int32) if Q_threshold is not None else (q_cont >= 0.5).astype(np.int32)
            else:
                raw_error = np.full((B,), np.nan, dtype=np.float32)
                q_cont = np.full((B,), np.nan, dtype=np.float32)
                Q_bin = np.full((B,), 1, dtype=np.int32)

            num_classes = probs_avg.shape[1]

            if (
                oof_num_classes is not None
                and not oof_class_mismatch_warned
                and int(oof_num_classes) != int(num_classes)
            ):
                log(
                    f"[WARN] OOF probabilities report {oof_num_classes} classes, "
                    f"but model outputs {num_classes}. Using model outputs for scoring."
                )
                oof_class_mismatch_warned = True

            for i in range(B):
                if idx_global >= len(manifest_df):
                    break

                # image_path from manifest
                img_path = manifest_df.iloc[idx_global].get("image_path", None)
                if img_path is None:
                    for cand in ["filepath", "file_path", "path"]:
                        if cand in manifest_df.columns:
                            img_path = manifest_df.iloc[idx_global][cand]
                            break

                try:
                    label_val = int(labels[i].item())
                except Exception:
                    label_val = int(labels[i])

                # label confidences (in-sample)
                if 0 <= label_val < num_classes:
                    conf_R_D = float(probs_D[i, label_val].item())
                    conf_R_R = float(probs_R[i, label_val].item())
                    conf_R_D_before = float(probs_D_before[i, label_val].item())
                    conf_R_R_before = float(probs_R_before[i, label_val].item())
                else:
                    conf_R_D = float("nan")
                    conf_R_R = float("nan")
                    conf_R_D_before = float("nan")
                    conf_R_R_before = float("nan")

                label_conf_R = float(np.nanmin([conf_R_D, conf_R_R]))
                label_conf_R_before = float(np.nanmin([conf_R_D_before, conf_R_R_before]))
                label_conf_R_in = label_conf_R
                label_conf_R_before_in = label_conf_R_before
                conf_R_D_in = conf_R_D
                conf_R_R_in = conf_R_R
                conf_R_D_before_in = conf_R_D_before
                conf_R_R_before_in = conf_R_R_before

                # Defaults for OOF overrides
                conf_R_D_oof = float("nan")
                conf_R_R_oof = float("nan")
                label_conf_R_oof = float("nan")
                label_conf_R_before_oof = float("nan")
                conf_R_D_before_oof = float("nan")
                conf_R_R_before_oof = float("nan")
                js_val = float(js[i].item())
                agreement_val = float(agreement[i].item())
                js_in = js_val
                agreement_in = agreement_val
                js_oof_val = float("nan")
                agreement_oof_val = float("nan")

                if oof_lookup is not None:
                    entry = oof_lookup.get(str(img_path))
                    if entry is not None:
                        d_arr = entry.get("D")
                        r_arr = entry.get("R")
                        db_arr = entry.get("D_before")
                        rb_arr = entry.get("R_before")

                        if d_arr is not None and 0 <= label_val < len(d_arr):
                            conf_R_D_oof = float(d_arr[label_val])
                        if db_arr is not None and 0 <= label_val < len(db_arr):
                            conf_R_D_before_oof = float(db_arr[label_val])
                        if r_arr is not None and 0 <= label_val < len(r_arr):
                            conf_R_R_oof = float(r_arr[label_val])
                        if rb_arr is not None and 0 <= label_val < len(rb_arr):
                            conf_R_R_before_oof = float(rb_arr[label_val])
                        if db_arr is not None and 0 <= label_val < len(db_arr):
                            label_conf_R_before_oof = float(db_arr[label_val])
                        if rb_arr is not None and 0 <= label_val < len(rb_arr):
                            if np.isfinite(label_conf_R_before_oof):
                                label_conf_R_before_oof = float(
                                    np.nanmin([label_conf_R_before_oof, float(rb_arr[label_val])])
                                )
                            else:
                                label_conf_R_before_oof = float(rb_arr[label_val])

                        if np.isfinite(conf_R_D_oof) or np.isfinite(conf_R_R_oof):
                            label_conf_R_oof = float(np.nanmin([conf_R_D_oof, conf_R_R_oof]))

                        # Optional disagreement from OOF
                        if d_arr is not None and r_arr is not None:
                            try:
                                pD_oof = torch.tensor(d_arr, dtype=torch.float32).unsqueeze(0)
                                pR_oof = torch.tensor(r_arr, dtype=torch.float32).unsqueeze(0)
                                js_tmp = js_divergence(pD_oof, pR_oof)
                                ag_tmp = agreement_from_js(js_tmp, alpha=alpha)
                                js_oof_val = float(js_tmp.squeeze().item())
                                agreement_oof_val = float(ag_tmp.squeeze().item())
                            except Exception:
                                js_oof_val = float("nan")
                                agreement_oof_val = float("nan")

                        # Override confidences with OOF when available
                        if np.isfinite(conf_R_D_oof):
                            conf_R_D = conf_R_D_oof
                        if np.isfinite(conf_R_R_oof):
                            conf_R_R = conf_R_R_oof
                        if np.isfinite(label_conf_R_oof):
                            label_conf_R = label_conf_R_oof
                        if np.isfinite(label_conf_R_before_oof):
                            label_conf_R_before = label_conf_R_before_oof

                        # Optionally override disagreement with OOF
                        if use_oof_for_disagreement and np.isfinite(js_oof_val) and np.isfinite(agreement_oof_val):
                            js_in = js_val
                            agreement_in = agreement_val
                            js_val = js_oof_val
                            agreement_val = agreement_oof_val

                row = {
                    "image_path": img_path,
                    "label": label_val,

                    # Provenance
                    "teacher_D_config": str(convnext_config_path),
                    "teacher_R_config": str(dinov2_config_path),
                    "ae_config_path": str(ae_config_path) if ae_config_path else "",
                    "inference_mode": str(mode),

                    # AE
                    "Q_score": int(Q_bin[i]),
                    "Q_score_continuous": float(q_cont[i]),
                    "reconstruction_error": float(raw_error[i]),

                    # Disagreement
                        "js_div": float(js_val),
                        "agreement": float(agreement_val),
                        "js_div_in": float(js_in) if use_oof_for_disagreement else float("nan"),
                        "agreement_in": float(agreement_in) if use_oof_for_disagreement else float("nan"),
                        "js_div_oof": float(js_oof_val),
                        "agreement_oof": float(agreement_oof_val),
                    "top1_agree": int(top1_agree[i].item()),

                    # Avg (for compatibility)
                    "C_score_before": float(C_avg_before[i].item()),
                    "entropy_before": float(ent_avg_before[i].item()),
                    "margin_before": float(mar_avg_before[i].item()),
                    "Pred_Class_before": int(Pred_avg_before[i].item()),
                    "label_confidence_R_before": label_conf_R_before,
                    "label_confidence_R_before_oof": label_conf_R_before_oof,
                    "label_confidence_R_before_insample": label_conf_R_before_in,

                    "C_score": float(C_avg[i].item()),
                    "entropy": float(ent_avg[i].item()),
                    "margin": float(mar_avg[i].item()),
                    "Pred_Class": int(Pred_avg[i].item()),
                    "label_confidence_R": label_conf_R,
                    "label_confidence_R_insample": label_conf_R_in,
                    "label_confidence_R_oof": label_conf_R_oof,

                    # Teacher-D
                    "C_score_D_before": float(C_D_before[i].item()),
                    "entropy_D_before": float(ent_D_before[i].item()),
                    "margin_D_before": float(mar_D_before[i].item()),
                    "Pred_Class_D_before": int(Pred_D_before[i].item()),
                    "label_confidence_R_D_before": conf_R_D_before,
                    "label_confidence_R_D_before_oof": conf_R_D_before_oof,

                    "C_score_D": float(C_D[i].item()),
                    "entropy_D": float(ent_D[i].item()),
                    "margin_D": float(mar_D[i].item()),
                    "Pred_Class_D": int(Pred_D[i].item()),
                    "label_confidence_R_D": conf_R_D,
                    "label_confidence_R_D_insample": conf_R_D_in,
                    "label_confidence_R_D_oof": conf_R_D_oof,

                    # Teacher-R
                    "C_score_R_before": float(C_R_before[i].item()),
                    "entropy_R_before": float(ent_R_before[i].item()),
                    "margin_R_before": float(mar_R_before[i].item()),
                    "Pred_Class_R_before": int(Pred_R_before[i].item()),
                    "label_confidence_R_R_before": conf_R_R_before,
                    "label_confidence_R_R_before_oof": conf_R_R_before_oof,

                    "C_score_R": float(C_R[i].item()),
                    "entropy_R": float(ent_R[i].item()),
                    "margin_R": float(mar_R[i].item()),
                    "Pred_Class_R": int(Pred_R[i].item()),
                    "label_confidence_R_R": conf_R_R,
                    "label_confidence_R_R_insample": conf_R_R_in,
                    "label_confidence_R_R_oof": conf_R_R_oof,
                }

                if compute_kl and kl_DR is not None and kl_RD is not None:
                    row["kl_DR"] = float(kl_DR[i].item())
                    row["kl_RD"] = float(kl_RD[i].item())

                # Probabilities: avg in p0..pK for evaluator compatibility
                for c in range(num_classes):
                    row[f"p{c}"] = float(probs_avg[i, c].item())
                    row[f"p{c}_before"] = float(probs_avg_before[i, c].item())

                # Probabilities: per-teacher
                for c in range(num_classes):
                    row[f"p{c}_D"] = float(probs_D[i, c].item())
                    row[f"p{c}_D_before"] = float(probs_D_before[i, c].item())
                    row[f"p{c}_R"] = float(probs_R[i, c].item())
                    row[f"p{c}_R_before"] = float(probs_R_before[i, c].item())

                results.append(row)
                idx_global += 1

    results_df = pd.DataFrame(results)

    # Tiering (dual-head) + export tier manifests using legacy exporter
    results_df = assign_tiers_dualhead(results_df, base_cfg)

    # Ensure tiering outputs exist in base config (exporter reads base_cfg['tiering']['outputs']).
    base_cfg.setdefault("tiering", {})
    base_cfg["tiering"].setdefault("outputs", {})
    base_cfg["tiering"].setdefault("use", True)

    export_tier_manifests_codec(results_df, base_cfg, manifest_df)

    # Save
    output_path = _resolve_output_path(base_cfg, mode=mode, out_csv=out_csv)
    results_df.to_csv(output_path, index=False)

    log("=" * 80)
    log("CoDeC Phase 3 COMPLETE (dual-head)")
    log("=" * 80)
    log(f"✓ Results saved: {output_path}")
    log(f"✓ Total images processed: {len(results_df):,}")

    return results_df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CoDeC Phase 3: dual-head inference scoring")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--convnext", required=True)
    ap.add_argument("--dinov2", required=True)
    ap.add_argument("--ae", default=None)
    ap.add_argument("--mode", default="master", choices=["master", "train", "val", "gold_standard"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--compute-kl", action="store_true")
    ap.add_argument("--scores-oof", default=None, help="Optional OOF scores CSV to override label confidences")
    ap.add_argument(
        "--oof-use-disagreement",
        action="store_true",
        help="If set, replace agreement/js_div columns with OOF-derived values when available.",
    )

    args = ap.parse_args()

    run_full_inference_dualhead(
        base_config_path=args.base_config,
        convnext_config_path=args.convnext,
        dinov2_config_path=args.dinov2,
        ae_config_path=args.ae,
        mode=args.mode,
        out_csv=args.out,
        compute_kl=bool(args.compute_kl),
        scores_oof_path=args.scores_oof,
        use_oof_for_disagreement=bool(args.oof_use_disagreement),
    )
