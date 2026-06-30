#!/usr/bin/env python3
"""OOF discipline utilities for CoDeC dual-head teachers.

This module builds out-of-fold (OOF) probabilities by training teacher models
on K-1 folds and inferring on the held-out fold. Outputs are stitched into a
single CSV compatible with Phase 3/4 consumers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.codec_dualhead_inference_scoring import run_full_inference_dualhead
from src.models import train_model


def _make_stratify_labels(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = ["label"] if "label" in df.columns else []
    if not cols:
        raise ValueError("Manifest must contain at least a 'label' column for stratification")
    combined = df[cols].fillna("missing").astype(str)
    return combined.apply(lambda r: "|".join(r.values.tolist()), axis=1).to_numpy()


def _with_suffix(path: Path, suffix: str) -> Path:
    stem = path.stem
    ext = path.suffix or ".yaml"
    return path.with_name(f"{stem}{suffix}{ext}")


def _rewrite_model_config(
    cfg_path: str,
    *,
    base_cfg_path: str,
    train_manifest: Path,
    val_manifest: Path,
    fold_dir: Path,
    fold_idx: int,
    train_epochs: Optional[int] = None,
) -> Path:
    cfg = Path(cfg_path)
    if not cfg.exists():
        raise FileNotFoundError(f"Model config not found: {cfg}")

    with cfg.open("r") as f:
        data = yaml.safe_load(f) or {}

    # Training manifests (both legacy and new keys)
    tr = data.get("training", {}) or {}
    tr["train_manifest"] = str(train_manifest)
    tr["val_manifest"] = str(val_manifest)

    if train_epochs is not None:
        # Be tolerant to different config schemas.
        tr["epochs"] = int(train_epochs)
        tr["num_epochs"] = int(train_epochs)
        tr["max_epochs"] = int(train_epochs)

        # Keep common scheduler config in sync for shorter runs.
        if isinstance(tr.get("scheduler_params"), dict):
            sp = dict(tr.get("scheduler_params") or {})
            if "T_max" in sp:
                sp["T_max"] = int(train_epochs)
            tr["scheduler_params"] = sp

        if "warmup_epochs" in tr:
            try:
                tr["warmup_epochs"] = min(int(tr["warmup_epochs"]), int(train_epochs))
            except Exception:
                pass

    data["training"] = tr

    data_section = data.get("data", {}) or {}
    data_section["train_manifest_path"] = str(train_manifest)
    data_section["validation_manifest_path"] = str(val_manifest)
    data["data"] = data_section

    # Output / calibration paths per fold
    model_cfg = data.get("model", {}) or {}
    out_dir = Path(model_cfg.get("output_dir", cfg.parent / "oof_models"))
    out_dir = fold_dir / out_dir.name
    model_cfg["output_dir"] = str(out_dir)

    ckpt_name = model_cfg.get("checkpoint_name", f"{cfg.stem}.pth")
    ckpt_stem = Path(ckpt_name).stem
    model_cfg["checkpoint_name"] = f"{ckpt_stem}_fold{fold_idx}.pth"
    model_cfg["calibration_path"] = str(out_dir / f"T_optimal_fold{fold_idx}.pth")
    data["model"] = model_cfg

    # Logging dir (avoid collisions)
    log_cfg = data.get("logging", {}) or {}
    log_dir = log_cfg.get("log_dir", "outputs/logs")
    log_cfg["log_dir"] = str(fold_dir / Path(log_dir).name)
    data["logging"] = log_cfg

    fold_dir.mkdir(parents=True, exist_ok=True)

    # train_model.train() expects base_config_coweps2.yaml/base_config.yaml to be colocated
    # with the model config path (it searches the config folder). Keep a copy per fold.
    base_src = Path(base_cfg_path)
    if not base_src.exists():
        raise FileNotFoundError(f"Base config not found: {base_src}")
    (fold_dir / "base_config_coweps2.yaml").write_text(base_src.read_text(), encoding="utf-8")

    out_cfg_path = _with_suffix(cfg, f"_fold{fold_idx}")
    out_cfg_path = fold_dir / out_cfg_path.name
    with out_cfg_path.open("w") as f:
        yaml.safe_dump(data, f)
    return out_cfg_path


def _rewrite_base_config(
    base_cfg_path: str,
    *,
    manifest_path: Path,
    fold_dir: Path,
    out_scores: Path,
    tier_dir: Path,
    skip_qscore: bool,
    fold_idx: int,
) -> Path:
    with Path(base_cfg_path).open("r") as f:
        cfg = yaml.safe_load(f) or {}

    manifests = cfg.get("manifests", {}) or {}
    # IMPORTANT:
    # - legacy helper `_resolve_manifest_for_mode(mode='master')` ignores manifests.master_list
    #   and always maps to paths.processed_dir/master_list.csv.
    # - For OOF, we want to score ONLY the held-out fold CSV, so we configure the fold CSV
    #   as validate manifest(s) and run inference in mode='val'.
    manifests["validate_label"] = str(manifest_path)
    manifests["validate"] = str(manifest_path)  # backward-compat
    manifests["master_list"] = str(manifest_path)  # keep for tooling that reads it directly
    cfg["manifests"] = manifests

    inference = cfg.get("inference", {}) or {}
    inference["save_scores_csv"] = str(out_scores)
    inference["skip_qscore"] = bool(skip_qscore)
    cfg["inference"] = inference

    tiering = cfg.get("tiering", {}) or {}
    outputs = tiering.get("outputs", {}) or {}
    outputs_dir = tier_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    outputs["dir"] = str(outputs_dir)
    outputs.setdefault("tier_a", "tier_A.csv")
    outputs.setdefault("tier_b", "tier_B.csv")
    outputs.setdefault("tier_c", "tier_C.csv")
    tiering["outputs"] = outputs
    cfg["tiering"] = tiering

    out_path = fold_dir / f"base_oof_fold{fold_idx}.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f)
    return out_path


def build_oof_scores(
    *,
    manifest_path: str,
    base_config_path: str,
    convnext_config_path: str,
    dinov2_config_path: str,
    ae_config_path: Optional[str],
    k_folds: int = 5,
    stratify_cols: Optional[Iterable[str]] = None,
    work_dir: str = "/tmp/codec_oof",
    output_csv: str = "data/scores/codec_oof_scores_master.csv",
    seed: int = 42,
    train_epochs: Optional[int] = None,
    resume: bool = False,
) -> pd.DataFrame:
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")

    df = pd.read_csv(manifest_p)
    if "label" not in df.columns:
        raise ValueError(f"Manifest missing 'label' column: {manifest_p}")

    strat_cols = list(stratify_cols) if stratify_cols else ["label", "source"]
    strat_labels = _make_stratify_labels(df, strat_cols)

    skf = StratifiedKFold(n_splits=int(k_folds), shuffle=True, random_state=int(seed))

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    fold_dfs: List[pd.DataFrame] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(df, strat_labels)):
        fold_root = work / f"fold{fold_idx}"
        fold_root.mkdir(parents=True, exist_ok=True)

        out_scores = fold_root / "scores.csv"
        expected_rows = int(len(va_idx))
        if bool(resume) and out_scores.exists():
            try:
                existing = pd.read_csv(out_scores)
                has_probs = any(str(c).startswith("p") for c in existing.columns)
                if len(existing) == expected_rows and has_probs:
                    print(f"\n[OOF] Fold {fold_idx}: found existing scores.csv ({len(existing)} rows), skipping")
                    existing["fold"] = fold_idx
                    fold_dfs.append(existing)
                    continue
            except Exception as e:
                print(f"\n[OOF] Fold {fold_idx}: failed to read existing scores.csv ({e}), rebuilding")

        train_csv = fold_root / "train.csv"
        val_csv = fold_root / "val.csv"
        df.iloc[tr_idx].to_csv(train_csv, index=False)
        df.iloc[va_idx].to_csv(val_csv, index=False)

        conv_cfg = _rewrite_model_config(
            convnext_config_path,
            base_cfg_path=base_config_path,
            train_manifest=train_csv,
            val_manifest=val_csv,
            fold_dir=fold_root / "convnext",
            fold_idx=fold_idx,
            train_epochs=train_epochs,
        )
        dinov2_cfg = _rewrite_model_config(
            dinov2_config_path,
            base_cfg_path=base_config_path,
            train_manifest=train_csv,
            val_manifest=val_csv,
            fold_dir=fold_root / "dinov2",
            fold_idx=fold_idx,
            train_epochs=train_epochs,
        )

        base_fold_cfg = _rewrite_base_config(
            base_config_path,
            manifest_path=val_csv,
            fold_dir=fold_root,
            out_scores=out_scores,
            tier_dir=fold_root / "tiers",
            skip_qscore=ae_config_path is None,
            fold_idx=fold_idx,
        )

        print(f"\n[OOF] Fold {fold_idx}: train={len(tr_idx)} val={len(va_idx)}")
        print(f"[OOF] convnext cfg : {conv_cfg}")
        print(f"[OOF] dinov2 cfg  : {dinov2_cfg}")

        # Train teachers
        train_model.train(str(conv_cfg))
        train_model.train(str(dinov2_cfg))

        # Infer on holdout fold with newly trained teachers
        df_fold = run_full_inference_dualhead(
            base_config_path=str(base_fold_cfg),
            convnext_config_path=str(conv_cfg),
            dinov2_config_path=str(dinov2_cfg),
            ae_config_path=ae_config_path,
            mode="val",
            out_csv=str(out_scores),
            compute_kl=False,
        )
        df_fold["fold"] = fold_idx
        fold_dfs.append(df_fold)

    full_df = pd.concat(fold_dfs, ignore_index=True)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"\n✓ OOF scores saved: {out_path} ({len(full_df)} rows)")
    return full_df


__all__ = ["build_oof_scores"]
