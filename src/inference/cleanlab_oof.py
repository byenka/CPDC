#!/usr/bin/env python3
"""Cleanlab-only Out-of-Fold (OOF) utilities (single-head).

Goal
- Build fair OOF prediction probabilities for Cleanlab label-issue detection.
- Avoid coupling to CoDeC dual-head/AE modules.

Fairness / anti-leak discipline
- For each outer fold:
  - Train model on outer-train subset.
  - Calibrate/monitor on an *inner split of the outer-train* (not on the held-out fold).
  - Run inference on the held-out fold only after training is complete.

Notes
- Training uses src/models/train_model.py, which expects base_config_coweps2.yaml
  to be colocated with the model config file. We copy base config into each fold folder.
- Inference uses src/inference/inference_scoring.run_full_inference (single-head).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import train_model
from src.inference.inference_scoring import run_full_inference


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


def _rewrite_model_config_for_fold(
    cfg_path: str,
    *,
    base_cfg_path: str,
    outer_train_csv: Path,
    inner_val_csv: Path,
    fold_dir: Path,
    fold_idx: int,
    train_epochs: int,
) -> Path:
    cfg = Path(cfg_path)
    if not cfg.exists():
        raise FileNotFoundError(f"Model config not found: {cfg}")

    with cfg.open("r") as f:
        data = yaml.safe_load(f) or {}

    # Inject training manifests (both legacy and new keys)
    tr = data.get("training", {}) or {}
    tr["train_manifest"] = str(outer_train_csv)
    tr["val_manifest"] = str(inner_val_csv)
    tr["num_epochs"] = int(train_epochs)
    tr["epochs"] = int(train_epochs)
    tr["max_epochs"] = int(train_epochs)

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

    # Data aliases used by train_model.py fallback logic
    data_section = data.get("data", {}) or {}
    data_section["train_manifest_path"] = str(outer_train_csv)
    data_section["validation_manifest_path"] = str(inner_val_csv)
    data["data"] = data_section

    # Output paths per fold
    model_cfg = data.get("model", {}) or {}
    out_dir = fold_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cfg["output_dir"] = str(out_dir)
    model_cfg["checkpoint_name"] = f"convnext_cleanlab_oof_fold{fold_idx}.pth"
    model_cfg["calibration_path"] = str(out_dir / f"T_optimal_cleanlab_oof_fold{fold_idx}.pth")
    data["model"] = model_cfg

    # Logging dir per fold
    log_cfg = data.get("logging", {}) or {}
    log_cfg["log_dir"] = str(fold_dir / "logs")
    data["logging"] = log_cfg

    fold_dir.mkdir(parents=True, exist_ok=True)

    # train_model.train() requires base_config_coweps2.yaml or base_config.yaml to be colocated
    base_src = Path(base_cfg_path)
    if not base_src.exists():
        raise FileNotFoundError(f"Base config not found: {base_src}")
    (fold_dir / "base_config_coweps2.yaml").write_text(base_src.read_text(), encoding="utf-8")

    out_cfg_path = _with_suffix(cfg, f"_Cleanlab_oof_fold{fold_idx}")
    out_cfg_path = fold_dir / out_cfg_path.name
    with out_cfg_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return out_cfg_path


def _rewrite_inference_config_using_global_temperature(
    trained_cfg_path: Path,
    *,
    global_temperature_path: Optional[str],
    fold_dir: Path,
    fold_idx: int,
) -> Path:
    with trained_cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    if global_temperature_path:
        model_cfg = data.get("model", {}) or {}
        model_cfg["calibration_path"] = str(global_temperature_path)
        data["model"] = model_cfg

    out_cfg_path = fold_dir / f"inference_Cleanlab_oof_fold{fold_idx}.yaml"
    with out_cfg_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out_cfg_path


def _rewrite_base_config_for_holdout_inference(
    base_cfg_path: str,
    *,
    holdout_csv: Path,
    out_scores_csv: Path,
    fold_dir: Path,
    fold_idx: int,
    batch_size: Optional[int],
) -> Path:
    with Path(base_cfg_path).open("r") as f:
        cfg = yaml.safe_load(f) or {}

    manifests = cfg.get("manifests", {}) or {}
    manifests["validate_label"] = str(holdout_csv)
    manifests["validate"] = str(holdout_csv)
    manifests["master_list"] = str(holdout_csv)
    cfg["manifests"] = manifests

    inference = cfg.get("inference", {}) or {}
    inference["save_scores_csv"] = str(out_scores_csv)
    inference["skip_qscore"] = True
    if batch_size is not None:
        inference["batch_size"] = int(batch_size)
    cfg["inference"] = inference

    # Disable tiering to keep outputs lean & avoid any extra logic
    tiering = cfg.get("tiering", {}) or {}
    tiering["use"] = False
    cfg["tiering"] = tiering

    out_path = fold_dir / f"base_Cleanlab_oof_fold{fold_idx}.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def build_cleanlab_oof_scores_convnext(
    *,
    manifest_path: str,
    base_config_path: str,
    convnext_config_path: str,
    output_csv: str,
    k_folds: int = 5,
    stratify_cols: Optional[Iterable[str]] = None,
    work_dir: str = "/tmp/cleanlab_oof",
    seed: int = 42,
    train_epochs: int = 3,
    inner_val_fraction: float = 0.10,
    global_temperature_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Build stitched OOF scores CSV for ConvNeXt-only baseline."""

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

    fold_dfs: list[pd.DataFrame] = []

    for fold_idx, (tr_idx, ho_idx) in enumerate(skf.split(df, strat_labels)):
        fold_root = work / f"fold{fold_idx}"
        fold_root.mkdir(parents=True, exist_ok=True)

        out_scores = fold_root / "scores_holdout.csv"
        expected_rows = int(len(ho_idx))
        if bool(resume) and out_scores.exists():
            try:
                existing = pd.read_csv(out_scores)
                has_probs = any(str(c).startswith("p") for c in existing.columns)
                if len(existing) == expected_rows and has_probs:
                    print(f"\n[Cleanlab-OOF] Fold {fold_idx}: found existing scores ({len(existing)} rows), skipping")
                    existing["fold"] = fold_idx
                    fold_dfs.append(existing)
                    continue
            except Exception as e:
                print(f"\n[Cleanlab-OOF] Fold {fold_idx}: failed to read existing scores ({e}), rebuilding")

        outer_train_df = df.iloc[tr_idx].copy()
        holdout_df = df.iloc[ho_idx].copy()

        # Inner split (from outer-train only) for val/calibration so we do not leak into held-out.
        if not (0.0 < float(inner_val_fraction) < 0.5):
            raise ValueError("inner_val_fraction must be in (0, 0.5)")

        inner_labels = _make_stratify_labels(outer_train_df, strat_cols)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(inner_val_fraction), random_state=int(seed) + fold_idx)
        inner_tr_idx, inner_va_idx = next(sss.split(outer_train_df, inner_labels))

        inner_train_df = outer_train_df.iloc[inner_tr_idx].copy()
        inner_val_df = outer_train_df.iloc[inner_va_idx].copy()

        outer_train_csv = fold_root / "train_outer.csv"
        inner_val_csv = fold_root / "val_inner.csv"
        holdout_csv = fold_root / "holdout.csv"

        inner_train_df.to_csv(outer_train_csv, index=False)
        inner_val_df.to_csv(inner_val_csv, index=False)
        holdout_df.to_csv(holdout_csv, index=False)

        # 1) Train per fold (val = inner split of outer train)
        train_cfg = _rewrite_model_config_for_fold(
            convnext_config_path,
            base_cfg_path=base_config_path,
            outer_train_csv=outer_train_csv,
            inner_val_csv=inner_val_csv,
            fold_dir=fold_root,
            fold_idx=fold_idx,
            train_epochs=int(train_epochs),
        )

        print(f"\n[Cleanlab-OOF] Fold {fold_idx}: outer_train={len(inner_train_df)} inner_val={len(inner_val_df)} holdout={len(holdout_df)}")
        print(f"[Cleanlab-OOF] Training config: {train_cfg}")
        train_model.train(str(train_cfg))

        # 2) Inference on held-out fold with single-head inference_scoring
        infer_cfg = _rewrite_inference_config_using_global_temperature(
            train_cfg,
            global_temperature_path=global_temperature_path,
            fold_dir=fold_root,
            fold_idx=fold_idx,
        )

        base_fold_cfg = _rewrite_base_config_for_holdout_inference(
            base_config_path,
            holdout_csv=holdout_csv,
            out_scores_csv=out_scores,
            fold_dir=fold_root,
            fold_idx=fold_idx,
            batch_size=batch_size,
        )

        df_fold = run_full_inference(
            base_config_path=str(base_fold_cfg),
            ensemble_a_configs=[str(infer_cfg)],
            ae_config_path=None,
            mode="val",
        )
        df_fold["fold"] = fold_idx
        fold_dfs.append(df_fold)

    full_df = pd.concat(fold_dfs, ignore_index=True)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"\n✓ Cleanlab-OOF scores saved: {out_path} ({len(full_df)} rows)")
    return full_df


__all__ = ["build_cleanlab_oof_scores_convnext"]
