#!/usr/bin/env python3
"""CoDeC Phase 4 (target): Progressive sampling + CleanSet using dual-head agreement.

This keeps the Phase 4 shape (stage0/1/2 + optional CleanSet) but uses additional
signals emitted by dual-head Phase 3:
- agreement / js_div
- label_confidence_R_D and label_confidence_R_R (min over teachers)

Design goals
- Does not modify legacy `src/sampling/progressive_sampling.py`.
- Reuses its helpers where safe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.data.manifest_utils import MasterSpec, attach_master_metadata

# Reuse legacy helpers
from src.sampling.progressive_sampling import (  # noqa: E402
    _apply_c_clip,
    _load_base_config,
    _load_tier_df,
    _merge_train_with_tiers,
    _print_label_stats,
    _resolve_paths,
)


def run_progressive_sampling_dualhead(
    *,
    base_config_path: str,
    scores_file: Optional[str] = None,
    tier_a_path: Optional[str] = None,
    tier_b_path: Optional[str] = None,
    tier_c_path: Optional[str] = None,
    train_manifest_override: Optional[str] = None,
    prefix: str = "",
    export_eval: bool = False,
    eval_prob_source: str = "avg",
    export_train_full: bool = True,
) -> Dict[str, Any]:
    base_cfg = _load_base_config(base_config_path)
    paths = _resolve_paths(base_cfg)

    processed_dir: Path = paths["processed_dir"]
    final_dir: Path = paths["final_dir"]

    scores_p = Path(scores_file) if scores_file else paths["scores_file"]
    tier_a_p = Path(tier_a_path) if tier_a_path else paths["tier_a"]
    tier_b_p = Path(tier_b_path) if tier_b_path else paths["tier_b"]
    tier_c_p = Path(tier_c_path) if tier_c_path else paths["tier_c"]

    train_manifest_p = Path(train_manifest_override) if train_manifest_override else paths["train_manifest"]
    val_manifest_p = paths["val_manifest"]

    print("=" * 80)
    print("[CoDeC] Fase 4 - Progressive Sampling (Dual-Head)")
    print("=" * 80)
    print(f"Base config         : {base_config_path}")
    print(f"Train manifest      : {train_manifest_p}")
    print(f"Scores file         : {scores_p}")
    print(f"Tier A path         : {tier_a_p}")
    print(f"Tier B path         : {tier_b_p}")
    print(f"Tier C path         : {tier_c_p}")
    print("-" * 80)

    for p in [scores_p, tier_a_p, tier_b_p, tier_c_p, train_manifest_p]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    train_df = pd.read_csv(train_manifest_p)
    val_df = pd.read_csv(val_manifest_p) if val_manifest_p.exists() else None

    # Attach master metadata if possible
    try:
        master_spec = MasterSpec.from_base_config(base_cfg)
        train_df = attach_master_metadata(train_df, master_spec, context="codec_progressive_sampling_dualhead.train_manifest")
        if val_df is not None and "image_path" in val_df.columns:
            val_df = attach_master_metadata(val_df, master_spec, context="codec_progressive_sampling_dualhead.val_manifest")
    except Exception as e:
        print(f"[WARN] Gagal attach_master_metadata ke train/val manifest: {e}")

    print(f"✓ Train samples : {len(train_df)}")
    _print_label_stats(train_df, "Train (full)")

    tier_a_df = _load_tier_df(tier_a_p, "A")
    tier_b_df = _load_tier_df(tier_b_p, "B")
    tier_c_df = _load_tier_df(tier_c_p, "C")

    print(f"✓ Tier counts (ALL): A={len(tier_a_df)}, B={len(tier_b_df)}, C={len(tier_c_df)}")

    a_train, b_train, c_train = _merge_train_with_tiers(train_df, tier_a_df, tier_b_df, tier_c_df)
    print(f"✓ Tier counts (TRAIN): A={len(a_train)}, B={len(b_train)}, C={len(c_train)}")

    sampling_cfg = base_cfg.get("sampling", {}) or {}

    if eval_prob_source not in {"avg", "D", "R"}:
        raise ValueError(f"Invalid eval_prob_source: {eval_prob_source} (expected avg|D|R)")

    # Optional: allow stage0/1/2 labels to be overwritten from scores probabilities.
    # This is controlled in YAML under sampling.* so users can choose the label source
    # (original vs predicted) without changing trainer code.
    stage_label_source = str(sampling_cfg.get("stage_label_source", "original")).lower()
    stage_pred_prob_source = str(sampling_cfg.get("stage_predicted_prob_source", "avg")).strip()
    stage_keep_original = bool(sampling_cfg.get("stage_keep_original_label", True))
    if stage_label_source not in {"original", "predicted"}:
        raise ValueError(
            f"Invalid sampling.stage_label_source: {stage_label_source} (expected original|predicted)"
        )
    if stage_pred_prob_source not in {"avg", "D", "R"}:
        raise ValueError(
            f"Invalid sampling.stage_predicted_prob_source: {stage_pred_prob_source} (expected avg|D|R)"
        )

    c_train_filtered = _apply_c_clip(c_train, sampling_cfg, label_col="label")
    print(f"✓ Tier C after c_clip (TRAIN): {len(c_train_filtered)}")

    stage0_df = a_train.copy()
    stage0_df["stage"] = "stage0_A_only"

    stage1_df = pd.concat([a_train, b_train], ignore_index=True).drop_duplicates(subset=["image_path"])
    stage1_df["stage"] = "stage1_A_B"

    stage2_df = pd.concat([a_train, b_train, c_train_filtered], ignore_index=True).drop_duplicates(subset=["image_path"])
    stage2_df["stage"] = "stage2_A_B_Csubset"

    # Stage label correction (optional)
    if stage_label_source == "predicted":
        scores_df_stage = pd.read_csv(scores_p)
        if "image_path" not in scores_df_stage.columns:
            raise ValueError("sampling.stage_label_source=predicted but scores CSV has no 'image_path' column")

        def _find_pcols_stage(df: pd.DataFrame, *, prob_source: str) -> list[str]:
            mid_tag = "" if prob_source == "avg" else f"_{prob_source}"
            out: list[str] = []
            for c in df.columns:
                if not isinstance(c, str) or not c.startswith("p"):
                    continue
                if mid_tag and not c.endswith(mid_tag):
                    continue
                if not mid_tag and ("_" in c):
                    continue
                mid = c[1 : -len(mid_tag)] if mid_tag else c[1:]
                if mid.isdigit():
                    out.append(c)
            return sorted(out, key=lambda c: int(c[1 : -len(mid_tag)] if mid_tag else c[1:]))

        pcols_pred = _find_pcols_stage(scores_df_stage, prob_source=stage_pred_prob_source)
        if not pcols_pred:
            raise ValueError(
                f"sampling.stage_label_source=predicted but no prob columns found for stage_predicted_prob_source={stage_pred_prob_source}"
            )

        probs = scores_df_stage[pcols_pred].to_numpy(dtype="float64")
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums
        pred = probs.argmax(axis=1).astype(int)
        pred_df = pd.DataFrame({"image_path": scores_df_stage["image_path"].astype(str), "label_corrected": pred})

        def _apply_pred(df_stage: pd.DataFrame, name: str) -> pd.DataFrame:
            merged = df_stage.merge(pred_df, on="image_path", how="left")
            if stage_keep_original and "label" in merged.columns:
                merged["label_original"] = merged["label"]
            m = merged["label_corrected"].notna()
            if "label" in merged.columns:
                before = pd.to_numeric(merged.loc[m, "label"], errors="coerce")
                after = pd.to_numeric(merged.loc[m, "label_corrected"], errors="coerce")
                n_changed = int((before != after).sum()) if len(before) else 0
                merged.loc[m, "label"] = merged.loc[m, "label_corrected"].astype(int)
                print(
                    f"[Stages] stage_label_source=predicted applied to {name} (prob_source={stage_pred_prob_source}). "
                    f"changed_labels={n_changed} / {int(m.sum())}"
                )
            return merged

        stage0_df = _apply_pred(stage0_df, "Stage0")
        stage1_df = _apply_pred(stage1_df, "Stage1")
        stage2_df = _apply_pred(stage2_df, "Stage2")

    print("-" * 80)
    print(f"Stage0 (A only)      : {len(stage0_df)}")
    print(f"Stage1 (A+B)         : {len(stage1_df)}")
    print(f"Stage2 (A+B+Csubset) : {len(stage2_df)}")

    _print_label_stats(stage0_df, "Stage0 (A only)")
    _print_label_stats(stage1_df, "Stage1 (A+B)")
    _print_label_stats(stage2_df, "Stage2 (A+B+Csubset)")

    # CleanSet (dual-head)
    cleanset_cfg = (sampling_cfg.get("cleanset", {}) or {})
    enabled = bool(cleanset_cfg.get("enabled", False))

    cleanset_df = None
    scores_df_for_eval = None
    if enabled:
        print("-" * 80)
        print("[CleanSet] Dual-head mode aktif - membangun CleanSet dari Stage2 + scores")

        scores_df = pd.read_csv(scores_p)
        scores_df_for_eval = scores_df
        if "image_path" not in scores_df.columns:
            print("[CleanSet] WARNING: kolom 'image_path' tidak ada di scores_file, CleanSet dilewati.")
        else:
            # We only need a small set of columns for merge + filtering
            want = [
                "image_path",
                "tier",
                "Q_score_continuous",
                "Q_score",
                "agreement",
                "js_div",
                "label_confidence_R_D",
                "label_confidence_R_R",
                "label_confidence_R",
                "entropy",
                "margin",
            ]
            have = [c for c in want if c in scores_df.columns]
            scores_sub = scores_df[have].copy()

            merged = stage2_df.merge(scores_sub, on="image_path", how="left", suffixes=("", "_score"))

            label_col = cleanset_cfg.get("label_col", "label")
            min_R = float(cleanset_cfg.get("min_label_confidence", 0.90))
            min_q = cleanset_cfg.get("min_q_score", None)
            min_q = float(min_q) if min_q is not None else None

            # New (dual-head) thresholds
            min_agreement = cleanset_cfg.get("min_agreement", 0.85)
            min_agreement = float(min_agreement) if min_agreement is not None else None

            max_js = cleanset_cfg.get("max_js_div", None)
            max_js = float(max_js) if max_js is not None else None

            max_entropy = cleanset_cfg.get("max_entropy", None)
            max_entropy = float(max_entropy) if max_entropy is not None else None

            min_margin = cleanset_cfg.get("min_margin", None)
            min_margin = float(min_margin) if min_margin is not None else None

            clean = merged.copy()
            if "tier" in clean.columns:
                clean = clean[clean["tier"] == "A"]

            # Optional label correction (YAML-controlled)
            # Default behavior: do NOT change labels; only filter.
            label_source = str(cleanset_cfg.get("label_source", "original")).lower()
            pred_prob_source = str(cleanset_cfg.get("predicted_prob_source", "avg")).strip()
            keep_original = bool(cleanset_cfg.get("keep_original_label", True))
            if label_source not in {"original", "predicted"}:
                raise ValueError(f"Invalid sampling.cleanset.label_source: {label_source} (expected original|predicted)")
            if pred_prob_source not in {"avg", "D", "R"}:
                raise ValueError(
                    f"Invalid sampling.cleanset.predicted_prob_source: {pred_prob_source} (expected avg|D|R)"
                )
            if label_source == "predicted":
                # Derive prediction from scores probabilities (not from existing Pred_Class to avoid ambiguity).
                def _find_pcols(df: pd.DataFrame, *, prob_source: str) -> list[str]:
                    mid_tag = "" if prob_source == "avg" else f"_{prob_source}"
                    out: list[str] = []
                    for c in df.columns:
                        if not isinstance(c, str) or not c.startswith("p"):
                            continue
                        if mid_tag and not c.endswith(mid_tag):
                            continue
                        if not mid_tag and ("_" in c):
                            continue
                        mid = c[1 : -len(mid_tag)] if mid_tag else c[1:]
                        if mid.isdigit():
                            out.append(c)
                    return sorted(out, key=lambda c: int(c[1 : -len(mid_tag)] if mid_tag else c[1:]))

                pcols_pred = _find_pcols(scores_df, prob_source=pred_prob_source)
                if not pcols_pred:
                    raise ValueError(
                        f"label_source=predicted but no prob columns found in scores for predicted_prob_source={pred_prob_source}"
                    )
                probs = scores_df[pcols_pred].to_numpy(dtype="float64")
                row_sums = probs.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                probs = probs / row_sums
                pred = probs.argmax(axis=1).astype(int)
                pred_df = pd.DataFrame({"image_path": scores_df["image_path"].astype(str), "label_corrected": pred})
                clean = clean.merge(pred_df, on="image_path", how="left")

                # Apply correction: optionally overwrite `label` so trainers that read `label` will use corrected labels.
                if keep_original and "label" in clean.columns:
                    clean["label_original"] = clean["label"]
                if "label_corrected" in clean.columns:
                    # Only overwrite when prediction exists.
                    m = clean["label_corrected"].notna()
                    if "label" in clean.columns:
                        before = pd.to_numeric(clean.loc[m, "label"], errors="coerce")
                        after = pd.to_numeric(clean.loc[m, "label_corrected"], errors="coerce")
                        n_changed = int((before != after).sum()) if len(before) else 0
                        clean.loc[m, "label"] = clean.loc[m, "label_corrected"].astype(int)
                        print(
                            f"[CleanSet] label_source=predicted applied (prob_source={pred_prob_source}). "
                            f"changed_labels={n_changed} / {int(m.sum())}"
                        )

            # Build min label confidence across teachers when available
            if "label_confidence_R_D" in clean.columns and "label_confidence_R_R" in clean.columns:
                lcd = pd.to_numeric(clean["label_confidence_R_D"], errors="coerce")
                lcr = pd.to_numeric(clean["label_confidence_R_R"], errors="coerce")
                clean["min_label_confidence_dual"] = pd.concat([lcd, lcr], axis=1).min(axis=1)
                conf_col = "min_label_confidence_dual"
            else:
                conf_col = "label_confidence_R" if "label_confidence_R" in clean.columns else None

            if conf_col is not None and conf_col in clean.columns:
                clean = clean[pd.to_numeric(clean[conf_col], errors="coerce") >= float(min_R)]

            # Agreement gate
            if min_agreement is not None and "agreement" in clean.columns:
                clean = clean[pd.to_numeric(clean["agreement"], errors="coerce") >= float(min_agreement)]

            # Optional JS gate
            if max_js is not None and "js_div" in clean.columns:
                clean = clean[pd.to_numeric(clean["js_div"], errors="coerce") <= float(max_js)]

            # Optional entropy/margin gates (dual-head avg columns are kept for compatibility)
            if max_entropy is not None and "entropy" in clean.columns:
                clean = clean[pd.to_numeric(clean["entropy"], errors="coerce") <= float(max_entropy)]

            if min_margin is not None and "margin" in clean.columns:
                clean = clean[pd.to_numeric(clean["margin"], errors="coerce") >= float(min_margin)]

            # Quality gate
            if min_q is not None:
                if "Q_score_continuous" in clean.columns:
                    clean = clean[pd.to_numeric(clean["Q_score_continuous"], errors="coerce") >= float(min_q)]
                elif "Q_score" in clean.columns and float(min_q) > 0.5:
                    clean = clean[pd.to_numeric(clean["Q_score"], errors="coerce") == 1]

            cleanset_df = clean
            print(f"[CleanSet] Total sampel setelah filter: {len(cleanset_df)}")
            _print_label_stats(cleanset_df, "CleanSet (TierA + min_R + agreement + Q)", label_col=label_col)

    # If CleanSet wasn't built, we may still need scores for eval exports.
    if export_eval and scores_df_for_eval is None:
        scores_df_for_eval = pd.read_csv(scores_p)

    processed_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    base_cols = [c for c in train_df.columns]
    extra_cols = [c for c in stage2_df.columns if c not in base_cols]
    ordered_cols = base_cols + extra_cols

    out_stage0 = processed_dir / f"{prefix}student_stage0_train.csv"
    out_stage1 = processed_dir / f"{prefix}student_stage1_train.csv"
    out_stage2 = processed_dir / f"{prefix}student_stage2_train.csv"

    stage0_df[ordered_cols].to_csv(out_stage0, index=False)
    stage1_df[ordered_cols].to_csv(out_stage1, index=False)
    stage2_df[ordered_cols].to_csv(out_stage2, index=False)

    # Export full train manifests for evaluation (ECE / reliability): include probability columns.
    train_full_paths: Dict[str, str] = {}
    if export_train_full:
        # Reuse scores_df_for_eval if already loaded; otherwise load scores.
        if scores_df_for_eval is None:
            scores_df_for_eval = pd.read_csv(scores_p)

        if "image_path" not in scores_df_for_eval.columns:
            print("[WARN] export_train_full requested but scores CSV has no image_path; skipping train_full exports.")
        else:
            def _find_pcols_full(df: pd.DataFrame, *, prob_source: str, suffix: str = "") -> list[str]:
                mid_tag = "" if prob_source == "avg" else f"_{prob_source}"
                if suffix == "":
                    out: list[str] = []
                    for c in df.columns:
                        if not isinstance(c, str) or not c.startswith("p"):
                            continue
                        if mid_tag and not c.endswith(mid_tag):
                            continue
                        if not mid_tag and ("_" in c):
                            continue
                        mid = c[1 : -len(mid_tag)] if mid_tag else c[1:]
                        if mid.isdigit():
                            out.append(c)
                    return sorted(out, key=lambda c: int(c[1 : -len(mid_tag)] if mid_tag else c[1:]))
                want_tail = f"{mid_tag}{suffix}"
                out: list[str] = []
                for c in df.columns:
                    if not isinstance(c, str) or not c.startswith("p"):
                        continue
                    if not c.endswith(want_tail):
                        continue
                    mid = c[1 : -len(want_tail)]
                    if mid.isdigit():
                        out.append(c)
                return sorted(out, key=lambda c: int(c[1 : -len(want_tail)]))

            p_after = _find_pcols_full(scores_df_for_eval, prob_source=eval_prob_source, suffix="")
            p_before = _find_pcols_full(scores_df_for_eval, prob_source=eval_prob_source, suffix="_before")
            if not p_after:
                print(
                    f"[WARN] export_train_full: no probability columns found for eval_prob_source={eval_prob_source}; skipping."
                )
            else:
                # Keep all existing columns, just attach probabilities (and any codec columns not present).
                codec_cols = [
                    "tier",
                    "Q_score_continuous",
                    "Q_score",
                    "agreement",
                    "js_div",
                    "label_confidence_R_D",
                    "label_confidence_R_R",
                    "label_confidence_R",
                    "entropy",
                    "margin",
                    "min_label_confidence_dual",
                ]
                scores_keep = [
                    "image_path",
                    *[c for c in codec_cols if c in scores_df_for_eval.columns],
                    *[c for c in (p_after + p_before) if c in scores_df_for_eval.columns],
                ]
                scores_keep = list(dict.fromkeys(scores_keep))
                scores_sub = scores_df_for_eval[scores_keep].drop_duplicates(subset=["image_path"]).copy()

                def _export_full(df_stage: pd.DataFrame, out_path: Path) -> None:
                    merged = df_stage.merge(scores_sub, on="image_path", how="left", suffixes=("", "_score"))
                    merged.to_csv(out_path, index=False)

                out_stage0_full = processed_dir / f"{prefix}student_stage0_train_full.csv"
                out_stage1_full = processed_dir / f"{prefix}student_stage1_train_full.csv"
                out_stage2_full = processed_dir / f"{prefix}student_stage2_train_full.csv"
                _export_full(stage0_df, out_stage0_full)
                _export_full(stage1_df, out_stage1_full)
                _export_full(stage2_df, out_stage2_full)
                train_full_paths.update(
                    {
                        "stage0_train_full_path": str(out_stage0_full),
                        "stage1_train_full_path": str(out_stage1_full),
                        "stage2_train_full_path": str(out_stage2_full),
                    }
                )
                print(f"✓ Saved Stage0 train_full CSV : {out_stage0_full}")
                print(f"✓ Saved Stage1 train_full CSV : {out_stage1_full}")
                print(f"✓ Saved Stage2 train_full CSV : {out_stage2_full}")

    eval_paths: Dict[str, str] = {}
    if export_eval:
        if scores_df_for_eval is None or "image_path" not in scores_df_for_eval.columns:
            print("[WARN] export_eval requested but scores_df has no image_path; skipping eval exports.")
        else:
            def _find_pcols(df: pd.DataFrame, *, prob_source: str, suffix: str) -> list[str]:
                mid_tag = "" if prob_source == "avg" else f"_{prob_source}"
                pcols: list[str] = []
                if suffix == "":
                    for c in df.columns:
                        if not isinstance(c, str) or not c.startswith("p"):
                            continue
                        if mid_tag and not c.endswith(mid_tag):
                            continue
                        if not mid_tag and ("_" in c):
                            continue
                        mid = c[1 : -len(mid_tag)] if mid_tag else c[1:]
                        if mid.isdigit():
                            pcols.append(c)
                    return sorted(pcols, key=lambda c: int(c[1 : -len(mid_tag)] if mid_tag else c[1:]))

                want_tail = f"{mid_tag}{suffix}"
                for c in df.columns:
                    if not isinstance(c, str) or not c.startswith("p"):
                        continue
                    if not c.endswith(want_tail):
                        continue
                    mid = c[1 : -len(want_tail)]
                    if mid.isdigit():
                        pcols.append(c)
                return sorted(pcols, key=lambda c: int(c[1 : -len(want_tail)]))

            p_after = _find_pcols(scores_df_for_eval, prob_source=eval_prob_source, suffix="")
            p_before = _find_pcols(scores_df_for_eval, prob_source=eval_prob_source, suffix="_before")
            if not p_after:
                print(f"[WARN] export_eval: no probability columns found for eval_prob_source={eval_prob_source}; skipping.")
            else:
                # Keep it compact but evaluation-ready.
                codec_cols = [
                    "tier",
                    "Q_score_continuous",
                    "Q_score",
                    "agreement",
                    "js_div",
                    "label_confidence_R_D",
                    "label_confidence_R_R",
                    "label_confidence_R",
                    "entropy",
                    "margin",
                ]
                base_eval_cols = [c for c in ["image_path", "label", "stage"] if c in stage2_df.columns]
                stage_keep = [c for c in codec_cols if c in stage2_df.columns]

                scores_keep = ["image_path"] + [c for c in (p_after + p_before) if c in scores_df_for_eval.columns]
                scores_keep = list(dict.fromkeys(scores_keep))

                scores_sub = scores_df_for_eval[scores_keep].drop_duplicates(subset=["image_path"]).copy()

                def _export_eval(df_stage: pd.DataFrame, out_path: Path) -> None:
                    merged = df_stage.merge(scores_sub, on="image_path", how="left")
                    out_cols = base_eval_cols + stage_keep + [c for c in (p_after + p_before) if c in merged.columns]
                    out_cols = list(dict.fromkeys(out_cols))
                    merged[out_cols].to_csv(out_path, index=False)

                out_stage0_eval = processed_dir / f"{prefix}student_stage0_eval.csv"
                out_stage1_eval = processed_dir / f"{prefix}student_stage1_eval.csv"
                out_stage2_eval = processed_dir / f"{prefix}student_stage2_eval.csv"
                _export_eval(stage0_df, out_stage0_eval)
                _export_eval(stage1_df, out_stage1_eval)
                _export_eval(stage2_df, out_stage2_eval)
                eval_paths.update(
                    {
                        "stage0_eval_path": str(out_stage0_eval),
                        "stage1_eval_path": str(out_stage1_eval),
                        "stage2_eval_path": str(out_stage2_eval),
                    }
                )
                print(f"✓ Saved Stage0 eval CSV : {out_stage0_eval}")
                print(f"✓ Saved Stage1 eval CSV : {out_stage1_eval}")
                print(f"✓ Saved Stage2 eval CSV : {out_stage2_eval}")

    cleanset_out_path = None
    cleanset_eval_out_path = None
    if cleanset_df is not None:
        cleanset_filename = cleanset_cfg.get("output_name", "student_cleanset_train.csv")
        if prefix and not str(cleanset_filename).startswith(prefix):
            cleanset_filename = f"{prefix}{cleanset_filename}"
        cleanset_out_path = processed_dir / str(cleanset_filename)

        c_extra_cols = [c for c in cleanset_df.columns if c not in base_cols]
        c_ordered_cols = base_cols + c_extra_cols
        cleanset_df[c_ordered_cols].to_csv(cleanset_out_path, index=False)
        print(f"✓ Saved CleanSet manifest : {cleanset_out_path}")

        if export_eval and scores_df_for_eval is not None and "image_path" in scores_df_for_eval.columns:
            try:
                # Reuse already computed p_after/p_before if present in local scope
                # (if not, safely recompute).
                def _safe_find(df: pd.DataFrame, src: str, suffix: str) -> list[str]:
                    mid_tag = "" if src == "avg" else f"_{src}"
                    if suffix == "":
                        out = []
                        for c in df.columns:
                            if not isinstance(c, str) or not c.startswith("p"):
                                continue
                            if mid_tag and not c.endswith(mid_tag):
                                continue
                            if not mid_tag and ("_" in c):
                                continue
                            mid = c[1 : -len(mid_tag)] if mid_tag else c[1:]
                            if mid.isdigit():
                                out.append(c)
                        return sorted(out, key=lambda c: int(c[1 : -len(mid_tag)] if mid_tag else c[1:]))
                    want_tail = f"{mid_tag}{suffix}"
                    out = []
                    for c in df.columns:
                        if not isinstance(c, str) or not c.startswith("p"):
                            continue
                        if not c.endswith(want_tail):
                            continue
                        mid = c[1 : -len(want_tail)]
                        if mid.isdigit():
                            out.append(c)
                    return sorted(out, key=lambda c: int(c[1 : -len(want_tail)]))

                p_after2 = _safe_find(scores_df_for_eval, eval_prob_source, "")
                p_before2 = _safe_find(scores_df_for_eval, eval_prob_source, "_before")
                if p_after2:
                    scores_keep2 = ["image_path"] + [c for c in (p_after2 + p_before2) if c in scores_df_for_eval.columns]
                    scores_keep2 = list(dict.fromkeys(scores_keep2))
                    scores_sub2 = scores_df_for_eval[scores_keep2].drop_duplicates(subset=["image_path"]).copy()
                    merged_c = cleanset_df.merge(scores_sub2, on="image_path", how="left")

                    codec_cols2 = [
                        "tier",
                        "Q_score_continuous",
                        "Q_score",
                        "agreement",
                        "js_div",
                        "label_confidence_R_D",
                        "label_confidence_R_R",
                        "label_confidence_R",
                        "entropy",
                        "margin",
                        "min_label_confidence_dual",
                    ]
                    base_eval_cols2 = [c for c in ["image_path", "label", "stage"] if c in merged_c.columns]
                    keep2 = [c for c in codec_cols2 if c in merged_c.columns]
                    out_cols2 = base_eval_cols2 + keep2 + [c for c in (p_after2 + p_before2) if c in merged_c.columns]
                    out_cols2 = list(dict.fromkeys(out_cols2))

                    cleanset_eval_out_path = processed_dir / f"{prefix}student_cleanset_eval.csv"
                    merged_c[out_cols2].to_csv(cleanset_eval_out_path, index=False)
                    print(f"✓ Saved CleanSet eval CSV : {cleanset_eval_out_path}")
            except Exception as e:
                print(f"[WARN] Failed to export CleanSet eval CSV: {e}")

        # Optional: a full train manifest for CleanSet (same idea as stage*_train_full)
        if export_train_full and scores_df_for_eval is not None and "image_path" in scores_df_for_eval.columns:
            try:
                def _safe_find_full(df: pd.DataFrame, src: str, suffix: str) -> list[str]:
                    mid_tag = "" if src == "avg" else f"_{src}"
                    if suffix == "":
                        out = []
                        for c in df.columns:
                            if not isinstance(c, str) or not c.startswith("p"):
                                continue
                            if mid_tag and not c.endswith(mid_tag):
                                continue
                            if not mid_tag and ("_" in c):
                                continue
                            mid = c[1 : -len(mid_tag)] if mid_tag else c[1:]
                            if mid.isdigit():
                                out.append(c)
                        return sorted(out, key=lambda c: int(c[1 : -len(mid_tag)] if mid_tag else c[1:]))
                    want_tail = f"{mid_tag}{suffix}"
                    out = []
                    for c in df.columns:
                        if not isinstance(c, str) or not c.startswith("p"):
                            continue
                        if not c.endswith(want_tail):
                            continue
                        mid = c[1 : -len(want_tail)]
                        if mid.isdigit():
                            out.append(c)
                    return sorted(out, key=lambda c: int(c[1 : -len(want_tail)]))

                p_after_c = _safe_find_full(scores_df_for_eval, eval_prob_source, "")
                p_before_c = _safe_find_full(scores_df_for_eval, eval_prob_source, "_before")
                if p_after_c:
                    codec_cols_c = [
                        "tier",
                        "Q_score_continuous",
                        "Q_score",
                        "agreement",
                        "js_div",
                        "label_confidence_R_D",
                        "label_confidence_R_R",
                        "label_confidence_R",
                        "entropy",
                        "margin",
                        "min_label_confidence_dual",
                    ]
                    scores_keep_c = [
                        "image_path",
                        *[c for c in codec_cols_c if c in scores_df_for_eval.columns],
                        *[c for c in (p_after_c + p_before_c) if c in scores_df_for_eval.columns],
                    ]
                    scores_keep_c = list(dict.fromkeys(scores_keep_c))
                    scores_sub_c = scores_df_for_eval[scores_keep_c].drop_duplicates(subset=["image_path"]).copy()
                    merged_full = cleanset_df.merge(scores_sub_c, on="image_path", how="left", suffixes=("", "_score"))
                    cleanset_train_full_path = processed_dir / f"{prefix}student_cleanset_train_full.csv"
                    merged_full.to_csv(cleanset_train_full_path, index=False)
                    train_full_paths["cleanset_train_full_path"] = str(cleanset_train_full_path)
                    print(f"✓ Saved CleanSet train_full CSV : {cleanset_train_full_path}")
            except Exception as e:
                print(f"[WARN] Failed to export CleanSet train_full CSV: {e}")

    final_dataset = stage2_df[ordered_cols].copy()
    final_path = final_dir / f"{prefix}coweps_final_dataset.csv"
    final_dataset.to_csv(final_path, index=False)

    print("=" * 80)
    print(f"✓ Saved Stage0 manifest : {out_stage0}")
    print(f"✓ Saved Stage1 manifest : {out_stage1}")
    print(f"✓ Saved Stage2 manifest : {out_stage2}")
    print(f"✓ Final dataset (Stage2) : {final_path}")

    return {
        "success": True,
        "stage0_path": str(out_stage0),
        "stage1_path": str(out_stage1),
        "stage2_path": str(out_stage2),
        "final_path": str(final_path),
        "stage0_n": int(len(stage0_df)),
        "stage1_n": int(len(stage1_df)),
        "stage2_n": int(len(stage2_df)),
        "cleanset_path": str(cleanset_out_path) if cleanset_out_path else None,
        "cleanset_n": int(len(cleanset_df)) if cleanset_df is not None else 0,
        "export_eval": bool(export_eval),
        "eval_prob_source": str(eval_prob_source),
        "cleanset_eval_path": str(cleanset_eval_out_path) if cleanset_eval_out_path else None,
        **eval_paths,
        **train_full_paths,
    }
