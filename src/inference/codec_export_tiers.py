from __future__ import annotations

import os
from typing import Dict

import pandas as pd


def _resolve_tiering_outputs(base_config: Dict) -> Dict[str, str]:
    tcfg = base_config.get("tiering", {}) or {}
    outdir = tcfg.get("outputs", {}).get("dir") or base_config.get("paths", {}).get("processed_dir", "data/processed")
    tier_a = os.path.join(outdir, tcfg.get("outputs", {}).get("tier_a", "tier_A.csv"))
    tier_b = os.path.join(outdir, tcfg.get("outputs", {}).get("tier_b", "tier_B.csv"))
    tier_c = os.path.join(outdir, tcfg.get("outputs", {}).get("tier_c", "tier_C.csv"))
    os.makedirs(outdir, exist_ok=True)
    return {"A": tier_a, "B": tier_b, "C": tier_c}


def export_tier_manifests_codec(results_df: pd.DataFrame, base_config: Dict, manifest_df: pd.DataFrame) -> None:
    paths = _resolve_tiering_outputs(base_config)

    def _with_suffix(path: str, suffix: str = "_full") -> str:
        stem, ext = os.path.splitext(path)
        return f"{stem}{suffix}{ext or '.csv'}"

    # Merge meta columns from manifest
    meta_cols = [c for c in ["source", "device", "year", "grade"] if c in manifest_df.columns and c not in results_df.columns]
    cols_for_merge = ["image_path"] + meta_cols if "image_path" in manifest_df.columns else meta_cols

    merged = results_df.merge(manifest_df[cols_for_merge], on="image_path", how="left") if cols_for_merge else results_df

    # Export richer columns for audit (if present)
    export_cols = [c for c in [
        "image_path",
        "label",
        "tier",
        "Q_score",
        "Q_score_continuous",
        "agreement",
        "js_div",
        "label_confidence_R_D",
        "label_confidence_R_R",
        "label_confidence_R",
        "entropy",
        "margin",
        "C_score_D",
        "C_score_R",
        "C_score",
        "source",
        "device",
        "year",
        "grade",
    ] if c in merged.columns]

    for tier_name, path in paths.items():
        df_t = merged[merged["tier"] == tier_name]

        # Version 1: full info (all columns in merged, akin to master scores rows)
        full_path = _with_suffix(path, "_full")
        df_t.to_csv(full_path, index=False)

        # Version 2: slim audit view (legacy subset)
        slim_cols = export_cols if export_cols else list(df_t.columns)
        df_t[slim_cols].to_csv(path, index=False)

        print(f"Tier {tier_name}: {len(df_t):,} → {path} (slim), {full_path} (full)")
