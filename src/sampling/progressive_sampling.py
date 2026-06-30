#!/usr/bin/env python
"""
CoWePS v2.5 - Progressive Sampling (Fase 4)

Tujuan:
- Menggunakan hasil inferensi (Fase 3) + tiering (A/B/C)
- Menghasilkan manifest untuk training STUDENT:
    * student_stage0_train.csv : hanya Tier A
    * student_stage1_train.csv : Tier A + B
    * student_stage2_train.csv : Tier A + B + subset C (opsional, tergantung config)
- Menghasilkan satu dataset final:
    * coweps_final_dataset.csv (pakai Stage 2)

Input utama:
- base_config_coweps.yaml
- full_inference_results.csv  (path di base_config['inference']['save_scores_csv'])
- tier_A.csv, tier_B.csv, tier_C.csv (path di base_config['tiering']['outputs'])

Output:
- data/processed/student_stage0_train.csv
- data/processed/student_stage1_train.csv
- data/processed/student_stage2_train.csv
- data/final/coweps_final_dataset.csv

Catatan:
- Skrip ini TIDAK mengubah label; hanya memilih subset sample dari
  gold_standard_train.csv berdasarkan Tier (A/B/C) dan config sampling.
- Distribusi label per-stage dicetak sebagai diagnostik (tidak memengaruhi isi CSV).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import pandas as pd

from src.data.manifest_utils import MasterSpec, attach_master_metadata


# -----------------------------------------------------------------------------
# Konfigurasi & path
# -----------------------------------------------------------------------------
def _load_base_config(base_config_path: str) -> Dict[str, Any]:
    base_p = Path(base_config_path)
    if not base_p.exists():
        raise FileNotFoundError(f"Base config tidak ditemukan: {base_config_path}")
    with base_p.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _resolve_paths(base_cfg: Dict[str, Any]) -> Dict[str, Path]:
    paths = base_cfg.get("paths", {}) or {}
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    final_dir = Path(paths.get("final_dir", "data/final"))
    scores_file = Path(
        base_cfg.get("inference", {}).get(
            "save_scores_csv",
            str(Path("data/scores/full_inference_results.csv")),
        )
    )

    tiering = base_cfg.get("tiering", {}) or {}
    tout = tiering.get("outputs", {}) or {}
    tier_dir = Path(tout.get("dir", str(processed_dir)))
    tier_a = tier_dir / tout.get("tier_a", "tier_A.csv")
    tier_b = tier_dir / tout.get("tier_b", "tier_B.csv")
    tier_c = tier_dir / tout.get("tier_c", "tier_C.csv")

    manifests = base_cfg.get("manifests", {}) or {}
    train_manifest = Path(
        manifests.get("train", str(processed_dir / "gold_standard_train.csv"))
    )
    val_manifest = Path(
        manifests.get("validate", str(processed_dir / "gold_standard_validate.csv"))
    )

    return {
        "processed_dir": processed_dir,
        "final_dir": final_dir,
        "scores_file": scores_file,
        "tier_a": tier_a,
        "tier_b": tier_b,
        "tier_c": tier_c,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
    }


# -----------------------------------------------------------------------------
# Load tier & merge dengan train manifest
# -----------------------------------------------------------------------------
def _load_tier_df(path: Path, tier_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tier {tier_name} tidak ditemukan: {path}")
    df = pd.read_csv(path)
    if "image_path" not in df.columns:
        raise ValueError(f"File {path} tidak mengandung kolom 'image_path'.")
    df = df.copy()
    df["tier"] = tier_name
    return df


def _merge_train_with_tiers(
    train_df: pd.DataFrame,
    tier_a: pd.DataFrame,
    tier_b: pd.DataFrame,
    tier_c: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Join train manifest dengan tier A/B/C berdasar 'image_path'.

    Hanya sample yang ada di train_df yang dipakai untuk training STUDENT.
    """
    key = "image_path"
    keep_cols = [c for c in tier_a.columns if c != "tier"]

    # Pastikan kolom 'image_path' ada di train_df
    if key not in train_df.columns:
        raise ValueError("Train manifest harus memiliki kolom 'image_path'.")

    a_train = train_df.merge(
        tier_a[keep_cols + ["tier"]], on=key, how="inner", suffixes=("", "_tier")
    )
    b_train = train_df.merge(
        tier_b[keep_cols + ["tier"]], on=key, how="inner", suffixes=("", "_tier")
    )
    c_train = train_df.merge(
        tier_c[keep_cols + ["tier"]], on=key, how="inner", suffixes=("", "_tier")
    )

    return a_train, b_train, c_train


# -----------------------------------------------------------------------------
# Filter Tier C (opsional) + diagnostik label
# -----------------------------------------------------------------------------
def _apply_c_clip(
    c_df: pd.DataFrame, sampling_cfg: Dict[str, Any], label_col: str = "label"
) -> pd.DataFrame:
    """
    Terapkan filter opsional untuk Tier C berdasarkan config:
      sampling.stages[2].c_clip.entropy_max
      sampling.stages[2].c_clip.min_per_class
    Jika tidak ada config, return c_df apa adanya.
    """
    stages = sampling_cfg.get("stages", [])
    if len(stages) < 3:
        return c_df

    stage2 = stages[2]
    c_clip = stage2.get("c_clip", {}) or {}
    entropy_max = c_clip.get("entropy_max", None)
    min_per_class = int(c_clip.get("min_per_class", 0) or 0)

    result = c_df.copy()

    if entropy_max is not None and "entropy" in result.columns:
        result = result[result["entropy"] <= float(entropy_max)]

    # Jika ingin memastikan minimum per kelas, lakukan per-label sampling
    if min_per_class > 0 and label_col in result.columns and len(result) > 0:
        parts = []
        for lbl, grp in result.groupby(label_col):
            if len(grp) <= min_per_class:
                parts.append(grp)
            else:
                parts.append(grp.sample(n=min_per_class, random_state=42))
        if parts:
            result = pd.concat(parts, ignore_index=True)

    return result


def _print_label_stats(df: pd.DataFrame, name: str, label_col: str = "label") -> None:
    """
    Cetak distribusi label (diagnostik saja, tidak memengaruhi isi CSV).
    """
    if df is None:
        print(f"[STATS] {name}: df is None, skip.")
        return
    if label_col not in df.columns:
        print(f"[STATS] {name}: kolom '{label_col}' tidak ada, skip distribusi label.")
        return

    total = len(df)
    if total == 0:
        print(f"[STATS] {name}: kosong (n=0).")
        return

    vc = df[label_col].value_counts().sort_index()
    print(f"[STATS] {name} - distribusi label (n={total}):")
    for lbl, cnt in vc.items():
        pct = 100.0 * cnt / total
        print(f"   label={lbl}: {cnt} ({pct:.2f}%)")
    print()


# -----------------------------------------------------------------------------
# Pipeline utama
# -----------------------------------------------------------------------------
def run_progressive_sampling(base_config_path: str, prefix: str = "") -> Dict[str, Any]:
    base_cfg = _load_base_config(base_config_path)
    paths = _resolve_paths(base_cfg)

    processed_dir = paths["processed_dir"]
    final_dir = paths["final_dir"]
    scores_file = paths["scores_file"]
    tier_a_path = paths["tier_a"]
    tier_b_path = paths["tier_b"]
    tier_c_path = paths["tier_c"]
    train_manifest_path = paths["train_manifest"]
    val_manifest_path = paths["val_manifest"]

    print("=" * 80)
    print("[CoWePS v2.5] Fase 4 - Progressive Sampling (Student Dataset)")
    print("=" * 80)
    print(f"Base config         : {base_config_path}")
    print(f"Train manifest      : {train_manifest_path}")
    print(f"Val manifest        : {val_manifest_path}")
    print(f"Scores file         : {scores_file}")
    print(f"Tier A path         : {tier_a_path}")
    print(f"Tier B path         : {tier_b_path}")
    print(f"Tier C path         : {tier_c_path}")
    print("-" * 80)

    # Check existence (pastikan Fase 3 sudah selesai)
    for p in [scores_file, tier_a_path, tier_b_path, tier_c_path, train_manifest_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Load manifests
    train_df = pd.read_csv(train_manifest_path)
    val_df = pd.read_csv(val_manifest_path) if val_manifest_path.exists() else None

    # Pastikan train_df sudah membawa metadata standar dari master_list
    # (image_path, label, source, mask_path, device, year, id_patient)
    try:
        master_spec = MasterSpec.from_base_config(base_cfg)
        train_df = attach_master_metadata(
            train_df,
            master_spec,
            context="progressive_sampling.train_manifest",
        )
        if val_df is not None and "image_path" in val_df.columns:
            # Val tidak selalu dipakai untuk training student, tetapi
            # aman untuk menyeragamkan metadata jika ingin dianalisis.
            val_df = attach_master_metadata(
                val_df,
                master_spec,
                context="progressive_sampling.val_manifest",
            )
    except Exception as e:
        # Jangan matikan pipeline jika master_list belum lengkap; cukup log.
        print(f"[WARN] Gagal attach_master_metadata ke train/val manifest: {e}")

    print(f"✓ Train samples (original gold standard) : {len(train_df)}")
    if val_df is not None:
        print(f"✓ Val samples (gold standard)           : {len(val_df)}")

    _print_label_stats(train_df, "Gold Standard Train (full)")

    # Load tiers
    tier_a_df = _load_tier_df(tier_a_path, "A")
    tier_b_df = _load_tier_df(tier_b_path, "B")
    tier_c_df = _load_tier_df(tier_c_path, "C")

    print(
        f"✓ Tier counts (ALL splits): "
        f"A={len(tier_a_df)}, B={len(tier_b_df)}, C={len(tier_c_df)}"
    )

    # Restrict to train set
    a_train, b_train, c_train = _merge_train_with_tiers(
        train_df, tier_a_df, tier_b_df, tier_c_df
    )
    print(
        f"✓ Tier counts (TRAIN only): "
        f"A={len(a_train)}, B={len(b_train)}, C={len(c_train)}"
    )

    # Ambil config sampling (opsional)
    sampling_cfg = base_cfg.get("sampling", {}) or {}

    # Apply C-clip (opsional)
    c_train_filtered = _apply_c_clip(c_train, sampling_cfg, label_col="label")
    print(f"✓ Tier C after c_clip (TRAIN): {len(c_train_filtered)}")

    # Stage 0: Tier A saja
    stage0_df = a_train.copy()
    stage0_df["stage"] = "stage0_A_only"

    # Stage 1: Tier A + B (union)
    stage1_df = pd.concat([a_train, b_train], ignore_index=True)
    stage1_df = stage1_df.drop_duplicates(subset=["image_path"])
    stage1_df["stage"] = "stage1_A_B"

    # Stage 2: Tier A + B + subset C
    stage2_df = pd.concat([a_train, b_train, c_train_filtered], ignore_index=True)
    stage2_df = stage2_df.drop_duplicates(subset=["image_path"])
    stage2_df["stage"] = "stage2_A_B_Csubset"

    print("-" * 80)
    print(f"Stage0 (A only)              : {len(stage0_df)}")
    print(f"Stage1 (A+B)                 : {len(stage1_df)}")
    print(f"Stage2 (A+B+Csubset)         : {len(stage2_df)}")

    # Distribusi label per stage (diagnostik)
    # Distribusi label per stage (diagnostik)
    _print_label_stats(stage0_df, "Stage0 (A only)")
    _print_label_stats(stage1_df, "Stage1 (A+B)")
    _print_label_stats(stage2_df, "Stage2 (A+B+Csubset)")

    # -------------------------------------------------------------------------
    # CleanSet (opsional) - subset lebih ketat dari Stage2 berbasis skor inferensi
    # -------------------------------------------------------------------------
    cleanset_cfg = (sampling_cfg.get("cleanset", {}) or {})
    cleanset_enabled = bool(cleanset_cfg.get("enabled", False))
    cleanset_df = None
    cleanset_out_path = None

    if cleanset_enabled:
        print("-" * 80)
        print("[CleanSet] Mode aktif - membangun CleanSet dari Stage2 + scores_file")
        # Pastikan file skor tersedia
        if not scores_file.exists():
            print(f"[CleanSet] WARNING: scores_file tidak ditemukan: {scores_file}")
        else:
            scores_df = pd.read_csv(scores_file)
            if "image_path" not in scores_df.columns:
                print(
                    "[CleanSet] WARNING: kolom 'image_path' tidak ada di scores_file, CleanSet dilewati."
                )
            else:
                # Pilih kolom yang relevan untuk join
                merge_cols = ["image_path"]
                for col in [
                    "label_confidence_R",
                    "entropy",
                    "margin",
                    "Q_score",
                    "Q_score_continuous",
                ]:
                    if col in scores_df.columns and col not in merge_cols:
                        merge_cols.append(col)
                scores_sub = scores_df[merge_cols].copy()

                # Merge skor ke Stage2 (A+B+Csubset) via image_path
                stage2_scores = stage2_df.merge(
                    scores_sub, on="image_path", how="left", suffixes=("", "_score")
                )

                # Ambil threshold dari config
                min_R = cleanset_cfg.get("min_label_confidence", 0.9)
                max_entropy = cleanset_cfg.get("max_entropy", None)
                min_margin = cleanset_cfg.get("min_margin", None)
                min_q_score = cleanset_cfg.get("min_q_score", None)  # <--- NEW: Ambil threshold Q-score
                min_per_class_clean = cleanset_cfg.get("min_per_class", 0)
                label_col = cleanset_cfg.get("label_col", "label")

                # Mulai dari Stage2 lalu batasi ke Tier A saja (paling tepercaya)
                clean = stage2_scores.copy()
                if "tier" in clean.columns:
                    clean = clean[clean["tier"] == "A"]

                # Filter berdasarkan R (label_confidence_R)
                if "label_confidence_R" in clean.columns and min_R is not None:
                    clean = clean[clean["label_confidence_R"] >= float(min_R)]

                # Filter entropi (opsional)
                if max_entropy is not None and "entropy" in clean.columns:
                    clean = clean[clean["entropy"] <= float(max_entropy)]

                # Filter margin (opsional)
                if min_margin is not None and "margin" in clean.columns:
                    clean = clean[clean["margin"] >= float(min_margin)]

                # Filter Quality / Q-score (FIX: Mengaktifkan Quality Gate)
                if min_q_score is not None:
                    # Prioritaskan continuous score (0.0 - 1.0)
                    if "Q_score_continuous" in clean.columns:
                        clean = clean[clean["Q_score_continuous"] >= float(min_q_score)]
                        print(f"[CleanSet] Filtered by Q_score_continuous >= {min_q_score}")
                    # Fallback ke binary score jika continuous tidak ada (jarang terjadi)
                    elif "Q_score" in clean.columns:
                        if float(min_q_score) > 0.5: # Asumsi user ingin 'Good'
                            clean = clean[clean["Q_score"] == 1]
                            print(f"[CleanSet] Filtered by Q_score == 1 (Binary)")

                # Minimum per kelas (opsional)
                if (
                    min_per_class_clean > 0
                    and label_col in clean.columns
                    and len(clean) > 0
                ):
                    parts = []
                    for lbl, grp in clean.groupby(label_col):
                        if len(grp) <= min_per_class_clean:
                            parts.append(grp)
                        else:
                            parts.append(
                                grp.sample(n=min_per_class_clean, random_state=42)
                            )
                    if parts:
                        clean = pd.concat(parts, ignore_index=True)

                cleanset_df = clean
                print(f"[CleanSet] Total sampel setelah filter: {len(cleanset_df)}")
                _print_label_stats(
                    cleanset_df,
                    "CleanSet (subset Stage2, R/Q/entropy/margin)",
                    label_col=label_col,
                )


    # Siapkan output dirs
    processed_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Kolom minimal untuk manifest student
    # (ambil semua kolom train_df + skor/tier/stage; hindari duplikasi nama)
    # train_df pada titik ini sudah diperkaya metadata dari master_list
    base_cols = [c for c in train_df.columns]
    extra_cols = [c for c in stage2_df.columns if c not in base_cols]
    ordered_cols = base_cols + extra_cols

    # Simpan per-stage
    out_stage0 = processed_dir / f"{prefix}student_stage0_train.csv"
    out_stage1 = processed_dir / f"{prefix}student_stage1_train.csv"
    out_stage2 = processed_dir / f"{prefix}student_stage2_train.csv"

    stage0_df[ordered_cols].to_csv(out_stage0, index=False)
    stage1_df[ordered_cols].to_csv(out_stage1, index=False)
    stage2_df[ordered_cols].to_csv(out_stage2, index=False)

    print("-" * 80)
    print(f"✓ Saved Stage0 manifest : {out_stage0}")
    print(f"✓ Saved Stage1 manifest : {out_stage1}")
    print(f"✓ Saved Stage2 manifest : {out_stage2}")

    # Simpan CleanSet (jika diaktifkan dan berhasil dibangun)
    cleanset_out_path = None
    if cleanset_df is not None:
        cleanset_filename = cleanset_cfg.get(
            "output_name", "student_cleanset_train.csv"
        )
        # Jika prefix ada, tambahkan ke filename (kecuali filename sudah mengandung prefix)
        if prefix and not cleanset_filename.startswith(prefix):
            cleanset_filename = f"{prefix}{cleanset_filename}"
            
        cleanset_out_path = processed_dir / cleanset_filename

        # Ikuti pola kolom yang sama: base_cols + kolom tambahan yang unik
        c_extra_cols = [c for c in cleanset_df.columns if c not in base_cols]
        c_ordered_cols = base_cols + c_extra_cols

        cleanset_df[c_ordered_cols].to_csv(cleanset_out_path, index=False)
        print(f"✓ Saved CleanSet manifest : {cleanset_out_path}")

    # Dataset final (pakai Stage 2)
    final_dataset = stage2_df[ordered_cols].copy()
    final_path = final_dir / f"{prefix}coweps_final_dataset.csv"
    final_dataset.to_csv(final_path, index=False)
    print(f"✓ Final dataset (Stage2)   : {final_path}")
    print("=" * 80)

    return {
        "success": True,
        "stage0_path": str(out_stage0),
        "stage1_path": str(out_stage1),
        "stage2_path": str(out_stage2),
        "final_path": str(final_path),
        "stage0_n": len(stage0_df),
        "stage1_n": len(stage1_df),
        "stage2_n": len(stage2_df),
        "cleanset_path": str(cleanset_out_path)
        if (cleanset_df is not None and cleanset_out_path is not None)
        else None,
        "cleanset_n": int(len(cleanset_df)) if cleanset_df is not None else 0,
    }



def main():
    ap = argparse.ArgumentParser(
        description="CoWePS v2.5 - Progressive Sampling (Fase 4)"
    )
    ap.add_argument(
        "--base",
        required=True,
        help="Path ke base_config_coweps.yaml",
    )
    ap.add_argument(
        "--prefix",
        default="",
        help="Prefix untuk nama file output (misal: 'rev41_')",
    )
    args = ap.parse_args()

    print("Fase 4: Progressive Sampling (Student Dataset)")
    print("=" * 80 + "\n")

    results = run_progressive_sampling(args.base, prefix=args.prefix)

    if results.get("success", False):
        print("\n✅ Progressive sampling completed successfully!")
        print(f"Stage0 samples: {results['stage0_n']}")
        print(f"Stage1 samples: {results['stage1_n']}")
        print(f"Stage2 samples: {results['stage2_n']}")
        print(f"Final dataset : {results['final_path']}")
    else:
        print("\n❌ Progressive sampling failed!")
        if "error" in results:
            print(f"Error: {results['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
