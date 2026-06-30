"""
CoWePS v2.5 - Manifest Builder (modular & offline, auto-detect raw layout, mask-safe)

Perbaikan:
- Scan RAW akan MENGABAIKAN file bernama *_mask.* (agar masker tidak dianggap image).
- Setelah scan RAW: harmonize_labels() + audit_and_attach_masks() selalu dipanggil.
- Deduplikasi berdasarkan image_path.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _infer_label_col(df: pd.DataFrame) -> str:
    for c in ["label", "weak_label_class", "grade"]:
        if c in df.columns:
            return c
    raise ValueError("Manifest sumber harus memiliki salah satu kolom: label | weak_label_class | grade.")

def _ensure_columns(df: pd.DataFrame, needed: List[str]) -> pd.DataFrame:
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _resolve_paths(base: str, maybe_path: str) -> str:
    p = Path(maybe_path)
    if p.exists():
        return str(p)
    return str(Path(base) / maybe_path)

def _is_mask_filename(p: Path) -> bool:
    """True bila nama file adalah masker (mengandung '_mask.' sebelum ekstensi)."""
    stem = p.stem.lower()
    name = p.name.lower()
    return ("_mask" in stem) or name.endswith("_mask.png") or name.endswith("_mask.jpg") or name.endswith("_mask.jpeg") or name.endswith("_mask.bmp") or name.endswith("_mask.tif") or name.endswith("_mask.tiff")

# -----------------------------------------------------------------------------
# Scanner data/raw (auto-detect dua layout)
# -----------------------------------------------------------------------------

def _scan_raw_dir(raw_dir: str) -> pd.DataFrame:
    """
    Auto-detect:
      A) Hierarki: raw/<SOURCE>/<0..4>/*.{png,jpg,...} → source=<SOURCE>
      B) Flat    : raw/<0..4>/*.{png,jpg,...}          → source="FLAT"
    Mengabaikan file *_mask.* agar tidak terbaca sebagai citra input.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    raw = Path(raw_dir)
    if not raw.exists():
        raise FileNotFoundError(f"Raw dir not found: {raw_dir}")

    rows: List[Dict] = []

    child_dirs = [d for d in raw.iterdir() if d.is_dir()]
    class_set = {"0", "1", "2", "3", "4"}
    child_names = {d.name for d in child_dirs}

    # Longgar: cukup 0..4 ada → flat. Folder lain diabaikan.
    is_flat = class_set.issubset(child_names)

    print(f"[scan_raw] child_dirs = {sorted(child_names)}")
    print(f"[scan_raw] layout     = {'FLAT' if is_flat else 'HIERARCHICAL'} under {raw_dir}")

    if is_flat:
        for cls_dir in sorted(child_dirs, key=lambda p: p.name):
            if cls_dir.name not in class_set:
                continue
            y = int(cls_dir.name)
            found = 0
            for img in cls_dir.rglob("*"):
                if img.is_file() and img.suffix.lower() in exts and not _is_mask_filename(img):
                    rows.append({
                        "image_path": str(img.resolve()),
                        "label": y,
                        "source": "FLAT"
                    })
                    found += 1
            print(f"[scan_raw][FLAT] class {y}: +{found} files (masked files ignored)")
    else:
        for source_dir in sorted(child_dirs, key=lambda p: p.name):
            if not source_dir.is_dir():
                continue
            source_name = source_dir.name
            subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
            subnames = {d.name for d in subdirs}
            has_any_class = bool(subnames.intersection(class_set))
            if not has_any_class:
                continue
            for cls_dir in sorted(subdirs, key=lambda p: p.name):
                if cls_dir.name not in class_set:
                    continue
                y = int(cls_dir.name)
                found = 0
                for img in cls_dir.rglob("*"):
                    if img.is_file() and img.suffix.lower() in exts and not _is_mask_filename(img):
                        rows.append({
                            "image_path": str(img.resolve()),
                            "label": y,
                            "source": source_name
                        })
                        found += 1
                print(f"[scan_raw][SRC={source_name}] class {y}: +{found} files (masked files ignored)")

    if not rows:
        example = [str(p) for p in raw.rglob("*")][:10]
        raise RuntimeError(
            "Tidak menemukan gambar pada struktur "
            f"{raw_dir}/<source>/<0..4>/*.ext ATAU {raw_dir}/<0..4>/*.ext\n"
            f"Hint: contoh isi awal raw (10 entri): {example}"
        )

    df = pd.DataFrame(rows)
    print(f"[scan_raw] Found {len(df)} images total (after mask filtering).")
    return df

# -----------------------------------------------------------------------------
# Harmonisasi & audit
# -----------------------------------------------------------------------------

def harmonize_labels(df: pd.DataFrame) -> pd.DataFrame:
    label_col = _infer_label_col(df)
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    if (df["label"] < 0).any() or (df["label"] > 4).any():
        print("[harmonize_labels] WARNING: label di luar 0..4 ditemukan. Akan di-clip ke [0,4].")
        df["label"] = df["label"].clip(lower=0, upper=4)
    return df

def audit_and_attach_masks(df: pd.DataFrame) -> pd.DataFrame:
    if "mask_path" not in df.columns:
        def infer_mask(p):
            p = str(p)
            base, _ = os.path.splitext(p)
            return base + "_mask.png"
        df["mask_path"] = df["image_path"].map(infer_mask)
    df["exists_mask"] = df["mask_path"].map(lambda p: Path(p).exists())
    return df

def _merge_sources_csv(source_csvs: List[str], base_dir: Optional[str] = None) -> pd.DataFrame:
    frames = []
    for csv_p in source_csvs:
        p = Path(csv_p) if base_dir is None else Path(_resolve_paths(base_dir, csv_p))
        if not p.exists():
            raise FileNotFoundError(f"Sumber CSV tidak ditemukan: {csv_p}")
        f = pd.read_csv(p)
        lbl = _infer_label_col(f)
        keep_cols = ["image_path", lbl, "patient_id", "source", "device", "year", "mask_path"]
        f = _ensure_columns(f, [c for c in keep_cols if c not in f.columns])
        f = f[keep_cols].copy()
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df = harmonize_labels(df)
    df = audit_and_attach_masks(df)
    return df

# -----------------------------------------------------------------------------
# Core APIs
# -----------------------------------------------------------------------------

def build_master_list(
    base_config: Dict,
    source_csvs: Optional[List[str]] = None,
    strict_paths: bool = False
) -> pd.DataFrame:
    """
    Bangun master_list.csv.
    Mode A: Berikan `source_csvs` (disarankan) → merge & harmonize.
    Mode B: Jika None → scan folder data/raw (auto-detect flat/hierarki), lalu harmonize+audit.
    """
    paths = base_config.get("paths", {}) or {}
    raw_dir = paths.get("raw_dir") or os.path.join(paths.get("data_root", ""), "raw")
    if not raw_dir:
        raise ValueError("Base config 'paths.raw_dir' atau 'paths.data_root' harus diisi.")

    if source_csvs:
        print("[build_master_list] Using source CSVs mode.")
        master = _merge_sources_csv(source_csvs, base_dir=Path(paths.get("data_root", ".")))
    else:
        print("[build_master_list] Using SCAN RAW mode (auto-detect layout).")
        master = _scan_raw_dir(raw_dir)
        # Penting: harmonize + attach masks JUGA untuk jalur scan raw
        master = harmonize_labels(master)
        master = audit_and_attach_masks(master)

    # Absolutkan path
    master["image_path"] = master["image_path"].map(lambda p: str(Path(p).resolve()))

    # Kolom meta opsional
    for c in ["patient_id", "source", "device", "year"]:
        if c not in master.columns:
            master[c] = np.nan

    # Deduplikasi based on image_path
    before = len(master)
    master = master.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    after = len(master)
    if after < before:
        print(f"[build_master_list] Deduplicated {before - after} rows by image_path.")

    # Audit eksistensi path bila diminta
    if strict_paths:
        missing = (~master["image_path"].map(lambda p: Path(p).exists())).sum()
        if missing > 0:
            raise FileNotFoundError(f"Ada {missing} image_path yang tidak ditemukan di disk.")

    # Simpan master snapshot
    processed = Path(paths.get("processed_dir", "data/processed"))
    out_p = processed / "master_list.csv"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_p, index=False)
    print(f"[build_master_list] ✓ master_list.csv ditulis: {out_p} (rows={len(master)})")
    return master

def patient_disjoint_split(
    df: pd.DataFrame,
    val_size: float = 0.20,
    random_state: int = 42,
    strict_per_source: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_col = _infer_label_col(df)
    has_patient = "patient_id" in df.columns and df["patient_id"].notna().any()

    if strict_per_source and "source" in df.columns and df["source"].notna().any():
        frames_tr, frames_va = [], []
        for _, dfg in df.groupby(df["source"].fillna("unknown")):
            tr_s, va_s = _split_single(dfg, label_col, has_patient, val_size, random_state)
            frames_tr.append(tr_s); frames_va.append(va_s)
        train = pd.concat(frames_tr, ignore_index=True)
        val   = pd.concat(frames_va, ignore_index=True)
        return train, val
    else:
        return _split_single(df, label_col, has_patient, val_size, random_state)

def _split_single(
    df: pd.DataFrame,
    label_col: str,
    has_patient: bool,
    val_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if has_patient:
        n_splits = max(2, int(round(1.0 / val_size)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        idx = np.arange(len(df))
        for tr_idx, va_idx in sgkf.split(idx, y=df[label_col].values, groups=df["patient_id"].astype(str).values):
            train = df.iloc[tr_idx].copy()
            val   = df.iloc[va_idx].copy()
            break
    else:
        n_splits = max(2, int(round(1.0 / val_size)))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        idx = np.arange(len(df))
        for tr_idx, va_idx in skf.split(idx, y=df[label_col].values):
            train = df.iloc[tr_idx].copy()
            val   = df.iloc[va_idx].copy()
            break
    return train, val

def write_manifests(
    base_config: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Dict[str, str]:
    paths = base_config.get("paths", {}) or {}
    processed = Path(paths.get("processed_dir", "data/processed"))
    manifests = base_config.get("manifests", {}) or {}

    out_train = Path(manifests.get("train", processed / "gold_standard_train.csv"))
    out_val   = Path(manifests.get("validate", processed / "gold_standard_validate.csv"))
    snap_tr   = Path(manifests.get("training_manifest", processed / "training_manifest.csv"))
    snap_va   = Path(manifests.get("validation_manifest", processed / "validation_manifest.csv"))

    for p in [out_train, out_val, snap_tr, snap_va]:
        p.parent.mkdir(parents=True, exist_ok=True)

    keep_main = ["image_path", "mask_path", "label", "patient_id", "source", "device", "year"]
    train_df = _ensure_columns(train_df, keep_main)[keep_main]
    val_df   = _ensure_columns(val_df, keep_main)[keep_main]

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    train_df.to_csv(snap_tr, index=False)
    val_df.to_csv(snap_va, index=False)

    print(f"[write_manifests] ✓ train: {out_train} (rows={len(train_df)})")
    print(f"[write_manifests] ✓ valid: {out_val} (rows={len(val_df)})")
    return {
        "train": str(out_train),
        "validate": str(out_val),
        "training_manifest": str(snap_tr),
        "validation_manifest": str(snap_va),
    }
