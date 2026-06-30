#!/usr/bin/env python
"""Manifest utilities for CoWePS.

Tujuan:
- Menjadikan `master_list.csv` sebagai single source of truth.
- Memastikan semua manifest turunan (gold_standard, tiered train sets,
  CleanSet, Naive, student splits, dsb.) tetap membawa kolom kunci
  seperti `id_patient` dan metadata dasar lain.

Fungsi di file ini sengaja dibuat ringan, tanpa dependensi selain pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


# Kolom minimal yang diharapkan ada di master_list
REQUIRED_MASTER_COLS: List[str] = [
    "image_path",
    "label",
    "source",
    "mask_path",
    "device",
    "year",
    "id_patient",
]


@dataclass
class MasterSpec:
    """Spesifikasi lokasi master_list.

    Bisa dipanggil dengan path eksplisit atau dengan base_config
    (yang punya `paths.processed_dir`).
    """

    path: Path

    @staticmethod
    def from_base_config(base_cfg: dict, default: str = "data/processed/master_list.csv") -> "MasterSpec":
        paths = (base_cfg or {}).get("paths", {}) or {}
        processed_dir = Path(paths.get("processed_dir", "data/processed"))
        raw_path = paths.get("master_list", None)
        if raw_path:
            p = Path(raw_path)
        else:
            p = processed_dir / "master_list.csv"
        return MasterSpec(path=p if p.is_absolute() else p)


def load_master(master: MasterSpec | str | Path) -> pd.DataFrame:
    """Load master_list.csv dan cek kolom wajib.

    Parameters
    ----------
    master : MasterSpec | str | Path
        Lokasi master_list.
    """

    if isinstance(master, MasterSpec):
        path = master.path
    else:
        path = Path(master)

    if not path.exists():
        raise FileNotFoundError(f"master_list tidak ditemukan: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_MASTER_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "master_list tidak memiliki kolom wajib: "
            + ", ".join(missing)
            + f". Path: {path}"
        )

    return df


def attach_master_metadata(
    df: pd.DataFrame,
    master: MasterSpec | str | Path,
    required_cols: Optional[Iterable[str]] = None,
    on: str = "image_path",
    context: str = "manifest",
) -> pd.DataFrame:
    """Join metadata dari master_list ke DataFrame lain via `image_path`.

    - Tidak mengubah jumlah baris (left join).
    - Jika kolom sudah ada di df, versi df yang dipertahankan.
    - Melempar error bila ada kolom wajib yang hilang setelah join.
    """

    if on not in df.columns:
        raise ValueError(f"[{context}] DataFrame tidak punya kolom kunci '{on}'.")

    master_df = load_master(master)

    # Hanya ambil kolom yang relevan untuk join metadata
    if required_cols is None:
        # Default: semua kolom wajib master, kecuali key join
        required_cols = [c for c in REQUIRED_MASTER_COLS if c != on]

    required_cols = list(required_cols)

    take_cols: List[str] = [on]
    for col in required_cols:
        if col in master_df.columns and col not in take_cols:
            take_cols.append(col)

    meta = master_df[take_cols].copy()

    # Hindari suffix yang bertele-tele: jika bentrok nama, pertahankan versi df
    # dan drop kolom *_master setelah merge.
    merged = df.merge(meta, on=on, how="left", suffixes=("", "_master"))

    # Resolusi conflict: jika ada kolom X_master, drop saja (prioritas ke X asli)
    conflict_cols = [c for c in merged.columns if c.endswith("_master")]
    if conflict_cols:
        merged = merged.drop(columns=conflict_cols)

    # Validasi kolom wajib ada
    missing_after = [c for c in REQUIRED_MASTER_COLS if c not in merged.columns]
    if missing_after:
        raise ValueError(
            f"[{context}] Manifest setelah attach_master_metadata masih kehilangan kolom: "
            + ", ".join(missing_after)
        )

    return merged


def ensure_required_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    context: str = "manifest",
) -> None:
    """Validasi bahwa DataFrame memiliki kolom yang dibutuhkan.

    Fungsi ini tidak mengubah df, hanya melempar error jika ada yang hilang.
    """

    required = list(required)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] Kolom wajib hilang: {', '.join(missing)}")
