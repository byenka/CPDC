#!/usr/bin/env python
"""
CoWePS v2.5 Progressive Sampling Module
Fase 4: Tiered + Curriculum + Domain×Class Quotas

Perubahan utama dari v2.4:
- BUKAN lagi "filter Q==1 + sort C_score saja".
- Memakai tiering A/B/C (entropy, margin, agreement) dari base_config.
- Sampling per-stage (curriculum): Stage-0 (A), Stage-1 (A+B), Stage-2 (A+B+subset C).
- Kuota minimum per kombinasi (domain × class).
- Tetap menghormati Q_score==1 jika tersedia (opsional).

Input  default: data/scores/full_inference_results_<mode>.csv (Fase 3)
Tiers default: data/processed/tier_A.csv, tier_B.csv, tier_C.csv (jika ada)
Output: data/final/coweps_final_dataset.csv  (+ metadata JSON)

Catatan:
- Tetap menggunakan utilitas logging & load_config bawaan proyek.
- Tidak ada network download; seluruh bobot dan artefak harus lokal.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Tambahkan root proyek ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.utils import load_config, setup_logging

# ---------------------------------------------------------------------------
# Utilitas tiering (sinkron dengan Fase 3)
# ---------------------------------------------------------------------------

def _assign_tiers_from_thresholds(df: pd.DataFrame, base_config: dict) -> pd.DataFrame:
    """
    Jika kolom 'tier' belum tersedia dari Fase 3, tetapkan tier A/B/C
    berdasarkan thresholds pada base_config['tiering'].
    Menggunakan kolom: 'entropy', 'margin', (opsional) 'agreement'.
    """
    tcfg = base_config.get('tiering', {})
    if not tcfg or not tcfg.get('use', False):
        df['tier'] = 'A'
        return df

    th = tcfg.get('thresholds', {})
    def decide_row(row):
        ent = row.get('entropy', np.nan)
        mar = row.get('margin', np.nan)
        agr = row.get('agreement', np.nan) if 'agreement' in row else np.nan

        a_ok = True
        b_ok = True

        if 'entropy' in th:
            a_ok &= (ent <= th['entropy'].get('a_max', np.inf))
            b_ok &= (ent <= th['entropy'].get('b_max', np.inf))

        if 'margin' in th:
            a_ok &= (mar >= th['margin'].get('a_min', -np.inf))
            b_ok &= (mar >= th['margin'].get('b_min', -np.inf))

        if 'agreement' in th and not np.isnan(agr):
            a_ok &= (agr >= th['agreement'].get('a_min', -np.inf))
            b_ok &= (agr >= th['agreement'].get('b_min', -np.inf))

        if a_ok:
            return 'A'
        if b_ok:
            return 'B'
        return 'C'

    df['tier'] = df.apply(decide_row, axis=1)
    return df


def _load_tier_manifests_if_any(base_config: dict) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Coba muat tier_A/B/C.csv jika tersedia. Jika tidak, kembalikan None.
    """
    tcfg = base_config.get('tiering', {})
    outdir = tcfg.get('outputs', {}).get('dir') or base_config['paths']['processed_dir']
    A_path = os.path.join(outdir, tcfg.get('outputs', {}).get('tier_a', 'tier_A.csv'))
    B_path = os.path.join(outdir, tcfg.get('outputs', {}).get('tier_b', 'tier_B.csv'))
    C_path = os.path.join(outdir, tcfg.get('outputs', {}).get('tier_c', 'tier_C.csv'))

    tiers = {'A': None, 'B': None, 'C': None}
    if os.path.exists(A_path):
        tiers['A'] = pd.read_csv(A_path)
    if os.path.exists(B_path):
        tiers['B'] = pd.read_csv(B_path)
    if os.path.exists(C_path):
        tiers['C'] = pd.read_csv(C_path)
    return tiers


# ---------------------------------------------------------------------------
# Seleksi per-stage (curriculum) dengan kuota domain×kelas
# ---------------------------------------------------------------------------

def _apply_q_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Terapkan quality gate bila Q_score tersedia.
    Jika kolom Q_score tidak ada, kembalikan df apa adanya.
    """
    if 'Q_score' in df.columns:
        return df[df['Q_score'] == 1].copy()
    return df.copy()


def _weighted_confidence(df: pd.DataFrame, tier_weights: Dict[str, float]) -> pd.Series:
    """
    Skor seleksi = C_score * tier_weight[tier].
    Jika kolom C_score tidak ada, gunakan probabilitas maksimum 'C_score' yang
    seharusnya sudah ada dari Fase 3 (syarat minimal).
    """
    w = df['tier'].map(lambda t: float(tier_weights.get(t, 1.0)))
    return df['C_score'] * w


def _ensure_columns(df: pd.DataFrame):
    """
    Pastikan kolom minimal exist untuk downstream:
    image_path, Pred_Class, C_score, tier, (opsional) source, device, year, label
    """
    required = ['image_path', 'Pred_Class', 'C_score']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in scores: {missing}")

    if 'tier' not in df.columns:
        df['tier'] = 'A'

    # Normalisasi tipe
    df['Pred_Class'] = df['Pred_Class'].astype(int)
    return df


def _clip_candidates_C(df_c: pd.DataFrame, clip_cfg: dict) -> pd.DataFrame:
    """
    Batasi tier C sesuai konfigurasi (entropy_max, min_per_class, dsb.) jika ada.
    """
    if df_c.empty or not clip_cfg:
        return df_c

    # Entropy max
    ent_max = clip_cfg.get('entropy_max', None)
    if ent_max is not None and 'entropy' in df_c.columns:
        df_c = df_c[df_c['entropy'] <= float(ent_max)]

    # min_per_class hanya dipastikan di luar (saat agregasi akhir).
    return df_c


def _select_per_class_with_domain_quota(
    df_stage: pd.DataFrame,
    target_per_class: int,
    quotas_cfg: dict,
    logger
) -> pd.DataFrame:
    """
    Seleksi per kelas dengan kuota minimum per domain×kelas (jika meta tersedia).
    Strategi:
      1) Untuk setiap class, bagi kandidat by (source, class) → tarik minimal min_per_domain_class
         selama tersedia (prioritas C_weight tinggi).
      2) Sisa kuota diisi oleh kandidat terbaik dari class tersebut tanpa memaksa domain.
    """
    use_quota = quotas_cfg.get('by_domain_class', False)
    min_per_dc = int(quotas_cfg.get('min_per_domain_class', 0))

    selected_parts = []

    classes = sorted(df_stage['Pred_Class'].unique().tolist())
    for cls in classes:
        pool_c = df_stage[df_stage['Pred_Class'] == cls].copy()
        if pool_c.empty:
            logger.warning(f"[Class {cls}] No candidates available.")
            continue

        # Urutkan berdasarkan 'select_score' yang sudah dihitung
        if 'select_score' in pool_c.columns:
            pool_c = pool_c.sort_values(by='select_score', ascending=False)
        else:
            pool_c = pool_c.sort_values(by='C_score', ascending=False)

        # 1) Tarik kuota minimum per domain×kelas jika meta tersedia
        hard_take = []
        if use_quota and min_per_dc > 0 and 'source' in pool_c.columns:
            for src, grp in pool_c.groupby('source'):
                take_n = min(min_per_dc, len(grp))
                if take_n > 0:
                    hard_take.append(grp.head(take_n))
            pre = pd.concat(hard_take, ignore_index=True) if hard_take else pd.DataFrame(columns=pool_c.columns)
        else:
            pre = pd.DataFrame(columns=pool_c.columns)

        # 2) Lengkapi sisa kuota dari pool yang belum terambil
        already = set(pre.index)
        remaining_needed = max(0, target_per_class - len(pre))
        if remaining_needed > 0:
            rest = pool_c.drop(index=pre.index, errors='ignore').head(remaining_needed)
            final_cls = pd.concat([pre, rest], ignore_index=True)
        else:
            final_cls = pre.head(target_per_class)

        # Report ringkas
        logger.info(
            f"[Class {cls}] selected {len(final_cls)}/{target_per_class} "
            f"(quota per domain {min_per_dc if use_quota else 0})"
        )
        selected_parts.append(final_cls)

    if not selected_parts:
        return pd.DataFrame(columns=df_stage.columns)

    return pd.concat(selected_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------

def run_progressive_sampling(config_path: str = 'configs/base_config_coweps.yaml',
                             scores_file: Optional[str] = None,
                             mode: str = 'gold_standard') -> Dict:
    """
    Jalankan CoWePS v2.5 progressive sampling.
    - Membaca base_config (tiering, sampling, quotas)
    - Memuat hasil Fase 3 (scores) + (opsional) tier manifests
    - Menyusun seleksi bertahap sesuai curriculum
    - Menulis final dataset + metadata

    Args:
        config_path: path ke base config
        scores_file: path ke CSV hasil Fase 3; jika None, akan di-resolve dari config
        mode: 'master' | 'train' | 'val' | 'gold_standard' (hanya untuk penamaan file default)
    """
    # 1) Load config & logger
    if isinstance(config_path, dict):
        config = config_path
        cfg_name = 'config_dict'
    else:
        config = load_config(config_path)
        cfg_name = config_path

    logger = setup_logging(
        config['paths']['logs_dir'],
        'progressive_sampling_v25',
        config
    )

    logger.info("\n" + "="*80)
    logger.info("FASE 4: COWEPS V2.5 PROGRESSIVE SAMPLING")
    logger.info("Tiered + Curriculum + Domain×Class Quotas")
    logger.info("="*80)
    logger.info(f"Config: {cfg_name}")

    # 2) Resolve paths
    scores_dir = config['paths']['scores_dir']
    final_dir  = config['paths']['final_dir']
    os.makedirs(final_dir, exist_ok=True)

    if scores_file is None:
        # Gunakan path default dari base config inference.save_scores_csv,
        # jika mengandung <mode> gunakan itu, jika tidak, fallback ke scores_dir
        default_scores = config.get('inference', {}).get('save_scores_csv', None)
        if default_scores and os.path.exists(default_scores):
            scores_file = default_scores
        else:
            # fallback nama umum
            cand = os.path.join(scores_dir, f'full_inference_results_{mode}.csv')
            scores_file = cand if os.path.exists(cand) else os.path.join(scores_dir, 'full_inference_results.csv')

    final_output_file = os.path.join(final_dir, 'coweps_final_dataset.csv')
    meta_output_file  = os.path.join(final_dir, 'coweps_final_metadata.json')

    logger.info(f"\nPaths:")
    logger.info(f"  Scores   : {scores_file}")
    logger.info(f"  Final    : {final_output_file}")
    logger.info(f"  Metadata : {meta_output_file}")

    # 3) Load scores
    if not os.path.exists(scores_file):
        err = f"Scores file not found: {scores_file}"
        logger.error(err)
        return {'success': False, 'error': err}

    df = pd.read_csv(scores_file)
    logger.info(f"✓ Loaded scores: {len(df)} rows")

    # 4) Pastikan kolom-kolom penting
    df = _ensure_columns(df)

    # 5) Pastikan ada 'tier' (pakai dari Fase 3 jika ada; jika tidak, hitung dari thresholds)
    if 'tier' not in df.columns:
        logger.info("Tier column not found in scores → assigning tiers from thresholds...")
        df = _assign_tiers_from_thresholds(df, config)

    # 6) Terapkan Q gate bila ada
    df = _apply_q_gate(df)
    logger.info(f"After Q gate: {len(df)} candidates")

    # 7) Jika tier manifests tersedia (A/B/C), gunakan sebagai referensi field tambahan (source/device/…)
    tiers_ref = _load_tier_manifests_if_any(config)
    # (opsional) dapat dipakai untuk validasi, namun tidak wajib di-merge di sini.

    # 8) Ambil curriculum dari config
    scfg = config.get('sampling', {})
    stages = scfg.get('stages', [
        {'name': 'stage0_tierA_only', 'include_tiers': ['A'], 'tier_weights': {'A': 1.0}},
        {'name': 'stage1_tierA_B', 'include_tiers': ['A', 'B'], 'tier_weights': {'A': 1.0, 'B': 0.5}},
        {'name': 'stage2_tierA_B_Csubset', 'include_tiers': ['A', 'B', 'C'],
         'tier_weights': {'A': 1.0, 'B': 0.7, 'C': 0.2}, 'c_clip': {'entropy_max': 1.10}}
    ])
    quotas_cfg = scfg.get('quotas', {'by_domain_class': False, 'min_per_domain_class': 0})

    TARGET_PER_CLASS = scfg.get('target_per_class', 1038)
    NUM_CLASSES = int(config.get('model', {}).get('num_classes', 5))

    # 9) Jalankan tiap stage → tumpuk hasilnya unik per image_path
    selected_accum = pd.DataFrame(columns=df.columns)
    for stage in stages:
        name = stage.get('name', 'stage')
        inc_tiers = stage.get('include_tiers', ['A'])
        tier_weights = stage.get('tier_weights', {t: 1.0 for t in inc_tiers})
        c_clip = stage.get('c_clip', {})

        logger.info("\n" + "="*80)
        logger.info(f"STAGE: {name}")
        logger.info("="*80)
        logger.info(f"Include tiers: {inc_tiers}")
        logger.info(f"Tier weights : {tier_weights}")

        pool = df[df['tier'].isin(inc_tiers)].copy()
        if 'C' in inc_tiers:
            pool_c = pool[pool['tier'] == 'C']
            pool_nonc = pool[pool['tier'] != 'C']
            pool_c = _clip_candidates_C(pool_c, c_clip)
            pool = pd.concat([pool_nonc, pool_c], ignore_index=True)

        # Hitung skor seleksi (C_score * weight)
        pool['select_score'] = _weighted_confidence(pool, tier_weights)

        # Buang yang sudah pernah terambil di stage sebelumnya (berdasarkan image_path)
        if not selected_accum.empty and 'image_path' in selected_accum.columns:
            pool = pool[~pool['image_path'].isin(selected_accum['image_path'])]

        # Seleksi per-kelas dengan kuota domain×kelas
        stage_sel = _select_per_class_with_domain_quota(
            df_stage=pool,
            target_per_class=TARGET_PER_CLASS,
            quotas_cfg=quotas_cfg,
            logger=logger
        )

        logger.info(f"Stage '{name}' selected: {len(stage_sel)} samples")
        selected_accum = pd.concat([selected_accum, stage_sel], ignore_index=True)

        # Early stop jika seluruh kelas sudah memenuhi target
        ok = True
        for cls in range(NUM_CLASSES):
            cnt = len(selected_accum[selected_accum['Pred_Class'] == cls])
            if cnt < TARGET_PER_CLASS:
                ok = False
                break
        if ok:
            logger.info("All classes reached target after this stage. Stopping curriculum.")
            break

    # 10) Potong ke target final per kelas (kalau overfill)
    final_list = []
    for cls in range(NUM_CLASSES):
        cls_pool = selected_accum[selected_accum['Pred_Class'] == cls]
        if cls_pool.empty:
            continue
        # Priotitaskan skor seleksi tertinggi
        cls_pool = cls_pool.sort_values(by=['select_score', 'C_score'], ascending=False)
        final_list.append(cls_pool.head(TARGET_PER_CLASS))
    if not final_list:
        err = "No samples selected for any class after curriculum."
        logger.error(err)
        return {'success': False, 'error': err}

    final_df = pd.concat(final_list, ignore_index=True)

    # 11) Simpan hasil dan metadata
    # Kolom minimal untuk downstream manifest (konservatif)
    keep_cols = [c for c in [
        'image_path', 'mask_path', 'Pred_Class', 'C_score', 'tier',
        'entropy', 'margin', 'agreement', 'Q_score', 'Q_score_continuous',
        'source', 'device', 'year', 'label'
    ] if c in final_df.columns]
    final_df[keep_cols].to_csv(final_output_file, index=False)

    class_dist = final_df['Pred_Class'].value_counts().sort_index()
    metadata = {
        'total_selected': int(len(final_df)),
        'target_per_class': int(TARGET_PER_CLASS),
        'num_classes': int(NUM_CLASSES),
        'class_distribution': {int(k): int(v) for k, v in class_dist.to_dict().items()},
        'used_stages': [s.get('name', '') for s in stages],
        'quotas': quotas_cfg
    }
    with open(meta_output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("FASE 4 COMPLETE - FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total selected: {metadata['total_selected']}")
    for cls, cnt in class_dist.items():
        logger.info(f"Class {cls}: {int(cnt)}/{TARGET_PER_CLASS}")

    return {
        'success': True,
        'final_path': final_output_file,
        'metadata_path': meta_output_file,
        'total_samples': metadata['total_selected'],
        'samples_per_class': {int(k): int(v) for k, v in class_dist.to_dict().items()}
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CoWePS v2.5 Progressive Sampling')
    parser.add_argument('--config', type=str, default='configs/base_config_coweps.yaml')
    parser.add_argument('--scores', type=str, default=None)
    parser.add_argument('--mode', type=str, default='gold_standard',
                        choices=['master', 'train', 'val', 'gold_standard'])
    args = parser.parse_args()

    print("\n" + "="*80)
    print("CoWePS v2.5 Progressive Sampling Pipeline")
    print("Fase 4: Tier + Curriculum + Domain×Class Quotas")
    print("="*80 + "\n")

    res = run_progressive_sampling(args.config, args.scores, args.mode)
    if res.get('success', False):
        print("\n✅ Progressive sampling completed successfully!")
        print(f"Total samples selected: {res['total_samples']}")
        print(f"Results saved to: {res['final_path']}")
    else:
        print("\n❌ Progressive sampling failed!")
        print(f"Error: {res.get('error', 'Unknown error')}")
        sys.exit(1)
