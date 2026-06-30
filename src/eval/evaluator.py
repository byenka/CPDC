"""
CoWePS v2.5 - Evaluator
Evaluasi model (ID/OOD aware) dan/atau evaluasi dari CSV skor, dengan keluaran
laporan JSON + CSV. Sepenuhnya offline (dilarang download).

Dua mode utama:
1) evaluate_model(config_model_yaml, base_config_yaml, split='val'):
   - Load model dari checkpoint lokal (model_factory)
   - Load manifest split (train/val) via DRDataset
   - Forward → logits → probs → metrik (BA, F1, ACC, ECE, Brier, Entropy)
   - Simpan predictions.csv (berisi p0..p{C-1}), dan report.json

2) evaluate_from_scores(scores_csv, manifest_csv, base_config_yaml):
   - Merge label dari manifest → metrik yang sama,
     ECE/Brier dihitung HANYA jika kolom p0..p{C-1} tersedia.
   - Jika tidak ada kolom p*, tetap dihitung BA/F1/ACC + CM.

Output default mengikuti base_config:
  evaluation.report_dir  (mis. /.../outputs/reports)
"""

from __future__ import annotations
import os
import sys
import json
import re
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm

# path project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_processing import DRDataset
from src.models.model_factory import create_model_from_config
from src.metrics.metrics import (
    metric_accuracy, metric_balanced_accuracy, metric_f1_macro,
    metric_confusion_matrix, metric_ece, metric_brier_multiclass,
    metric_entropy_from_probs, per_class_report, reliability_bins,
    grouped_metrics
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_report_dir(base_config: Dict) -> str:
    rep = base_config.get('evaluation', {}).get('report_dir') \
          or os.path.join(base_config['paths']['outputs_dir'], 'reports')
    os.makedirs(rep, exist_ok=True)
    return rep


def _infer_label_col(df: pd.DataFrame) -> str:
    for c in ['label', 'weak_label_class', 'grade']:
        if c in df.columns:
            return c
    raise ValueError("Manifest must contain 'label' or 'weak_label_class' or 'grade'.")


def _collect_logits_labels(model, loader, device):
    logits_list, labels_list, paths_list, sources = [], [], [], []
    for batch in tqdm(loader, desc="Evaluating"):
        if len(batch) == 2:
            images, labels = batch
            meta = None
        else:
            images, labels, meta = batch
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(images)
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())
        # image_path akan di-serialize via manifest urutan loader
        # bila DRDataset tidak mengembalikan meta, kita isi None
        if meta is not None and isinstance(meta, dict) and 'image_path' in meta:
            paths_list.extend(meta['image_path'])
            sources.extend(meta.get('source', ['unknown'] * len(meta['image_path'])))
        else:
            # fallback: tanpa meta → biarkan kosong (nanti merge dari manifest)
            pass

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return logits, labels, paths_list, sources


def _prob_matrix(df: pd.DataFrame, suffix: str = "") -> Optional[np.ndarray]:
    """Extract probability matrix p0..pK with optional suffix filter (e.g., '_D')."""
    if suffix:
        pattern = re.compile(rf"^p(\d+){re.escape(suffix)}$")
    else:
        pattern = re.compile(r"^p(\d+)$")

    cols = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            idx = int(m.group(1))
            cols.append((idx, c))

    if not cols:
        return None

    cols = sorted(cols, key=lambda x: x[0])
    probs = df[[c for _, c in cols]].values.astype(np.float32)
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    probs = probs / row_sum
    return probs


def _save_predictions_csv(out_csv: str, probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                          image_paths: Optional[List[str]], manifest_df: Optional[pd.DataFrame]):
    df = pd.DataFrame({
        'true_label': y_true.astype(int),
        'pred_label': y_pred.astype(int),
        'confidence': probs.max(axis=1)
    })
    # per-class probs p0..pK
    for k in range(probs.shape[1]):
        df[f'p{k}'] = probs[:, k]

    # image_path bila tersedia (via meta) → else coba merge urutan dari manifest
    if image_paths and len(image_paths) == len(df):
        df.insert(0, 'image_path', image_paths)
    elif manifest_df is not None and 'image_path' in manifest_df.columns and len(manifest_df) >= len(df):
        df.insert(0, 'image_path', manifest_df['image_path'][:len(df)].values)

    df.to_csv(out_csv, index=False)
    return df


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray,
                  probs: Optional[np.ndarray], labels: List[int]) -> Dict:
    out = dict(
        accuracy=metric_accuracy(y_true, y_pred),
        balanced_accuracy=metric_balanced_accuracy(y_true, y_pred),
        f1_macro=metric_f1_macro(y_true, y_pred),
        confusion_matrix=metric_confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        per_class=per_class_report(y_true, y_pred, labels=labels)
    )
    if probs is not None:
        out['ece'] = metric_ece(probs, y_true, n_bins=15)
        out['brier'] = metric_brier_multiclass(probs, y_true)
        out['entropy'] = metric_entropy_from_probs(probs)
        out['reliability_bins'] = reliability_bins(probs, y_true, n_bins=15)
    return out


def _dualhead_metrics_block(scores_df: pd.DataFrame, y_true: np.ndarray, labels_list_full: List[int],
                           pred_col: str, prob_suffix: str) -> Optional[Dict]:
    if pred_col not in scores_df.columns:
        return None
    probs = _prob_matrix(scores_df, prob_suffix)
    y_pred = scores_df[pred_col].astype(int).values
    return _metrics_dict(y_true, y_pred, probs, labels=labels_list_full)


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def evaluate_model(
    model_config_path: str,
    base_config_path: Optional[str] = None,
    split: str = 'val',
    save_preds_csv: bool = True
) -> Dict:
    """
    Evaluasi langsung dari model (checkpoint lokal) di split: 'train' | 'val'.
    - Menghasilkan predictions CSV (dengan p0..pK) dan report JSON.

    Return: dict ringkas berisi jalur keluaran + metrik utama.
    """
    # Load configs
    with open(model_config_path, 'r') as f:
        mcfg = yaml.safe_load(f)

    if base_config_path is None:
        base_guess = Path(model_config_path).parent / 'base_config_coweps.yaml'
        if not base_guess.exists():
            raise FileNotFoundError("Base config tidak ditemukan (base_config_coweps.yaml).")
        base_config_path = str(base_guess)

    with open(base_config_path, 'r') as f:
        bcfg = yaml.safe_load(f)

    # Merge paths ke model config untuk factory
    mcfg['paths'] = bcfg['paths']

    # Resolve dataset split
    processed_dir = bcfg['paths']['processed_dir']
    manifests = bcfg.get('manifests', {}) or {}
    if split == 'train':
        manifest_csv = manifests.get('train') or os.path.join(processed_dir, 'gold_standard_train.csv')
    else:
        manifest_csv = manifests.get('validate') or os.path.join(processed_dir, 'gold_standard_validate.csv')
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    # Data & loader
    dataset = DRDataset(manifest_csv, mcfg, mode='val')
    loader = torch.utils.data.DataLoader(dataset, batch_size=mcfg.get('inference', {}).get('batch_size', 32),
                                         shuffle=False, num_workers=4, pin_memory=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model_from_config(mcfg, logger=None).to(device).eval()

    # Forward
    logits_list, labels_list = [], []
    image_paths = []
    manifest_df = pd.read_csv(manifest_csv)

    for i, batch in enumerate(tqdm(loader, desc=f"Forward ({split})")):
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels, meta = batch
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(images)
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())

        # DRDataset saat ini tidak mengembalikan meta; fallback: ambil dari manifest urutan
        # agar konsisten, kita gunakan manifest_df nantinya.

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).numpy().astype(int)
    probs = F.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    # Labels list
    num_classes = probs.shape[1]
    labels_list_full = list(range(num_classes))

    # Report dir
    report_dir = _resolve_report_dir(bcfg)
    stem = Path(model_config_path).stem
    out_csv = os.path.join(report_dir, f"predictions_{stem}_{split}.csv")
    out_json = os.path.join(report_dir, f"report_{stem}_{split}.json")

    # Save predictions
    if save_preds_csv:
        preds_df = _save_predictions_csv(out_csv, probs, labels, preds, image_paths, manifest_df)
    else:
        preds_df = None

    # Metrics (overall)
    report = dict()
    report['overall'] = _metrics_dict(labels, preds, probs, labels=labels_list_full)

    # Grouped by source (jika tersedia di manifest)
    if 'source' in manifest_df.columns:
        groups = manifest_df['source'].values[:len(labels)]
        gmet = grouped_metrics(labels, preds, groups, probs=probs, labels=labels_list_full)
        # cast dataclass → dict
        report['by_source'] = {k: vars(v) for k, v in gmet.items()}

    # Simpan JSON
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)

    return {
        'success': True,
        'report_json': out_json,
        'predictions_csv': out_csv if save_preds_csv else None,
        'overall': report['overall']
    }


def evaluate_from_scores(
    scores_csv: str,
    base_config_path: str,
    manifest_csv: Optional[str] = None
) -> Dict:
    """
    Evaluasi dari hasil CSV skor (mis. Fase-3 inference):
    - DIBUTUHKAN kolom true label → bila tak ada, wajib berikan manifest_csv
      agar bisa merge 'label'/'weak_label_class'/'grade'.
    - ECE/Brier akan dihitung hanya jika kolom p0..pK tersedia (probabilitas).
    """
    with open(base_config_path, 'r') as f:
        bcfg = yaml.safe_load(f)

    report_dir = _resolve_report_dir(bcfg)
    stem = Path(scores_csv).stem
    out_json = os.path.join(report_dir, f"report_from_scores_{stem}.json")

    if not os.path.exists(scores_csv):
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv}")
    scores_df = pd.read_csv(scores_csv)

    # Tentukan kolom pred (wajib)
    pred_col = None
    for c in ['pred_label', 'Pred_Class', 'pred']:
        if c in scores_df.columns:
            pred_col = c
            break
    if pred_col is None:
        raise ValueError("Scores CSV tidak memiliki kolom prediksi ('Pred_Class' atau 'pred_label').")

    # Ambil true label: dari scores atau merge manifest
    label_col = None
    for c in ['true_label', 'label', 'weak_label_class', 'grade']:
        if c in scores_df.columns:
            label_col = c
            break
    if label_col is None:
        if manifest_csv is None:
            raise ValueError("True label tidak ditemukan di scores CSV. Berikan manifest_csv untuk merge.")
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
        man = pd.read_csv(manifest_csv)
        lbl_col = _infer_label_col(man)
        # merge by image_path jika tersedia; bila tidak, asumsikan urutan sama panjang
        if 'image_path' in scores_df.columns and 'image_path' in man.columns:
            scores_df = scores_df.merge(man[['image_path', lbl_col, 'source']] if 'source' in man.columns else man[['image_path', lbl_col]],
                                        on='image_path', how='left', suffixes=('', '_m'))
            label_col = lbl_col
        else:
            # fallback: copy sequence
            if len(man) < len(scores_df):
                raise ValueError("Ukuran manifest < skor dan tidak ada 'image_path' untuk merge.")
            scores_df[lbl_col] = man[lbl_col].values[:len(scores_df)]
            if 'source' in man.columns:
                scores_df['source'] = man['source'].values[:len(scores_df)]
            label_col = lbl_col

    y_true = scores_df[label_col].astype(int).values
    y_pred = scores_df[pred_col].astype(int).values

    # Pull probs p0..pK bila ada
    prob_cols = [c for c in scores_df.columns if c.startswith('p') and c[1:].isdigit()]
    probs = None
    if prob_cols:
        probs = scores_df[prob_cols].values.astype(np.float32)
        # normalisasi safeguard
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        probs = probs / row_sum

    # Labels list
    num_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    labels_list_full = list(range(num_classes))

    # Metrics (overall)
    report = dict()
    report['overall'] = _metrics_dict(y_true, y_pred, probs, labels=labels_list_full)

    # Dual-head extras (per-teacher + pre-calibration + disagreement stats) when available
    dual = {}
    m_teacher_d = _dualhead_metrics_block(scores_df, y_true, labels_list_full, pred_col="Pred_Class_D", prob_suffix="_D")
    if m_teacher_d:
        dual['teacher_D'] = m_teacher_d

    m_teacher_r = _dualhead_metrics_block(scores_df, y_true, labels_list_full, pred_col="Pred_Class_R", prob_suffix="_R")
    if m_teacher_r:
        dual['teacher_R'] = m_teacher_r

    m_before = _dualhead_metrics_block(scores_df, y_true, labels_list_full, pred_col="Pred_Class_before", prob_suffix="_before")
    if m_before:
        dual['before_calibration'] = m_before

    # Disagreement / confidence summaries
    disag = {}
    if 'js_div' in scores_df.columns:
        js_vals = pd.to_numeric(scores_df['js_div'], errors='coerce').dropna()
        if len(js_vals):
            disag['js_div'] = {
                'mean': float(js_vals.mean()),
                'median': float(js_vals.median()),
                'p90': float(js_vals.quantile(0.90)),
                'p95': float(js_vals.quantile(0.95)),
            }
    if 'agreement' in scores_df.columns:
        ag_vals = pd.to_numeric(scores_df['agreement'], errors='coerce').dropna()
        if len(ag_vals):
            disag['agreement'] = {
                'mean': float(ag_vals.mean()),
                'median': float(ag_vals.median()),
                'p10': float(ag_vals.quantile(0.10)),
            }
    if 'top1_agree' in scores_df.columns:
        ta = pd.to_numeric(scores_df['top1_agree'], errors='coerce').dropna()
        if len(ta):
            disag['top1_agree_rate'] = float((ta > 0.5).mean())
    if 'kl_DR' in scores_df.columns or 'kl_RD' in scores_df.columns:
        kl_block = {}
        if 'kl_DR' in scores_df.columns:
            kl_dr = pd.to_numeric(scores_df['kl_DR'], errors='coerce').dropna()
            if len(kl_dr):
                kl_block['kl_DR_mean'] = float(kl_dr.mean())
        if 'kl_RD' in scores_df.columns:
            kl_rd = pd.to_numeric(scores_df['kl_RD'], errors='coerce').dropna()
            if len(kl_rd):
                kl_block['kl_RD_mean'] = float(kl_rd.mean())
        if kl_block:
            disag['kl'] = kl_block
    if disag:
        dual['disagreement'] = disag

    if dual:
        report['dualhead'] = dual

    # Grouped by source (opsional)
    if 'source' in scores_df.columns:
        groups = scores_df['source'].astype(str).values
        gmet = grouped_metrics(y_true, y_pred, groups, probs=probs, labels=labels_list_full)
        report['by_source'] = {k: vars(v) for k, v in gmet.items()}

    # Save JSON
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)

    return {
        'success': True,
        'report_json': out_json,
        'overall': report['overall']
    }
