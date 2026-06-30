"""
CoWePS v2.5 - Metrics Utilities
Modular helpers untuk menghitung metrik klasifikasi (ID/OOD aware) dan kalibrasi.

Fitur:
- Balanced Accuracy, F1-macro, Accuracy
- Confusion Matrix (numpy)
- ECE (Expected Calibration Error) dengan reliability bins
- Brier Score (multiclass)
- Entropy (dari probabilitas)
- Laporan per-kelas dan per-kelompok (mis. per 'source')

Catatan:
- Untuk ECE/Brier diperlukan probabilitas per-kelas (tensor/ndarray shape [N, C]).
- Bila hanya tersedia pred dan confidence tunggal, ECE/Brier tidak akurat → gunakan evaluator
  berbasis model/dataloader agar bisa menghitung logits→probs penuh.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, accuracy_score, classification_report


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def metric_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def metric_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average='macro'))


def metric_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def metric_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[int]] = None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def metric_entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> float:
    """
    Mean entropy over samples. probs: (N, C), rows sum to 1.
    """
    p = np.clip(probs, eps, 1.0)
    ent = -(p * np.log(p)).sum(axis=1)
    return float(ent.mean())


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def metric_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (Guo et al., 2017) – multiclass (top-1).
    probs: (N, C), y_true: (N,)
    """
    assert probs.ndim == 2, "probs harus (N, C)"
    conf = probs.max(axis=1)
    y_pred = probs.argmax(axis=1)
    acc = (y_pred == y_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if not np.any(in_bin):
            continue
        prop = in_bin.mean()
        acc_bin = acc[in_bin].mean()
        conf_bin = conf[in_bin].mean()
        ece += np.abs(conf_bin - acc_bin) * prop
    return float(ece)


def metric_brier_multiclass(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Brier score multiclass: mean( sum_k (p_k - y_k)^2 ).
    probs: (N, C), y_true: (N,)
    """
    n, c = probs.shape
    y_onehot = np.zeros((n, c), dtype=np.float32)
    y_onehot[np.arange(n), y_true.astype(int)] = 1.0
    diff = probs - y_onehot
    return float((diff ** 2).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[int]] = None) -> Dict:
    """
    Sklearn classification_report → dict terstruktur.
    """
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    # pastikan kunci numerik di-cast ke int
    out = {}
    for k, v in report.items():
        try:
            ki = int(k)
            out[ki] = v
        except Exception:
            out[k] = v
    return out


def reliability_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> Dict:
    """
    Kembalikan statistik per bin untuk plotting (tanpa menggambar).
    Keys: 'bin_lower', 'bin_upper', 'count', 'accuracy', 'avg_confidence'
    """
    conf = probs.max(axis=1)
    y_pred = probs.argmax(axis=1)
    acc = (y_pred == y_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    stats = {
        'bin_lower': [], 'bin_upper': [],
        'count': [], 'accuracy': [], 'avg_confidence': []
    }
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        cnt = int(in_bin.sum())
        stats['bin_lower'].append(float(lo))
        stats['bin_upper'].append(float(hi))
        stats['count'].append(cnt)
        if cnt > 0:
            stats['accuracy'].append(float(acc[in_bin].mean()))
            stats['avg_confidence'].append(float(conf[in_bin].mean()))
        else:
            stats['accuracy'].append(0.0)
            stats['avg_confidence'].append(0.0)
    return stats


# ---------------------------------------------------------------------------
# Grouped metrics
# ---------------------------------------------------------------------------

@dataclass
class GroupMetrics:
    count: int
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    ece: Optional[float] = None
    brier: Optional[float] = None
    entropy: Optional[float] = None


def grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    probs: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None,
    n_bins_ece: int = 15
) -> Dict[str, GroupMetrics]:
    """
    Hitung metrik per kelompok (mis. per 'source').
    Bila probs tersedia → ECE/Brier/Entropy dihitung juga.
    """
    uniq = np.unique(groups.astype(str))
    out: Dict[str, GroupMetrics] = {}
    for g in uniq:
        idx = (groups.astype(str) == g)
        yt = y_true[idx]
        yp = y_pred[idx]
        cnt = int(idx.sum())

        acc = metric_accuracy(yt, yp)
        ba = metric_balanced_accuracy(yt, yp)
        f1m = metric_f1_macro(yt, yp)

        ece = brier = ent = None
        if probs is not None:
            pr = probs[idx]
            ece = metric_ece(pr, yt, n_bins=n_bins_ece)
            brier = metric_brier_multiclass(pr, yt)
            ent = metric_entropy_from_probs(pr)

        out[str(g)] = GroupMetrics(
            count=cnt, accuracy=acc, balanced_accuracy=ba, f1_macro=f1m,
            ece=ece, brier=brier, entropy=ent
        )
    return out
