#!/usr/bin/env python
"""
CoWePS v2.5-ready - Generic Model Trainer untuk "Ensemble A"
Dengan kebijakan checkpoint berbasis metrik (BA/F1/ECE/Loss) + save_top_k.

Perubahan utama:
- Checkpointing via YAML: checkpointing.save_by | save_top_k | also_save_epochs | monitor_split
- Best checkpoint diambil sesuai metrik monitor (bukan lagi wajib val_loss)
- Temperature scaling dilakukan pada checkpoint terbaik sesuai monitor
- Tetap kompatibel: jika block 'checkpointing' tidak ada -> fallback ke best-by-loss

Author: CoWePS Implementation Team
"""

import os
import sys
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime
import shutil
from contextlib import contextmanager
import warnings
from typing import Optional, Dict, Tuple, List

import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score

warnings.filterwarnings('ignore')

# Tambahkan project root ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import create_model_from_config
from src.data.data_processing import DRDataset


# =============================================================================
# Output Management (MANDATORY)
# =============================================================================
def _default_run_id(seed: int) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{ts}_seed{int(seed)}"


def _make_unique_run_dir(output_root: Path, config_stem: str, run_id: str) -> Tuple[Path, str]:
    base = output_root / config_stem / run_id
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base, run_id

    # Never overwrite old runs; if collision, add suffix _v2, _v3, ...
    for i in range(2, 100):
        rid = f"{run_id}_v{i}"
        cand = output_root / config_stem / rid
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand, rid

    raise RuntimeError(f"Unable to allocate unique run dir for {config_stem}/{run_id}")


class _Tee:
    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file = file_handle

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()


@contextmanager
def _tee_stdout_stderr(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, f)
        sys.stderr = _Tee(old_err, f)
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


def _apply_output_policy(config: dict, config_path: str, run_id: Optional[str], output_root: str) -> Tuple[dict, Path, str]:
    """Mutate config so all outputs go into models/<config_stem>/<run_id>/.

    Returns: (config, run_dir, final_run_id)
    """
    cfg_path = Path(config_path)
    config_stem = cfg_path.stem
    seed = int(config.get('random_seed', 0) or 0)

    rid = str(run_id) if run_id else _default_run_id(seed)
    out_root = (Path(output_root) if Path(output_root).is_absolute() else (PROJECT_ROOT / output_root)).resolve()
    run_dir, rid = _make_unique_run_dir(out_root, config_stem, rid)

    # Force all important paths under run_dir
    config.setdefault('paths', {})
    config['paths']['checkpoints_dir'] = str(run_dir)

    config.setdefault('model', {})
    config['model']['output_dir'] = str(run_dir)
    config['model']['calibration_path'] = str(run_dir / 'T_optimal.pth')

    config.setdefault('logging', {})
    config['logging']['log_dir'] = str(run_dir / 'logs')

    config.setdefault('data', {})
    config['data']['scores_output_path'] = str(run_dir / 'scores.csv')

    # Persist a copy of the exact config used (for reproducibility)
    used_cfg_path = run_dir / 'config_used.yaml'
    used_cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

    # Minimal artifacts index (expanded by pipeline)
    idx_path = run_dir / 'artifacts_index.json'
    if not idx_path.exists():
        idx = {
            'created_at': datetime.now().isoformat(),
            'policy': 'models/<config_stem>/<run_id>/',
            'config_source': str(cfg_path.resolve()),
            'config_used': str(used_cfg_path.resolve()),
            'run_dir': str(run_dir.resolve()),
            'mappings': [
                {'kind': 'config_used', 'src': str(cfg_path.resolve()), 'dst': str(used_cfg_path.resolve())},
            ],
        }
        idx_path.write_text(json.dumps(idx, indent=2), encoding='utf-8')

    return config, run_dir, rid


# =============================================================================
# Temperature Calibration (ECE + T-scaling)
# =============================================================================
class ECELoss(nn.Module):
    """Expected Calibration Error (ECE) – Guo et al., 2017."""
    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=1)
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(0.0, 1.0, steps=self.n_bins + 1, device=logits.device)
        ece = torch.zeros(1, device=logits.device)

        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (confidences > lo) & (confidences <= hi)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                acc_in_bin = accuracies[in_bin].float().mean()
                conf_in_bin = confidences[in_bin].mean()
                ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
        return ece


@torch.no_grad()
def _collect_logits_labels(model: nn.Module, loader: DataLoader, device: torch.device, logger: Optional[logging.Logger]):
    if logger:
        logger.info("Collecting validation logits...")
    model.eval()
    logits_list, labels_list = [], []
    for batch in tqdm(loader, desc="Collecting logits", disable=logger is None):
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels, _ = batch
        images = images.to(device, non_blocking=True)
        logits = model(images)
        logits_list.append(logits.cpu())
        labels_list.append(labels)
    logits_all = torch.cat(logits_list).to(device)
    labels_all = torch.cat(labels_list).to(device)
    if logger:
        logger.info(f"Total validation samples: {len(labels_all)}")
    return logits_all, labels_all


def calibrate_temperature(model: nn.Module, val_loader: DataLoader, device: torch.device, logger: Optional[logging.Logger]):
    """Cari T optimal yang meminimalkan NLL; laporkan ECE sebelum/sesudah."""
    if logger:
        logger.info("\n" + "="*80)
        logger.info("TEMPERATURE CALIBRATION (T-scaling)")
        logger.info("="*80)

    logits_all, labels_all = _collect_logits_labels(model, val_loader, device, logger)

    log_t = nn.Parameter(torch.zeros(1, device=device))  # log(1)=0
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = ECELoss()
    optimizer = optim.LBFGS([log_t], lr=0.01, max_iter=50)

    def _T():
        # clamp untuk mencegah T ekstrem
        return torch.exp(log_t).clamp(1e-3, 100.0)

    def eval_func():
        optimizer.zero_grad(set_to_none=True)
        loss = nll_criterion(logits_all / _T(), labels_all)
        loss.backward()
        return loss

    optimizer.step(eval_func)

    T_opt = float(_T().item())

    with torch.no_grad():
        # BEFORE: softmax(logits)
        # AFTER : softmax(logits / T_opt)
        ece_before = float(ece_criterion(logits_all, labels_all, temperature=1.0).item())
        ece_after  = float(ece_criterion(logits_all, labels_all, temperature=T_opt).item())

        nll_before = _compute_nll(logits_all, labels_all, temperature=1.0)
        nll_after  = _compute_nll(logits_all, labels_all, temperature=T_opt)

        num_classes = int(getattr(model, 'num_classes', 5))
        brier_before = _compute_brier_multiclass_from_logits(logits_all, labels_all, num_classes=num_classes, temperature=1.0)
        brier_after  = _compute_brier_multiclass_from_logits(logits_all, labels_all, num_classes=num_classes, temperature=T_opt)

    if logger:
        logger.info("\n" + "="*80)
        logger.info("CALIBRATION RESULTS")
        logger.info("="*80)
        logger.info(f"Optimal Temperature: {T_opt:.4f}")
        logger.info(f"ECE before: {ece_before:.4f}")
        logger.info(f"ECE after : {ece_after:.4f}")
        logger.info(f"ECE improvement: {(ece_before - ece_after):.4f}")
        logger.info(f"NLL before: {nll_before:.4f}")
        logger.info(f"NLL after : {nll_after:.4f}")
        logger.info(f"Brier before: {brier_before:.4f}")
        logger.info(f"Brier after : {brier_after:.4f}")

    return T_opt


# =============================================================================
# Util: CoWePS metrics per-epoch
# =============================================================================
def _entropy_from_logits(logits: torch.Tensor) -> float:
    """Mean entropy over samples (natural log)."""
    logits = logits.to(dtype=torch.float32)
    probs = torch.softmax(logits, dim=1)
    ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)
    return float(ent.mean().item())


def _compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    target_device = labels.device
    logits = logits.to(device=target_device, dtype=torch.float32, non_blocking=True)
    labels = labels.to(device=target_device, dtype=torch.long, non_blocking=True)
    ece = ECELoss(n_bins=n_bins)
    with torch.no_grad():
        val = float(ece(logits, labels, temperature=1.0).item())
    return val


def _compute_nll(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> float:
    """Negative log-likelihood (categorical) from logits/temperature."""
    logits = logits.to(dtype=torch.float32)
    labels = labels.to(dtype=torch.long)
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        t = 1.0
    with torch.no_grad():
        return float(F.cross_entropy(logits / t, labels, reduction='mean').item())


def _compute_brier_multiclass_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    temperature: float = 1.0,
) -> float:
    """Brier score multiclass: mean(sum_k (p_k - y_k)^2)."""
    logits = logits.to(dtype=torch.float32)
    labels = labels.to(dtype=torch.long)
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        t = 1.0
    with torch.no_grad():
        probs = torch.softmax(logits / t, dim=1)
        y = torch.zeros_like(probs)
        y.scatter_(1, labels.view(-1, 1), 1.0)
        diff = probs - y
        return float((diff.pow(2).sum(dim=1)).mean().item())


def _compute_ece_with_temperature(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15, temperature: float = 1.0) -> float:
    target_device = labels.device
    logits = logits.to(device=target_device, dtype=torch.float32, non_blocking=True)
    labels = labels.to(device=target_device, dtype=torch.long, non_blocking=True)
    ece = ECELoss(n_bins=n_bins)
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        t = 1.0
    with torch.no_grad():
        val = float(ece(logits, labels, temperature=t).item())
    return val


# =============================================================================
# Robust Loss family (native, tanpa dependensi eksternal)
# =============================================================================
class BiTemperedLoss(nn.Module):
    """Bi-Tempered Logistic Loss (Amunategui et al.). Implementasi ringkas."""
    def __init__(self, t1: float = 0.7, t2: float = 1.2, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.ls = float(label_smoothing)
        self.reduction = reduction

    def _log_t(self, x, t):
        if abs(t - 1.0) < 1e-7:
            return torch.log(x)
        return (x.pow(1 - t) - 1) / (1 - t)

    def _exp_t(self, x, t):
        if abs(t - 1.0) < 1e-7:
            return torch.exp(x)
        return torch.relu(1 + (1 - t) * x).pow(1 / (1 - t))

    def forward(self, logits, target):
        """Tempered CE with correction term to keep loss non-negative.

        Definitions (consistent with user requirement):
        - probabilities from tempered softmax with t2
        - loss uses t1-tempered log and adds the (2-t1) correction term.
        """
        num_classes = logits.size(1)
        y = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1.0)
        if self.ls > 0.0:
            y = (1 - self.ls) * y + self.ls / num_classes

        # tempered softmax (numerically stabilized)
        x = logits - logits.detach().amax(dim=1, keepdim=True)
        probs = self._exp_t(x, self.t2)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # Core term: - sum y * log_t1(p)
        log_p = self._log_t(probs.clamp_min(1e-12), self.t1)
        core = -(y * log_p).sum(dim=1)

        # Correction term (vanishes when t1 -> 1): (sum y^(2-t1) - sum p^(2-t1)) / (2-t1)
        denom = (2.0 - self.t1)
        if abs(denom) < 1e-7:
            corr = 0.0
        else:
            y_pow = y.clamp_min(1e-12).pow(2.0 - self.t1).sum(dim=1)
            p_pow = probs.clamp_min(1e-12).pow(2.0 - self.t1).sum(dim=1)
            corr = (y_pow - p_pow) / denom

        loss = core + corr

        # Extra guard for rare numerical issues
        loss = loss.clamp_min(0.0)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SymmetricCrossEntropy(nn.Module):
    """SCE: alpha * CE + beta * RCE (reverse CE)."""
    def __init__(self, alpha: float = 0.1, beta: float = 1.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=1).clamp_min(1e-12)
        log_probs = torch.log(probs)
        rce = -torch.mean(log_probs.gather(1, target.view(-1, 1)))
        return self.alpha * ce + self.beta * rce


class GeneralizedCrossEntropy(nn.Module):
    """GCE (Zhang & Sabuncu): L_q = (1 - p_y^q) / q, 0<q<=1."""
    def __init__(self, q: float = 0.7, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.q = float(q)
        self.weight = weight

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1).clamp_min(1e-12)
        p_y = probs.gather(1, target.view(-1, 1)).squeeze(1)
        loss = (1 - p_y.pow(self.q)) / self.q
        if self.weight is not None:
            w = self.weight[target]
            loss = loss * w
        return loss.mean()


def build_robust_criterion(config: Dict, class_weights: Optional[torch.Tensor], num_classes: int = 5):
    rl = (config.get('robust_loss') or {})
    rl_type = str(rl.get('type', 'cross_entropy')).lower()

    tr = config.get('training', {}) or {}
    ls = float(tr.get('label_smoothing', rl.get('params', {}).get('label_smoothing', 0.0)))

    if rl_type in ('cross_entropy', 'ce'):
        return nn.CrossEntropyLoss(weight=(class_weights if class_weights is not None else None),
                                   label_smoothing=ls)

    if rl_type in ('bi_tempered', 'bitempered', 'bi-tempered'):
        p = rl.get('params', {}) or {}
        t1 = float(p.get('t1', 0.7))
        t2 = float(p.get('t2', 1.2))
        ls_local = float(p.get('label_smoothing', ls))
        return BiTemperedLoss(t1=t1, t2=t2, label_smoothing=ls_local)

    if rl_type in ('symmetric_ce', 'symmetricce', 'sce'):
        p = rl.get('params', {}) or {}
        alpha = float(p.get('alpha', 0.1))
        beta  = float(p.get('beta', 1.0))
        return SymmetricCrossEntropy(alpha=alpha, beta=beta,
                                     weight=(class_weights if class_weights is not None else None),
                                     label_smoothing=ls)

    if rl_type in ('gce', 'generalized_ce', 'generalized_cross_entropy'):
        p = rl.get('params', {}) or {}
        q = float(p.get('q', 0.7))
        return GeneralizedCrossEntropy(q=q, weight=(class_weights if class_weights is not None else None))

    return nn.CrossEntropyLoss(weight=(class_weights if class_weights is not None else None),
                               label_smoothing=ls)


# =============================================================================
# TRAINER
# =============================================================================
class TopKKeeper:
    """
    Menjaga top-K checkpoint terbaik berdasarkan skor monitor.
    Simpan sebagai min-heap (untuk kasus maximize). Untuk metric 'loss'/'ece' kita invert tandanya.
    """
    def __init__(self, k: int, maximize: bool, out_dir: str, base_name: str):
        self.k = max(1, int(k))
        self.maximize = maximize
        self.heap: List[Tuple[float, str]] = []
        self.out_dir = out_dir
        self.base_name = Path(base_name).stem  # tanpa .pth

    def _key(self, score: float) -> float:
        # Min-heap menaruh nilai terkecil di root (heap[0]).
        # Elemen di root adalah kandidat yang akan DIBUANG jika ada yang lebih baik.
        
        # KASUS MAXIMIZE (misal Accuracy):
        # Kita ingin membuang Accuracy TERKECIL.
        # Min-heap sudah menaruh nilai terkecil di root.
        # Jadi: Simpan score POSITIF apa adanya.
        
        # KASUS MINIMIZE (misal Loss):
        # Kita ingin membuang Loss TERBESAR.
        # Agar Loss TERBESAR menjadi nilai "terkecil" (di root min-heap), kita NEGATIFKAN.
        # Contoh: Loss 0.9 (buruk) vs 0.1 (baik).
        # Key: -0.9 vs -0.1.
        # Min-heap: -0.9 < -0.1. Jadi -0.9 (Loss 0.9) ada di root -> Siap dibuang.
        
        return score if self.maximize else -score

    def consider(self, epoch: int, score: float, state_dict: dict, tag: str) -> Optional[str]:
        """
        Pertimbangkan untuk menyimpan checkpoint. Kembalikan path jika disimpan.
        """
        key = self._key(score)
        ckpt_name = f"{self.base_name}_ep{epoch:02d}_{tag}{score:.4f}.pth"
        ckpt_path = os.path.join(self.out_dir, ckpt_name)

        # Be robust: ensure output directory exists before saving.
        # This prevents runtime crashes if the run_dir was not created or was removed.
        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except Exception:
            pass
        
        if len(self.heap) < self.k:
            torch.save(state_dict, ckpt_path)
            heapq.heappush(self.heap, (key, ckpt_path))
            return ckpt_path

        # Lihat kandidat terburuk (elemen yang siap dibuang)
        worst_key, worst_path = self.heap[0]

        # Jika kandidat baru lebih baik dari yang terburuk
        # Karena min-heap, "lebih baik" berarti nilainya LEBIH BESAR dari root.
        # (Ingat: Root adalah nilai terkecil/terburuk).
        better = key > worst_key
        if better:
            # buang terburuk
            try:
                if os.path.exists(worst_path):
                    os.remove(worst_path)
            except Exception:
                pass
            # simpan baru
            try:
                os.makedirs(self.out_dir, exist_ok=True)
            except Exception:
                pass
            torch.save(state_dict, ckpt_path)
            heapq.heapreplace(self.heap, (key, ckpt_path))
            return ckpt_path
        return None

    def best_path(self) -> Optional[str]:
        if not self.heap:
            return None
        # Cari path terbaik.
        # Heap menyimpan (key, path).
        # Key semakin besar = semakin baik (karena logika _key di atas).
        # Jadi kita cari elemen dengan key terbesar.
        best_item = max(self.heap, key=lambda x: x[0])
        return best_item[1]


class Trainer:
    """Generic trainer untuk Ensemble A models (v2.5-ready, metric-aware checkpointing)"""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger

        tr_cfg = config['training']
        self.num_epochs = int(tr_cfg['num_epochs'])
        self.use_amp = bool(tr_cfg.get('use_amp', True))
        self.grad_accum_steps = int(tr_cfg.get('gradient_accumulation_steps', 1))
        self.max_grad_norm = float(tr_cfg.get('max_grad_norm', 1.0))

        # === Freeze Backbone (head-only warmup) ===
        fb_cfg = tr_cfg.get('freeze_backbone') or {}
        self.freeze_epochs = int(fb_cfg.get('epochs', fb_cfg.get('freeze_epochs', 0)))
        self.freeze_enabled = bool(fb_cfg.get('enabled', self.freeze_epochs > 0))
        self.head_keywords = tuple(fb_cfg.get('head_keywords', ('head', 'classifier', 'fc', 'last_linear')))
        self.head_lr_mult = float(fb_cfg.get('head_lr_mult', 1.0))
        self.backbone_lr_mult = float(fb_cfg.get('backbone_lr_mult', 1.0))

        def _collect_head_modules(model: nn.Module) -> List[nn.Module]:
            mods: List[nn.Module] = []
            if hasattr(model, 'get_classifier'):
                try:
                    clf = model.get_classifier()
                    if isinstance(clf, nn.Module):
                        mods.append(clf)
                except Exception:
                    pass
            for attr in ('head', 'classifier'):
                try:
                    m = getattr(model, attr)
                    if isinstance(m, nn.Module):
                        mods.append(m)
                except Exception:
                    pass
            return mods

        def _partition_params(model: nn.Module, keywords: Tuple[str, ...]):
            head_modules = _collect_head_modules(model)
            head_param_ids = set()
            for m in head_modules:
                for p in m.parameters(recurse=True):
                    head_param_ids.add(id(p))

            named_params = list(model.named_parameters())
            if head_param_ids:
                head_pairs = [(n, p) for n, p in named_params if id(p) in head_param_ids]
                backbone_pairs = [(n, p) for n, p in named_params if id(p) not in head_param_ids]
            else:
                head_pairs = [(n, p) for n, p in named_params if any(k in n for k in keywords)]
                backbone_pairs = [(n, p) for n, p in named_params if not any(k in n for k in keywords)]

            head_params = [p for _, p in head_pairs]
            backbone_params = [p for _, p in backbone_pairs]
            head_names = [n for n, _ in head_pairs]
            backbone_names = [n for n, _ in backbone_pairs]
            head_ids = {id(p) for p in head_params}
            return head_params, backbone_params, head_names, backbone_names, head_ids

        self.head_params, self.backbone_params, self.head_param_names, self.backbone_param_names, self.head_param_ids = _partition_params(self.model, self.head_keywords)
        self.backbone_frozen = False

        # === Optimizer (pakai LR efektif bila tersedia) ===
        base_lr = float(tr_cfg.get('learning_rate', 1e-4))
        eff_lr  = float(tr_cfg.get('learning_rate_effective', base_lr))
        wd = float(tr_cfg.get('weight_decay', 0.0))
        opt_name = tr_cfg.get('optimizer', 'AdamW')
        lr_backbone = eff_lr * self.backbone_lr_mult
        lr_head = eff_lr * self.head_lr_mult
        use_param_groups = (
            (self.freeze_enabled or self.head_lr_mult != 1.0 or self.backbone_lr_mult != 1.0)
            and len(self.head_params) > 0 and len(self.backbone_params) > 0
        )

        if opt_name == 'AdamW':
            if use_param_groups:
                param_groups = [
                    {'params': self.backbone_params, 'lr': lr_backbone, 'tag': 'backbone'},
                    {'params': self.head_params, 'lr': lr_head, 'tag': 'head'},
                ]
                self.optimizer = optim.AdamW(param_groups, lr=eff_lr, weight_decay=wd)
            else:
                self.optimizer = optim.AdamW(self.model.parameters(), lr=eff_lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        if self.logger:
            freeze_msg = (
                f"Freeze backbone: enabled={self.freeze_enabled} | epochs={self.freeze_epochs} | "
                f"head_lr_mult={self.head_lr_mult} | backbone_lr_mult={self.backbone_lr_mult} | "
                f"head_keywords={self.head_keywords} | head_params={len(self.head_params)} | "
                f"backbone_params={len(self.backbone_params)} | use_param_groups={use_param_groups}"
            )
            self.logger.info(freeze_msg)
            self.logger.info(
                f"Param elems: head={sum(p.numel() for p in self.head_params):,} | "
                f"backbone={sum(p.numel() for p in self.backbone_params):,} | total={sum(p.numel() for p in self.model.parameters()):,}"
            )
            first_trainable = [n for n, p in zip(self.head_param_names + self.backbone_param_names, self.head_params + self.backbone_params) if p.requires_grad][:20]
            if first_trainable:
                self.logger.info(f"Trainable param names (first 20): {first_trainable}")
            if use_param_groups:
                self.logger.info(f"Param groups LR: backbone={lr_backbone:g}, head={lr_head:g}, base={eff_lr:g}")
            elif self.freeze_enabled:
                self.logger.info("Param groups disabled (fallback to single LR) but freeze schedule remains active via requires_grad toggling.")

        # Scheduler (opsional) + optional warmup
        sch_name = tr_cfg.get('scheduler', 'CosineAnnealingLR')
        self.warmup_epochs = int(tr_cfg.get('warmup_epochs', 0))

        if sch_name == 'CosineAnnealingLR':
            sch_params = tr_cfg.get('scheduler_params', {})
            t_max_main = int(sch_params.get('T_max', self.num_epochs))
            if self.warmup_epochs > 0:
                t_max_main = max(1, self.num_epochs - self.warmup_epochs)

            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max_main,
                eta_min=float(sch_params.get('eta_min', 1e-6))
            )

            if self.warmup_epochs > 0:
                def warmup_lambda(epoch: int) -> float:
                    if epoch < self.warmup_epochs:
                        return float(epoch + 1) / float(self.warmup_epochs)
                    return 1.0

                warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[self.warmup_epochs]
                )
            else:
                self.scheduler = main_scheduler
        else:
            self.scheduler = None

        # === Robust Loss Builder ===
        self.criterion = build_robust_criterion(config, class_weights)

        # === Knowledge Distillation (opsional) ===
        kd_cfg = (config.get('distillation') or {})
        self.kd_enabled: bool = bool(kd_cfg.get('enable', False))
        self.kd_T: float = float(kd_cfg.get('temperature', 2.0))
        self.kd_alpha: float = float(kd_cfg.get('alpha', 0.5))
        self.kd_after_epoch: int = int(kd_cfg.get('after_epoch', 0))
        self.teacher_model = None

        if self.kd_enabled:
            try:
                # Nama model teacher & checkpoint dari YAML
                t_name = str(kd_cfg.get('teacher_model', config['model'].get('model_name', '')))
                t_ckpt = str(kd_cfg.get('teacher_checkpoint', ''))
                if not t_name:
                    raise ValueError("distillation.teacher_model kosong, tidak bisa membangun teacher.")
                if not t_ckpt:
                    raise ValueError("distillation.teacher_checkpoint kosong, tidak bisa membangun teacher.")

                # Build teacher dengan config turunan + head 5 kelas.
                # Penting: panggil create_model_from_config dengan CONFIG, bukan argumen terpisah,
                # supaya kompatibel dengan definisi di model_factory + kebijakan offline.
                teacher_config = dict(config)
                teacher_model_cfg = dict(config.get('model', {}))
                teacher_model_cfg['model_name'] = t_name
                teacher_model_cfg['num_classes'] = int(config['model'].get('num_classes', 5))
                # Offline policy: gunakan checkpoint teacher sebagai local_weights_path
                teacher_model_cfg['local_weights_path'] = t_ckpt
                teacher_config['model'] = teacher_model_cfg

                t_model = create_model_from_config(teacher_config, logger=self.logger)

                if not os.path.exists(t_ckpt):
                    raise FileNotFoundError(f"Teacher checkpoint not found: {t_ckpt}")
                state = torch.load(t_ckpt, map_location='cpu')
                # Bisa raw state_dict atau dict dengan key 'state_dict'
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                incompat = t_model.load_state_dict(state, strict=True)

                missing = list(getattr(incompat, "missing_keys", []))
                unexpected = list(getattr(incompat, "unexpected_keys", []))

                # toleransi: hanya boleh head/classifier yang missing (sesuaikan prefix sesuai backbone Anda)
                allowed_prefix = ("head.", "classifier.", "fc.", "last_linear.")

                bad_missing = [k for k in missing if not k.startswith(allowed_prefix)]

                if bad_missing or unexpected:
                    raise RuntimeError(
                        f"Teacher checkpoint partial-load tidak aman.\n"
                        f"bad_missing({len(bad_missing)}): {bad_missing[:20]}\n"
                        f"unexpected({len(unexpected)}): {unexpected[:20]}"
                    )

                t_model.to(self.device)
                for p in t_model.parameters():
                    p.requires_grad = False
                t_model.eval()
                self.teacher_model = t_model

                if self.logger:
                    self.logger.info(
                        f"🧠 KD enabled in Trainer: teacher={t_name}, "
                        f"T={self.kd_T:.2f}, alpha={self.kd_alpha:.2f}, after_epoch={self.kd_after_epoch}"
                    )
            except Exception as e:
                if self.logger:
                    self.logger.error(f"🔴 KD disabled in Trainer (build teacher gagal): {e}")
                self.teacher_model = None
                self.kd_enabled = False


        self.scaler = GradScaler(enabled=self.use_amp)

        # Checkpointing policy
        ckpt_cfg = (config.get('checkpointing') or {})
        self.save_by = str(ckpt_cfg.get('save_by', 'loss')).lower()
        self.save_top_k = int(ckpt_cfg.get('save_top_k', 1))
        self.also_save_epochs = list(ckpt_cfg.get('also_save_epochs', []))
        self.monitor_split = str(ckpt_cfg.get('monitor_split', 'val')).lower()

        if self.monitor_split not in ("train", "val"):
            self.monitor_split = "val"  # fallback aman
        # Map save_by -> maximize?
        # - balanced_accuracy / f1: maximize
        # - ece / loss: minimize
        if self.save_by in ('balanced_accuracy', 'ba', 'f1'):
            self._maximize = True
        elif self.save_by in ('ece', 'loss'):
            self._maximize = False
        else:
            # fallback aman
            self.save_by = 'loss'
            self._maximize = False

        out_dir = self.config['model']['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        base_ckpt_name = self.config['model']['checkpoint_name']
        self.primary_ckpt_path = os.path.join(out_dir, base_ckpt_name)  # kompat lama

        # Top-K keeper
        self.topk = TopKKeeper(
            k=self.save_top_k,
            maximize=self._maximize,
            out_dir=out_dir,
            base_name=base_ckpt_name
        )

        # Tracking best (untuk log ringkas)
        self.best_monitor_value = (-float('inf') if self._maximize else float('inf'))
        self.best_epoch = 0
        self.best_val_ba = 0.0
        self.best_val_f1 = 0.0
        self.best_val_loss = float('inf')
        # Additional metrics to report at training completion
        self.best_val_acc = 0.0
        self.best_val_precision = 0.0
        self.best_val_recall = 0.0

        if self.logger:
            self.logger.info("\n" + "="*80)
            self.logger.info("TRAINER INITIALIZED")
            self.logger.info("="*80)
            self.logger.info(f"Optimizer: {opt_name} (LR_eff: {eff_lr}, WD: {wd})  [base_lr={base_lr}]")
            self.logger.info(f"Scheduler: {sch_name} (warmup_epochs={self.warmup_epochs})")
            self.logger.info(f"Mixed Precision: {self.use_amp}")
            self.logger.info(f"Grad Accum: {self.grad_accum_steps} steps")
            self.logger.info(f"Total Epochs: {self.num_epochs}")
            self.logger.info(f"Checkpointing: save_by={self.save_by} | top_k={self.save_top_k} | also_save_epochs={self.also_save_epochs}")

    def _set_backbone_requires_grad(self, enable: bool) -> None:
        """Enable/disable grads for backbone; always keep head trainable."""
        for n, p in self.model.named_parameters():
            if id(p) in self.head_param_ids:
                p.requires_grad = True
            else:
                p.requires_grad = enable

    def _trainable_param_stats(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in self.head_params)
        head_trainable = sum(p.numel() for p in self.head_params if p.requires_grad)
        backbone_total = sum(p.numel() for p in self.backbone_params)
        backbone_trainable = sum(p.numel() for p in self.backbone_params if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'head_total': head_total,
            'head_trainable': head_trainable,
            'backbone_total': backbone_total,
            'backbone_trainable': backbone_trainable,
        }

    def _log_trainable_params(self, epoch_idx: int, headline: str) -> None:
        stats = self._trainable_param_stats()
        trainable_names = [n for n, p in zip(self.head_param_names + self.backbone_param_names, self.head_params + self.backbone_params) if p.requires_grad]
        msg = (
            f"{headline} | epoch={epoch_idx} | trainable={stats['trainable']:,}/{stats['total']:,} "
            f"(head {stats['head_trainable']:,}/{stats['head_total']:,}, "
            f"backbone {stats['backbone_trainable']:,}/{stats['backbone_total']:,})"
        )
        if self.logger:
            self.logger.info(msg)
            if trainable_names:
                self.logger.info(f"Trainable params (first 20): {trainable_names[:20]}")
        else:
            print(msg)
            if trainable_names:
                print(f"Trainable params (first 20): {trainable_names[:20]}")

    def _apply_freeze_policy(self, epoch_idx: int) -> None:
        if not self.freeze_enabled or self.freeze_epochs <= 0:
            return

        if epoch_idx <= self.freeze_epochs:
            if not self.backbone_frozen:
                self._set_backbone_requires_grad(False)
                self.backbone_frozen = True
                self._log_trainable_params(epoch_idx, "🧊 Freezing backbone")
                stats = self._trainable_param_stats()
                if stats['trainable'] > 0.05 * stats['total'] or stats['head_trainable'] > 1_000_000:
                    trainable_names = [n for n, p in zip(self.head_param_names + self.backbone_param_names, self.head_params + self.backbone_params) if p.requires_grad]
                    raise RuntimeError(
                        f"Freeze sanity check failed: trainable={stats['trainable']} of total {stats['total']} (>5%) "
                        f"or head_trainable={stats['head_trainable']} (>1M). Trainable names (first 50): {trainable_names[:50]}"
                    )
            else:
                self._log_trainable_params(epoch_idx, "🧊 Backbone remains frozen")
        else:
            if self.backbone_frozen:
                self._set_backbone_requires_grad(True)
                self.backbone_frozen = False
                self._log_trainable_params(epoch_idx, "🔥 Unfreezing backbone")
            else:
                self._log_trainable_params(epoch_idx, "🔥 Backbone already unfrozen")

    def _compute_loss_with_kd(self, outputs, labels, images, epoch_idx: int):
        """
        Hitung loss CE atau CE+KD tergantung konfigurasi distillation.
        epoch_idx: epoch 1-based (supaya konsisten dengan TwoStageTrainer).
        """
        loss_ce = self.criterion(outputs, labels)

        # Jika KD tidak aktif atau teacher tidak ada, pakai CE murni
        if (not getattr(self, "kd_enabled", False)) or self.teacher_model is None:
            return loss_ce
        # Warmup: mulai KD setelah kd_after_epoch (1-based)
        if epoch_idx < getattr(self, "kd_after_epoch", 0):
            return loss_ce

        with torch.no_grad():
            t_logits = self.teacher_model(images)
            t_logits = t_logits.float().clamp_(-20, 20)

        T = float(getattr(self, "kd_T", 2.0))
        outputs = outputs.float()
        s_logp = F.log_softmax(outputs / T, dim=1)
        t_prob = F.softmax(t_logits / T, dim=1)
        loss_kd = F.kl_div(s_logp, t_prob, reduction='batchmean') * (T * T)


        alpha = float(getattr(self, "kd_alpha", 0.5))
        return alpha * loss_ce + (1.0 - alpha) * loss_kd

    def _forward_loss(self, images, labels, epoch_idx: int):
        outputs = self.model(images)
        loss = self._compute_loss_with_kd(outputs, labels, images, epoch_idx)
        return outputs, loss

    def train_epoch(self, epoch: int):
        """epoch: 0-based index; untuk KD kita pakai epoch_idx = epoch+1 (1-based)."""
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        logits_accum, labels_accum = [], []

        epoch_idx = epoch + 1  # 1-based, konsisten dengan kd_after_epoch
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}/{self.num_epochs}")
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, _ = batch

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)

                # loss dihitung di fp32 (lebih stabil untuk BiTempered + KD)
                outputs_fp32 = outputs.float()
                loss = self._compute_loss_with_kd(outputs_fp32, labels, images, epoch_idx)
                loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                outputs, loss = self._forward_loss(images, labels, epoch_idx)
                loss = loss / self.grad_accum_steps
                loss.backward()
                if (step + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * self.grad_accum_steps

            all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            # Keep logits/labels on CPU to avoid GPU RAM blow-up during metric aggregation.
            logits_accum.append(outputs.detach().to(dtype=torch.float32).cpu())
            labels_accum.append(labels.detach().to(dtype=torch.long).cpu())

            pbar.set_postfix({'loss': running_loss / (step + 1)})

        # Flush leftover grads when steps are not divisible by grad_accum_steps.
        total_steps = len(self.train_loader)
        remainder = total_steps % self.grad_accum_steps
        if remainder != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        epoch_loss = running_loss / len(self.train_loader)
        epoch_ba = balanced_accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        # New metrics: accuracy, precision, recall (macro averages for multi-class)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        logits_all = torch.cat(logits_accum, dim=0)
        labels_all = torch.cat(labels_accum, dim=0)
        # BEFORE metrics (no temperature scaling)
        epoch_ece = _compute_ece_with_temperature(logits_all, labels_all, n_bins=15, temperature=1.0)
        epoch_nll = _compute_nll(logits_all, labels_all, temperature=1.0)
        num_classes = int(self.config.get('model', {}).get('num_classes', 5))
        epoch_brier = _compute_brier_multiclass_from_logits(logits_all, labels_all, num_classes=num_classes, temperature=1.0)
        epoch_ent = _entropy_from_logits(logits_all)

        n_samples = int(labels_all.numel())
        return epoch_loss, epoch_ba, epoch_f1, epoch_acc, epoch_precision, epoch_recall, epoch_ece, epoch_ent, epoch_nll, epoch_brier, n_samples

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        logits_accum, labels_accum = [], []

        for batch in tqdm(self.val_loader, desc="Validation"):
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, _ = batch

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            running_loss += float(loss.item())

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            logits_accum.append(outputs.to(dtype=torch.float32).cpu())
            labels_accum.append(labels.to(dtype=torch.long).cpu())

        val_loss = running_loss / len(self.val_loader)
        val_ba = balanced_accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        # New metrics: accuracy, precision, recall
        val_acc = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        logits_all = torch.cat(logits_accum, dim=0)
        labels_all = torch.cat(labels_accum, dim=0)
        # BEFORE metrics (no temperature scaling)
        val_ece = _compute_ece_with_temperature(logits_all, labels_all, n_bins=15, temperature=1.0)
        val_nll = _compute_nll(logits_all, labels_all, temperature=1.0)
        num_classes = int(self.config.get('model', {}).get('num_classes', 5))
        val_brier = _compute_brier_multiclass_from_logits(logits_all, labels_all, num_classes=num_classes, temperature=1.0)
        val_ent = _entropy_from_logits(logits_all)

        n_samples = int(labels_all.numel())
        return val_loss, val_ba, val_f1, val_acc, val_precision, val_recall, val_ece, val_ent, val_nll, val_brier, n_samples

    def _metric_value(self, val_loss, val_ba, val_f1, val_ece) -> Tuple[float, str]:
        if self.save_by in ('balanced_accuracy', 'ba'):
            return float(val_ba), 'BA'
        if self.save_by == 'f1':
            return float(val_f1), 'F1'
        if self.save_by == 'ece':
            return float(val_ece), 'ECE'
        # default loss
        return float(val_loss), 'Loss'

    def _is_better(self, current: float, best: float) -> bool:
        return (current > best) if self._maximize else (current < best)

    def train(self):
        if self.logger:
            self.logger.info("\n" + "="*80)
            self.logger.info("TRAINING START")
            self.logger.info("="*80)

        # jalur simpan untuk "primary best" (kompat lama)
        primary_best_path = None

        for epoch in range(self.num_epochs):
            epoch_idx = epoch + 1
            self._apply_freeze_policy(epoch_idx)

            tr_loss, tr_ba, tr_f1, tr_acc, tr_precision, tr_recall, tr_ece, tr_ent, tr_nll, tr_brier, tr_n = self.train_epoch(epoch)
            val_loss, val_ba, val_f1, val_acc, val_precision, val_recall, val_ece, val_ent, val_nll, val_brier, val_n = self.validate()

            # ringkasan log
            if self.logger:
                self.logger.info(f"\nEpoch {epoch_idx}/{self.num_epochs}")
                self.logger.info(
                    f"  Train n={tr_n} | Loss: {tr_loss:.4f} | BA: {tr_ba:.4f} | F1: {tr_f1:.4f} | Acc: {tr_acc:.4f} | "
                    f"Prec: {tr_precision:.4f} | Rec: {tr_recall:.4f} | "
                    f"ECE_before: {tr_ece:.4f} | NLL_before: {tr_nll:.4f} | Brier_before: {tr_brier:.4f} | H: {tr_ent:.4f}"
                )
                self.logger.info(
                    f"  Val   n={val_n} | Loss: {val_loss:.4f} | BA: {val_ba:.4f} | F1: {val_f1:.4f} | Acc: {val_acc:.4f} | "
                    f"Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | "
                    f"ECE_before: {val_ece:.4f} | NLL_before: {val_nll:.4f} | Brier_before: {val_brier:.4f} | H: {val_ent:.4f}"
                )
            else:
                print(f"Epoch {epoch_idx}/{self.num_epochs} | "
                      f"Train: n={tr_n}, loss={tr_loss:.4f}, BA={tr_ba:.4f}, F1={tr_f1:.4f}, Acc={tr_acc:.4f}, Prec={tr_precision:.4f}, Rec={tr_recall:.4f}, ECE_before={tr_ece:.4f}, NLL_before={tr_nll:.4f}, Brier_before={tr_brier:.4f}, H={tr_ent:.4f} | "
                      f"Val: n={val_n}, loss={val_loss:.4f}, BA={val_ba:.4f}, F1={val_f1:.4f}, Acc={val_acc:.4f}, Prec={val_precision:.4f}, Rec={val_recall:.4f}, ECE_before={val_ece:.4f}, NLL_before={val_nll:.4f}, Brier_before={val_brier:.4f}, H={val_ent:.4f}")

            # simpan metrik konvensi
            self.best_val_loss = min(self.best_val_loss, val_loss)
            self.best_val_ba   = max(self.best_val_ba,   val_ba)
            self.best_val_f1   = max(self.best_val_f1,   val_f1)
            # track additional metrics
            self.best_val_acc = max(self.best_val_acc, val_acc)
            self.best_val_precision = max(self.best_val_precision, val_precision)
            self.best_val_recall = max(self.best_val_recall, val_recall)

            # nilai monitor
            if self.monitor_split == "train":
                monitor_value, tag = self._metric_value(tr_loss, tr_ba, tr_f1, tr_ece)
            else:
                monitor_value, tag = self._metric_value(val_loss, val_ba, val_f1, val_ece)

            # simpan wajib untuk epoch tertentu
            if (epoch + 1) in self.also_save_epochs:
                out_dir = self.config['model']['output_dir']
                os.makedirs(out_dir, exist_ok=True)
                force_path = os.path.join(out_dir, f"{Path(self.config['model']['checkpoint_name']).stem}_epoch{epoch+1:02d}.pth")
                torch.save(self.model.state_dict(), force_path)
                if self.logger:
                    self.logger.info(f"  ✓ Forced save (epoch list): {force_path}")
                else:
                    print(f"  ✓ Forced save (epoch list): {force_path}")

            # top-k keeper (utama)
            saved_path = self.topk.consider(epoch + 1, monitor_value, self.model.state_dict(), f"{tag}_")
            if saved_path is not None:
                if self.logger:
                    self.logger.info(f"  ✓ Saved (top-k by {tag}): {saved_path}")
                else:
                    print(f"  ✓ Saved (top-k by {tag}): {saved_path}")

            # update primary best (kompat lama)
            if self._is_better(monitor_value, self.best_monitor_value):
                self.best_monitor_value = monitor_value
                self.best_epoch = epoch + 1
                # salin model ke nama primitif (checkpoint_name lama) agar kode lama tetap bekerja
                try:
                    Path(self.primary_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                torch.save(self.model.state_dict(), self.primary_ckpt_path)
                primary_best_path = self.primary_ckpt_path
                if self.logger:
                    self.logger.info(f"  ✓ Primary best updated ({tag}): {self.best_monitor_value:.4f} -> {self.primary_ckpt_path}")
                else:
                    print(f"  ✓ Primary best updated ({tag}): {self.best_monitor_value:.4f} -> {self.primary_ckpt_path}")

            if self.scheduler:
                self.scheduler.step()

        # rangkuman
        if self.logger:
            self.logger.info("\n" + "="*80)
            self.logger.info("TRAINING COMPLETE")
            self.logger.info("="*80)
            self.logger.info(f"Best epoch (by {self.save_by}) : {self.best_epoch}")
            self.logger.info(f"Best monitor value            : {self.best_monitor_value:.4f}")
            self.logger.info(f"Best val loss                 : {self.best_val_loss:.4f}")
            self.logger.info(f"Best val BA                   : {self.best_val_ba:.4f}")
            self.logger.info(f"Best val F1                   : {self.best_val_f1:.4f}")
            self.logger.info(f"Best val Acc                  : {self.best_val_acc:.4f}")
            self.logger.info(f"Best val Precision            : {self.best_val_precision:.4f}")
            self.logger.info(f"Best val Recall               : {self.best_val_recall:.4f}")
        else:
            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80)
            print(f"Best epoch (by {self.save_by}) : {self.best_epoch}")
            print(f"Best monitor value            : {self.best_monitor_value:.4f}")
            print(f"Best val loss                 : {self.best_val_loss:.4f}")
            print(f"Best val BA                   : {self.best_val_ba:.4f}")
            print(f"Best val F1                   : {self.best_val_f1:.4f}")
            print(f"Best val Acc                  : {self.best_val_acc:.4f}")
            print(f"Best val Precision            : {self.best_val_precision:.4f}")
            print(f"Best val Recall               : {self.best_val_recall:.4f}")

        # Tentukan checkpoint terbaik untuk kalibrasi suhu:
        # - gunakan primary_best_path (by monitor)
        best_for_calib = primary_best_path if primary_best_path is not None else self.primary_ckpt_path
        return best_for_calib


# =============================================================================
# MAIN TRAIN FUNCTION
# =============================================================================
def train(config_path: str, run_id: Optional[str] = None, output_root: str = 'models'):
    """
    Workflow:
      1) Load config (model + base_config)
      2) Build datasets/dataloaders (DRDataset 512x512 + mask)
      3) Build model (timm/local weights)
      4) Train (best by configured metric via checkpointing)
      5) Load best checkpoint (by monitor)
      6) T-scaling calibration
      7) Save T_optimal
    """
    # Load config YAML terlebih dahulu
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Cari base_config di folder yang sama (dibutuhkan untuk merge default)
    base_config_guess = ['base_config_coweps2.yaml', 'base_config.yaml']
    base_config_path = None
    for g in base_config_guess:
        p = Path(config_path).parent / g
        if p.exists():
            base_config_path = p
            break
    if base_config_path is None:
        raise FileNotFoundError("Base config (base_config_coweps2.yaml/base_config.yaml) tidak ditemukan di folder yang sama.")

    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Merge yang diperlukan
    # merge: base sebagai default, YAML boleh override
    config['paths'] = {**(base_config.get('paths') or {}), **(config.get('paths') or {})}
    if 'random_seed' not in config:
        config['random_seed'] = base_config.get('random_seed', 42)
    # Only fill robust_loss from base when the model YAML does not override it.
    if ('robust_loss' not in config) and base_config.get('robust_loss'):
        config['robust_loss'] = base_config['robust_loss']

    # Enforce mandatory output policy (models/<config_stem>/<run_id>/)
    config, run_dir, final_run_id = _apply_output_policy(config, config_path, run_id=run_id, output_root=output_root)

    # Tee stdout/stderr to train.log for reproducibility (manual + pipeline)
    # NOTE: We use atexit to avoid reindenting the whole training function body.
    train_log_path = run_dir / 'train.log'
    train_log_path.parent.mkdir(parents=True, exist_ok=True)
    import atexit
    _tee_fh = train_log_path.open('a', encoding='utf-8')
    _old_out, _old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(_old_out, _tee_fh)
    sys.stderr = _Tee(_old_err, _tee_fh)

    def _cleanup_tee():
        try:
            sys.stdout = _old_out
            sys.stderr = _old_err
        finally:
            try:
                _tee_fh.close()
            except Exception:
                pass

    atexit.register(_cleanup_tee)

    print("\n" + "="*80)
    print("CoWePS v2.5-ready - Generic Model Trainer")
    print("="*80)
    print(f"[OUTPUT_POLICY] run_dir={run_dir} run_id={final_run_id}")

    # Siapkan logger (butuh config untuk tahu log_dir)
    log_cfg = config.get('logging') or {}
    save_logs = bool(log_cfg.get('save_logs', False))
    log_dir = log_cfg.get('log_dir', str(run_dir / 'logs'))

    logger = logging.getLogger(f"training_{Path(config_path).stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()         # <--- ganti logger.handlers = []
    logger.propagate = False        # <--- penting agar tidak dobel ke root logger

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if save_logs:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(
            log_dir,
            f"training_{Path(config_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.info("Logger initialized")

    # Ambil nama model dengan lebih robust (tidak asumsi selalu ada 'model_name')
    model_cfg = config.get("model", {}) or {}
    model_name = (
        model_cfg.get("model_name")  # bentuk yang kita pakai di CoWePS
        or model_cfg.get("name")     # fallback jika suatu saat pakai 'name'
        or config.get("model_name")  # fallback kalau user taruh di top-level
        or config.get("name")        # fallback lain
        or "UNKNOWN_MODEL"
    )

    print(f"\nModel: {model_name}")
    print(f"Config loaded from: {config_path}")
    print(f"Base config: {base_config_path.name}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Seed
    import random
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_seed'])

    # opsional: deterministik (bisa lebih lambat)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Datasets & manifests
    processed_dir = config['paths']['processed_dir']
    manifests   = base_config.get('manifests', {}) or {}
    data_alias  = base_config.get('data', {}) or {}
    tr_cfg_conf = (config.get('training') or {})

    # Prioritas:
    # 1) training.train_manifest / training.val_manifest  (per-model override)
    # 2) base_config.manifests.train / validate          (global)
    # 3) base_config.data.* alias                        (kompat v2.4)
    # 4) fallback gold_standard_*.csv di processed_dir
    train_manifest = (
        tr_cfg_conf.get('train_manifest')
        or manifests.get('train')
        or data_alias.get('train_manifest_path')
        or os.path.join(processed_dir, 'gold_standard_train.csv')
    )
    val_manifest = (
        tr_cfg_conf.get('val_manifest')
        or tr_cfg_conf.get('validate_manifest')
        or manifests.get('validate')
        or data_alias.get('validation_manifest_path')
        or os.path.join(processed_dir, 'gold_standard_validate.csv')
    )


    print("\nLoading datasets...")
    print(f"  Train manifest: {train_manifest}")
    print(f"  Val manifest:   {val_manifest}")

    if not os.path.exists(train_manifest):
        raise FileNotFoundError(f"Train manifest not found: {train_manifest}")
    if not os.path.exists(val_manifest):
        raise FileNotFoundError(f"Val manifest not found: {val_manifest}")

    train_dataset = DRDataset(train_manifest, config, mode='train')
    val_dataset   = DRDataset(val_manifest, config, mode='val')



    df_train = pd.read_csv(train_manifest)
    def _resolve_label_col(df: pd.DataFrame, cfg: dict) -> str:
        tr = (cfg.get('training', {}) or {})
        preferred = tr.get('label_column')
        if isinstance(preferred, str) and preferred.strip():
            preferred = preferred.strip()
            if preferred not in df.columns:
                raise ValueError(
                    f"Configured training.label_column='{preferred}' not found in train manifest columns: {list(df.columns)}"
                )
            return preferred
        if 'label' in df.columns:
            return 'label'
        if 'weak_label_class' in df.columns:
            return 'weak_label_class'
        if 'grade' in df.columns:
            return 'grade'
        raise ValueError(
            f"Train manifest has no supported label column. Expected one of ['label','weak_label_class','grade'] but got: {list(df.columns)}"
        )

    label_col = _resolve_label_col(df_train, config)

    # Guard: if both common label columns exist, they must match.
    if 'label' in df_train.columns and 'weak_label_class' in df_train.columns:
        a = pd.to_numeric(df_train['label'], errors='coerce')
        b = pd.to_numeric(df_train['weak_label_class'], errors='coerce')
        mismatch = (a.notna() & b.notna() & (a.astype('Int64') != b.astype('Int64'))).sum()
        if mismatch > 0:
            raise ValueError(
                f"Train manifest label mismatch: {mismatch} rows differ between 'label' and 'weak_label_class'. "
                "Set training.label_column explicitly or fix the manifest."
            )

    # Guard tegas (kalau mismatch, STOP biar kamu tahu ada masalah data)
    assert len(train_dataset) == len(df_train), (
        f"Mismatch dataset vs manifest: len(train_dataset)={len(train_dataset)} vs len(df_train)={len(df_train)}. "
        "Periksa file missing/corrupt atau filtering di DRDataset."
    )
    # --- Deterministic: sampler + dataloader ---
    seed = int(config.get("random_seed", 42))

    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id: int):
        import random as _random
        worker_seed = (seed + worker_id) % (2**32)
        _random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    # --- end deterministic ---

    counts = df_train[label_col].value_counts().sort_index()

    # BALANCED jika semua kelas jumlahnya sama persis
    is_balanced = (counts.max() == counts.min())

    sampler = None
    shuffle_mode = True
    class_weights = None

    use_sampler_when_imbalanced = True  # mode X vs Y

    if is_balanced:
        print("Dataset BALANCED. Mode Y: sampler=OFF, shuffle=True.")
    else:
        print("Dataset IMBALANCED. Mode X diterapkan.")

        freq = (counts.values / counts.values.sum()).astype(np.float32)
        inv = (1.0 / np.clip(freq, 1e-6, 1.0)).astype(np.float32)
        inv = (inv / inv.mean()).astype(np.float32)
        inv = np.clip(inv, 1.0, 3.0).astype(np.float32)

        if use_sampler_when_imbalanced:
            # Mode X1: WeightedRandomSampler
            class_to_weight = {int(cls): float(inv[i]) for i, cls in enumerate(counts.index)}
            sample_weights = df_train[label_col].map(lambda y: class_to_weight[int(y)]).values.astype(np.float32)
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=g
            )
            shuffle_mode = False
            class_weights = None

        else:
            # Mode X2: class-weighted loss (tanpa sampler)
            sampler = None
            shuffle_mode = True
            class_weights = torch.tensor(inv, dtype=torch.float32, device=device)

    # Loader SELALU dibuat (untuk balanced maupun imbalanced)
    batch_size = int(config['training']['batch_size'])
    val_batch_size = int(config['training'].get('val_batch_size', batch_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_mode,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )


    # =========================
    # LR LINEAR SCALING (v2.5 standard: training.lr_linear only)
    # =========================
    tr_cfg = config.get("training") or {}

    bs_now = int(tr_cfg.get("batch_size", 1))
    ga_now = int(tr_cfg.get("gradient_accumulation_steps", 1))
    eff_now = bs_now * ga_now

    # Warn jika masih ada legacy top-level (tidak dipakai)
    if config.get("lr_linear") is not None:
        msg = ("WARNING: 'lr_linear' ditemukan di top-level YAML. "
               "Standar v2.5: taruh di 'training.lr_linear'. Abaikan top-level.")
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        # opsional supaya tidak bikin bingung:
        # config.pop("lr_linear", None)

    lr_lin = tr_cfg.get("lr_linear") or {}
    lr_source = "training.lr_linear" if lr_lin else "default(fallback)"
    enabled = bool(lr_lin.get("enabled", False))  # default OFF

    msg_lr_head = f"[LR Linear Rule] source={lr_source} | enabled={enabled} | cfg={lr_lin}"
    if logger:
        logger.info(msg_lr_head)
    else:
        print(msg_lr_head)

    base_lr = float(tr_cfg.get("learning_rate", 1e-4))

    bs_base = int(lr_lin.get("base_batch_size", bs_now))
    ga_base = int(lr_lin.get("base_grad_accum", ga_now))
    eff_base = max(1, bs_base * ga_base)

    if not enabled:
        lr_eff = base_lr
        factor = 1.0
    else:
        factor = eff_now / eff_base

        cap = lr_lin.get("cap_max_factor", None)
        if cap is not None:
            try:
                factor = min(factor, float(cap))
            except Exception:
                pass

        lr_eff = base_lr * factor

        round_to = lr_lin.get("round_to", 0.0)
        try:
            round_to = float(round_to)
        except Exception:
            round_to = 0.0

        if round_to > 0:
            lr_eff = round(lr_eff / round_to) * round_to

    tr_cfg["learning_rate_effective"] = float(lr_eff)

    msg_lr = (f"[LR Linear Rule] enabled={enabled} | base_lr={base_lr:g} | base_eff={eff_base} "
              f"(bs={bs_base},ga={ga_base}) → now_eff={eff_now} (bs={bs_now},ga={ga_now}) "
              f"→ factor={factor:.4g} → effective_lr={lr_eff:g}")
    if logger:
        logger.info(msg_lr)
    else:
        print(msg_lr)



    # =========================
    # Model & Trainer
    # =========================
    print("\nCreating model...")
    model = create_model_from_config(config, logger=logger)

    trainer = Trainer(
        model, train_loader, val_loader, config, device,
        logger=logger, class_weights=class_weights
    )


    # Train & dapatkan path checkpoint terbaik (by monitor)
    best_ckpt_path = trainer.train()

    # Load checkpoint terbaik untuk calibration
    print("\nLoading best checkpoint untuk calibration...")
    state_dict = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"✓ Loaded: {best_ckpt_path}")

    # Calibrate temperature
    T_optimal = calibrate_temperature(model, val_loader, device, logger=logger)

    # Save T_optimal
    calibration_path = config['model']['calibration_path']
    os.makedirs(os.path.dirname(calibration_path), exist_ok=True)
    torch.save(T_optimal, calibration_path)

    # Persist calibration report (before/after)
    try:
        logits_all, labels_all = _collect_logits_labels(model, val_loader, device, logger=None)
        ece_criterion = ECELoss(n_bins=15)
        with torch.no_grad():
            ece_before = float(ece_criterion(logits_all, labels_all, temperature=1.0).item())
            ece_after = float(ece_criterion(logits_all, labels_all, temperature=float(T_optimal)).item())
            nll_before = _compute_nll(logits_all, labels_all, temperature=1.0)
            nll_after = _compute_nll(logits_all, labels_all, temperature=float(T_optimal))
            num_classes = int(getattr(model, 'num_classes', config.get('model', {}).get('num_classes', 5)))
            brier_before = _compute_brier_multiclass_from_logits(logits_all, labels_all, num_classes=num_classes, temperature=1.0)
            brier_after = _compute_brier_multiclass_from_logits(logits_all, labels_all, num_classes=num_classes, temperature=float(T_optimal))

        report = {
            'config_path': str(config_path),
            'best_checkpoint_path': str(best_ckpt_path),
            'calibration_path': str(calibration_path),
            'T_optimal': float(T_optimal),
            'ece_before': ece_before,
            'ece_after': ece_after,
            'nll_before': nll_before,
            'nll_after': nll_after,
            'brier_before': brier_before,
            'brier_after': brier_after,
        }

        out_dir = Path(config['model']['output_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path_json = out_dir / f"temperature_calibration_report_{Path(config_path).stem}.json"
        report_path_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
        report_path_yaml = out_dir / f"temperature_calibration_report_{Path(config_path).stem}.yaml"
        report_path_yaml.write_text(yaml.safe_dump(report, sort_keys=False), encoding='utf-8')
        if logger:
            logger.info(f"✓ Saved calibration report (json): {report_path_json}")
            logger.info(f"✓ Saved calibration report (yaml): {report_path_yaml}")
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Failed to persist calibration report: {e}")

    print(f"\n✓ Temperature saved: {calibration_path}")
    print("✓ Tier-1 trustworthiness metrics (ECE/NLL/Brier before/after) should be computed on ID/OOD test using this T_optimal via scripts/run_eval_registry.py")
    print(f"✓ Training complete untuk {config['model']['model_name']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Ensemble A model (metric-aware checkpointing)")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to model config (e.g., configs/convnext_config.yaml)")
    parser.add_argument('--run-id', type=str, default=None,
                        help="Unique run id (default: <timestamp>_seed<seed>)")
    parser.add_argument('--output-root', type=str, default='models',
                        help="Output root folder (default: models)")
    args = parser.parse_args()
    train(args.config, run_id=args.run_id, output_root=args.output_root)


