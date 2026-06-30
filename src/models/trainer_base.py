"""
CoWePS Trainer Base Module - v2.5 ready
- SafeTrainerMixins (AMP, grad clip, OOM guard)
- SAFE class weights
- StratifiedBatchSampler (balanced)
- TrainingMonitor (BA, per-class F1/recall, ECE, grad health)
- FundusDataset (kompatibel manifest v2.5: image_path/label/mask_path fleksibel)
- TwoStageTrainer (Stage-1 balanced, Stage-2 mixed) + OPTIONAL Knowledge Distillation
"""

import os
import gc
import cv2
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from contextlib import nullcontext
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# utils
from src.core.utils import (
    load_config, setup_logging, get_device, set_random_seed,
    load_dataframe, save_json, ProgressTracker
)

# robust loss (sudah ada di project Anda)
from src.models.train_model import BiTemperedLoss, SymmetricCrossEntropy, GeneralizedCrossEntropy

# factory untuk bangun teacher model (KD)
from src.models.model_factory import create_model_from_config


# ============================================================================
# SAFE TRAINER MIXINS
# ============================================================================
class SafeTrainerMixins:
    def __init__(self, model, optimizer, criterion, cfg, logger):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.logger = logger
        self.use_amp = bool(cfg.get('training', {}).get('mixed_precision', True))
        self.accum_steps = int(cfg.get('training', {}).get('gradient_accumulation_steps', 1))
        self.max_grad_norm = float(cfg.get('training', {}).get('max_grad_norm', 1.0))
        self.scaler = GradScaler(enabled=self.use_amp)

    def _autocast_ctx(self):
        if not self.use_amp:
            return nullcontext()
        amp_dtype = getattr(self, 'amp_dtype', torch.float16)
        return autocast(dtype=amp_dtype)

    def train_one_batch(self, batch, step_idx):
        images, targets = batch
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with self._autocast_ctx():
            outputs = self.model(images)
            loss = self.criterion(outputs, targets) / self.accum_steps
        if getattr(self, "scaler", None) is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return float(loss.item())

    def maybe_step(self, step_idx):
        if (step_idx + 1) % self.accum_steps != 0:
            return False
        if getattr(self, "scaler", None) is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def oom_guard(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.logger.error(f"🔴 OOM: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                raise
            else:
                raise


# ============================================================================
# SAFE CLASS WEIGHTS CALCULATION
# ============================================================================
def calculate_safe_class_weights(manifest_df: pd.DataFrame,
                                 num_classes: int = 5,
                                 max_weight_ratio: float = 2.0) -> torch.Tensor:
    """
    SAFE class weights (log scaling + cap rasio max_weight_ratio), mean-normalized.
    Bisa membaca kolom 'grade' atau 'label'.
    """
    print("\n📊 Calculating SAFE Class Weights...")

    label_col = 'grade' if 'grade' in manifest_df.columns else (
        'label' if 'label' in manifest_df.columns else None
    )
    if label_col is None:
        raise ValueError("Manifest must contain 'grade' or 'label' column for class weights.")
    class_counts = manifest_df[label_col].value_counts().sort_index()
    counts = np.zeros(num_classes)

    for c in range(num_classes):
        counts[c] = float(class_counts.get(c, 1))  # guard

    mean_count = counts.mean()
    weights = np.log(1.0 + (mean_count / counts))
    weights = weights / weights.mean()  # mean=1

    min_w, max_w = weights.min(), weights.max()
    if max_w / min_w > max_weight_ratio:
        print(f"⚠️ Weight ratio {max_w/min_w:.2f} exceeds limit {max_weight_ratio} → clipping")
        weights = np.clip(weights, min_w, min_w * max_weight_ratio)
        weights = weights / weights.mean()

    print("\n✅ Final Class Weights:")
    for i, w in enumerate(weights):
        print(f"   Grade {i}: weight={w:.3f} (n={int(counts[i]):,})")
    print(f"\n   Max/Min Ratio: {weights.max()/weights.min():.2f}x")
    return torch.FloatTensor(weights)


# ============================================================================
# STRATIFIED BATCH SAMPLER
# ============================================================================
class StratifiedBatchSampler(Sampler):
    """
    Balanced per-batch (batch_size harus kelipatan num_classes).
    """
    def __init__(self, dataset_labels: np.ndarray, batch_size: int,
                 num_classes: int = 5, drop_last: bool = True):
        self.labels = dataset_labels
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.drop_last = drop_last

        assert self.batch_size % self.num_classes == 0, \
            f"Batch size {self.batch_size} must be divisible by {self.num_classes}"
        self.samples_per_class = self.batch_size // self.num_classes

        self.class_indices = {c: np.where(self.labels == c)[0].tolist() for c in range(self.num_classes)}
        min_class_size = min(len(ix) for ix in self.class_indices.values())
        self.num_batches = max(1, min_class_size // self.samples_per_class)

        print(f"\n📦 Stratified Batch Sampler initialized:")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Samples per class per batch: {self.samples_per_class}")
        print(f"   Number of batches: {self.num_batches}")
        for c in range(self.num_classes):
            print(f"   Grade {c}: {len(self.class_indices[c]):,} samples")

    def __iter__(self):
        shuffled = {}
        for c in range(self.num_classes):
            ix = self.class_indices[c].copy()
            np.random.shuffle(ix)
            shuffled[c] = ix
        for bi in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                s = bi * self.samples_per_class
                e = s + self.samples_per_class
                batch.extend(shuffled[c][s:e])
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# ============================================================================
# TRAINING MONITOR
# ============================================================================
class TrainingMonitor:
    def __init__(self, num_classes: int = 5, logger=None,
                 explosion_threshold: float = 10.0, abort_threshold: float = 100.0,
                 min_balanced_accuracy: float = 0.15, ba_check_after_batches: int = 200):
        self.num_classes = num_classes
        self.logger = logger
        self.explosion_threshold = float(explosion_threshold)
        self.abort_threshold = float(abort_threshold)
        self.min_ba = float(min_balanced_accuracy)
        self.ba_check_after_batches = int(ba_check_after_batches)
        self.reset()

    def reset(self):
        self.predictions = []
        self.labels = []
        self.losses = []
        self.gradient_norms = []
        self.batch_class_distributions = []
        self.nan_count = 0
        self.gradient_explosion_count = 0
        self.confidences = []
        self.ece_bins = 20
        # Probabilistic metrics (require full probs)
        self._prob_n = 0
        self._sum_nll = 0.0
        self._sum_brier = 0.0
        self._sum_entropy = 0.0

    def update_batch(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss: float,
        gradient_norm: float = None,
        confidences: torch.Tensor = None,
        probs: torch.Tensor = None,
    ):
        self.predictions.extend(predictions.detach().cpu().numpy())
        self.labels.extend(labels.detach().cpu().numpy())
        self.losses.append(loss)
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
            if gradient_norm > self.explosion_threshold:
                self.gradient_explosion_count += 1
                if self.logger:
                    self.logger.warning(f"⚠️ Gradient explosion detected: {gradient_norm:.2f}")
        if np.isnan(loss):
            self.nan_count += 1
            if self.logger:
                self.logger.error("🔴 NaN loss detected!")
        batch_dist = Counter(labels.detach().cpu().numpy())
        self.batch_class_distributions.append(batch_dist)
        if confidences is not None:
            conf = confidences.detach().cpu().numpy()
            if conf.ndim > 1:
                conf = conf.max(axis=1)
            self.confidences.extend(conf.tolist())

        # Optional probabilistic metrics (NLL/Brier/Entropy) from full probs
        if probs is not None:
            try:
                pr = probs.detach().to(dtype=torch.float32).clamp_min(1e-12).cpu().numpy()
                yt = labels.detach().to(dtype=torch.long).cpu().numpy().astype(int)
                if pr.ndim == 2 and len(yt) == pr.shape[0]:
                    # renormalize row-wise for safety
                    row_sums = pr.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1.0
                    pr = pr / row_sums

                    p_true = pr[np.arange(len(yt)), yt]
                    nll = float(np.mean(-np.log(np.clip(p_true, 1e-12, 1.0))))

                    y_onehot = np.zeros_like(pr, dtype=np.float32)
                    y_onehot[np.arange(len(yt)), yt] = 1.0
                    brier = float(np.mean(np.sum((pr - y_onehot) ** 2, axis=1)))

                    ent = float(np.mean(-np.sum(pr * np.log(np.clip(pr, 1e-12, 1.0)), axis=1)))

                    bs = int(len(yt))
                    self._prob_n += bs
                    self._sum_nll += nll * bs
                    self._sum_brier += brier * bs
                    self._sum_entropy += ent * bs
            except Exception:
                # do not crash training due to metrics
                pass

    def update_gradient(self, gradient_norm: float, explosion_threshold: float = None):
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
            thr = self.explosion_threshold if explosion_threshold is None else float(explosion_threshold)
            if gradient_norm > thr:
                self.gradient_explosion_count += 1
                if self.logger:
                    self.logger.warning(f"⚠️ Gradient explosion detected: {gradient_norm:.2f}")

    def get_metrics(self) -> Dict[str, Any]:
        if len(self.predictions) == 0:
            return {}
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        accuracy = float((predictions == labels).mean())
        balanced_acc = float(balanced_accuracy_score(labels, predictions))
        try:
            f1_all = f1_score(labels, predictions, average=None, labels=list(range(self.num_classes)), zero_division=0)
            per_class_f1 = [float(x) for x in f1_all]
        except Exception:
            per_class_f1 = []
            for c in range(self.num_classes):
                per_class_f1.append(
                    f1_score(labels, predictions, labels=[c], average='macro', zero_division=0)
                    if (labels == c).sum() > 0 else 0.0
                )
        per_class_support = [int((labels == c).sum()) for c in range(self.num_classes)]
        try:
            cm = confusion_matrix(labels, predictions, labels=list(range(self.num_classes)))
        except Exception:
            cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

        per_class_recall = []
        for c in range(self.num_classes):
            tp = float(cm[c, c]); denom = float(cm[c].sum())
            per_class_recall.append((tp / denom) if denom > 0.0 else 0.0)

        pred_dist = Counter(predictions.tolist())
        pred_distribution = {c: float(pred_dist.get(c, 0) / len(predictions)) for c in range(self.num_classes)}

        avg_gradient = float(np.mean(self.gradient_norms)) if self.gradient_norms else 0.0
        max_gradient = float(np.max(self.gradient_norms)) if self.gradient_norms else 0.0

        ece = float('nan')
        try:
            if hasattr(self, 'confidences') and len(self.confidences) == len(predictions):
                conf = np.array(self.confidences, dtype=np.float32)
                bins = np.linspace(0.0, 1.0, int(getattr(self, 'ece_bins', 20)) + 1)
                ece_val = 0.0; N = len(conf)
                for i in range(len(bins) - 1):
                    lo, hi = bins[i], bins[i+1]
                    mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
                    if np.any(mask):
                        acc_b = float((predictions[mask] == labels[mask]).mean())
                        conf_b = float(conf[mask].mean())
                        ece_val += (mask.sum() / N) * abs(acc_b - conf_b)
                ece = float(ece_val)
        except Exception:
            pass

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'avg_loss': float(np.mean(self.losses)) if self.losses else 0.0,
            'per_class_f1': per_class_f1,
            'per_class_support': per_class_support,
            'per_class_recall': per_class_recall,
            'confusion_matrix': cm.tolist(),
            'prediction_distribution': pred_distribution,
            'avg_gradient_norm': avg_gradient,
            'max_gradient_norm': max_gradient,
            'nan_count': int(self.nan_count),
            'gradient_explosions': int(self.gradient_explosion_count),
            'min_class_f1': float(min(per_class_f1) if per_class_f1 else 0.0),
            'all_classes_learning': all(f1 > 0.01 for f1 in per_class_f1) if per_class_f1 else False,
            'ece': ece,
            # Probabilistic metrics (only if probs were provided)
            'nll': (float(self._sum_nll / self._prob_n) if self._prob_n > 0 else float('nan')),
            'brier_multiclass': (float(self._sum_brier / self._prob_n) if self._prob_n > 0 else float('nan')),
            'entropy_mean': (float(self._sum_entropy / self._prob_n) if self._prob_n > 0 else float('nan')),
        }

    def print_epoch_summary(self, epoch: int, stage: str = ""):
        m = self.get_metrics()
        if not m:
            return
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} {stage} SUMMARY")
        print(f"{'='*80}")
        acc = float(m.get('accuracy', 0.0))
        ba  = float(m.get('balanced_accuracy', 0.0))
        avg_loss = float(m.get('avg_loss', 0.0))
        ece = m.get('ece', float('nan'))
        nll = m.get('nll', float('nan'))
        brier = m.get('brier_multiclass', float('nan'))
        ent_mean = m.get('entropy_mean', float('nan'))
        print(f"Accuracy: {acc:.3f} | Balanced Accuracy: {ba:.3f} | Avg Loss: {avg_loss:.4f}")
        if not (isinstance(ece, float) and np.isnan(ece)):
            print(f"ECE (20 bins): {ece:.3f}")
        if not (isinstance(nll, float) and np.isnan(nll)):
            print(f"NLL (before): {nll:.4f}")
        if not (isinstance(brier, float) and np.isnan(brier)):
            print(f"Brier (before): {brier:.4f}")
        if not (isinstance(ent_mean, float) and np.isnan(ent_mean)):
            print(f"Entropy mean: {ent_mean:.4f}")

        pc_f1 = m.get('per_class_f1', [])
        pc_recall = m.get('per_class_recall', [])
        print("\nPer-class F1:")
        for c, f1v in enumerate(pc_f1):
            print(f"   Grade {c}: F1={f1v:.3f}")

        print("\nPer-class Recall:")
        for c in range(len(pc_recall)):
            print(f"   Grade {c}: recall={pc_recall[c]:.3f}")

        pred_dist = m.get('prediction_distribution', {})
        print("\n🎯 Prediction Distribution:")
        for c in range(len(pc_recall)):
            pct = float(pred_dist.get(c, 0.0) * 100.0)
            print(f"   Grade {c}: {pct:.1f}%")

        print("\n💊 Gradient Health:")
        print(f"   Average Norm: {float(m.get('avg_gradient_norm', 0.0)):.3f}")
        print(f"   Maximum Norm: {float(m.get('max_gradient_norm', 0.0)):.3f}")
        print(f"   Explosions (>{self.explosion_threshold}): {int(m.get('gradient_explosions', 0))}")
        print(f"   NaN Count: {int(m.get('nan_count', 0))}")

        if int(m.get('nan_count', 0)) > 0:
            print("\n🔴 CRITICAL: NaN losses detected!")
        if int(m.get('gradient_explosions', 0)) > 0:
            print(f"\n⚠️ WARNING: {int(m.get('gradient_explosions', 0))} gradient explosions")
        if not bool(m.get('all_classes_learning', True)):
            print("\n⚠️ WARNING: Some classes have F1 ≤ 0.01 (not learning)")
        if ba < self.min_ba:
            print("\n🔴 CRITICAL: BA below configured minimum")

    def should_stop_training(self) -> Tuple[bool, str]:
        m = self.get_metrics()
        if not m:
            return False, ""
        if m['nan_count'] > 5:
            return True, f"Too many NaN losses ({m['nan_count']})"
        total_seen = sum(sum(d.values()) for d in self.batch_class_distributions)
        if total_seen < self.ba_check_after_batches:
            return False, ""
        if m['balanced_accuracy'] < self.min_ba:
            return True, f"BA collapsed below {self.min_ba*100:.0f}% ({m['balanced_accuracy']:.3f})"
        if m['max_gradient_norm'] > self.abort_threshold:
            return True, f"Gradient explosion: {m['max_gradient_norm']:.2f}"
        return False, ""


# ============================================================================
# FUNDUS DATASET (v2.5 manifest friendly)
# ============================================================================
class FundusDataset(Dataset):
    """
    Kompatibel dengan manifest v2.5:
    - Path: pakai 'image_path' bila ada; fallback 'fundus_path'
    - Label: pakai 'label' bila ada; fallback 'grade'
    - mask_path opsional: bila kosong → coba infer <stem>_mask.png; jika tak ada → mask=1
    - image_id opsional: bila tak ada → stem dari image_path
    """
    def __init__(self, manifest_df: pd.DataFrame, transform=None, is_training: bool = True):
        self.manifest = manifest_df.reset_index(drop=True).copy()
        self.transform = transform
        self.is_training = is_training
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # Harmonisasi kolom
        if 'image_path' in self.manifest.columns:
            self.manifest['fundus_path'] = self.manifest['image_path']
        if 'label' in self.manifest.columns and 'grade' not in self.manifest.columns:
            self.manifest['grade'] = self.manifest['label']
        if 'image_id' not in self.manifest.columns:
            self.manifest['image_id'] = self.manifest.get('fundus_path', self.manifest.get('image_path', '')).map(
                lambda p: os.path.splitext(os.path.basename(str(p)))[0]
            )
        if 'source' not in self.manifest.columns:
            self.manifest['source'] = 'UNK'

        # Siapkan mask_path: jika tak ada atau kosong, infer
        if 'mask_path' not in self.manifest.columns:
            self.manifest['mask_path'] = ''
        self.manifest['mask_path'] = self.manifest['mask_path'].fillna('')
        def _infer_mask(p, m):
            if isinstance(m, str) and len(m) > 0:
                return m
            stem, ext = os.path.splitext(str(p))
            cand = f"{stem}_mask.png"
            return cand if os.path.exists(cand) else ''
        self.manifest['mask_path'] = [
            _infer_mask(p, m) for p, m in zip(self.manifest['fundus_path'], self.manifest['mask_path'])
        ]

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = int(idx[0])
        row = self.manifest.iloc[int(idx)]

        fundus_path = str(row['fundus_path'])
        mask_path   = str(row['mask_path']) if isinstance(row['mask_path'], str) else ''
        label = int(row['grade'])

        image = cv2.imread(fundus_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed: fundus_path='{fundus_path}' (idx={idx})")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"cv2.imread failed: mask_path='{mask_path}' (idx={idx})")
            image[mask == 0] = 0  # apply mask
        # else: tanpa mask → gunakan image apa adanya

        if self.transform:
            image = self.transform(image)
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        image = self.normalize(image)
        metadata = {'image_id': row['image_id'], 'source': row.get('source', 'UNK')}
        return image, label, metadata


# ============================================================================
# TRANSFORMS
# ============================================================================
def _parse_cj_param(x, name="brightness"):
    import numbers
    if x is None: return 0.0
    if isinstance(x, numbers.Number): return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            vals = [float(v) for v in s[1:-1].split(",")]
            if len(vals) == 2: return (float(vals[0]), float(vals[1]))
            if len(vals) == 1: return float(vals[0])
            raise ValueError(f"{name} must have 2 values if list string is used")
        try: return float(s)
        except Exception: raise ValueError(f"{name}='{x}' is not numeric")
    if isinstance(x, (list, tuple)):
        if len(x) == 2: return (float(x[0]), float(x[1]))
        if len(x) == 1: return float(x[0])
        raise ValueError(f"{name} length must be 2, got {len(x)}")
    raise ValueError(f"Unsupported type for {name}: {type(x)}")

def get_train_transforms(config: Dict, model_name: str) -> transforms.Compose:
    aug = config['training']['augmentation']
    is_vit = 'vit' in model_name.lower()
    b = _parse_cj_param(aug.get('brightness_range', 0.1), name="brightness")
    c = _parse_cj_param(aug.get('contrast_range',   0.1), name="contrast")
    rot = float(aug.get('rotation_degrees', 10))
    ops = [transforms.ToPILImage()]
    if is_vit:
        ops += [transforms.Resize(256), transforms.RandomCrop(224)]
    ops += [
        transforms.RandomHorizontalFlip(p=float(aug.get('horizontal_flip_prob', 0.5))),
        transforms.RandomRotation(degrees=rot),
        transforms.ColorJitter(brightness=b, contrast=c),
        transforms.ToTensor()
    ]
    return transforms.Compose(ops)

def get_val_transforms(model_name: str, config: Dict) -> transforms.Compose:
    is_vit = 'vit' in model_name.lower()
    if is_vit:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

def dynamic_sampler_params(cfg):
    bs = int(cfg.get('training', {}).get('batch_size', 15))
    num_classes = 5
    samples_per_class = max(1, bs // num_classes)
    return bs, samples_per_class


# ============================================================================
# TWO-STAGE TRAINER + OPTIONAL DISTILLATION
# ============================================================================
class TwoStageTrainer:
    """
    Stage 1: Balanced batches (StratifiedBatchSampler) → paksa semua kelas belajar
    Stage 2: Mixed sampler (random + stratified annealed) → optim performa
    + KD (opsional): α*CE + (1-α)*T^2*KL(softmax_t || softmax_s), mulai after_epoch
    """
    def __init__(self, model: nn.Module, model_name: str, config: Dict,
                 device: torch.device, logger):
        self.model = model.to(device)
        self.model_name = model_name
        self.config = config
        self.device = device
        self.logger = logger

        self.train_config = config['training']
        self.stage1_epochs = self.train_config['stage1']['epochs']
        self.stage2_epochs = self.train_config['stage2']['epochs']

        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # AMP dtype resolution
        self.use_amp = self.train_config.get('mixed_precision', True)
        if self.use_amp and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            supports_bf16 = capability[0] >= 8
            if supports_bf16:
                self.amp_dtype = torch.bfloat16
                self.scaler = None
                logger.info("✅ Using BFloat16 AMP")
            else:
                self.amp_dtype = torch.float16
                self.scaler = GradScaler()
                logger.info("⚙️  Using Float16 AMP with GradScaler")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
            logger.info("🛶 AMP disabled, using Float32")

        # KD config
        kd_cfg = self.config.get('distillation', {}) or {}
        self.kd_enabled: bool = bool(kd_cfg.get('enable', False))
        self.kd_T: float = float(kd_cfg.get('temperature', 2.0))
        self.kd_alpha: float = float(kd_cfg.get('alpha', 0.5))
        self.kd_after_epoch: int = int(kd_cfg.get('after_epoch', 0))
        self.teacher_model: Optional[nn.Module] = None

        if self.kd_enabled:
            try:
                self.teacher_model = self._build_teacher_model(kd_cfg)
                self.logger.info(
                    f"🧠 KD enabled: T={self.kd_T:.2f}, alpha={self.kd_alpha:.2f}, after_epoch={self.kd_after_epoch}"
                )
            except Exception as e:
                self.logger.error(f"🔴 KD disabled due to error building teacher: {e}")
                self.kd_enabled = False
                self.teacher_model = None

        # Monitor
        mon_cfg = self.config.get('monitoring', {})
        expl_thr = float(mon_cfg.get('gradient_explosion_threshold',
                                     self.train_config.get('grad_explosion_threshold', 25.0)))
        abort_thr = float(mon_cfg.get('gradient_abort_threshold', 100.0))
        min_ba = float(mon_cfg.get('min_balanced_accuracy',
                                   self.train_config.get('min_balanced_accuracy', 0.15)))
        ba_after = int(mon_cfg.get('ba_check_after_batches', 200))
        self.train_monitor = TrainingMonitor(logger=logger,
                                             explosion_threshold=expl_thr,
                                             abort_threshold=abort_thr,
                                             min_balanced_accuracy=min_ba,
                                             ba_check_after_batches=ba_after)
        self.val_monitor = TrainingMonitor(logger=logger,
                                           explosion_threshold=expl_thr,
                                           abort_threshold=abort_thr,
                                           min_balanced_accuracy=min_ba,
                                           ba_check_after_batches=ba_after)

        self.best_val_ba = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        # Gradient checkpointing (opsional)
        if self.train_config.get('memory_optimization', {}).get('use_gradient_checkpointing', False):
            m = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(m, 'set_grad_checkpointing'):
                m.set_grad_checkpointing(True)
                self.logger.info("Enabled gradient checkpointing on model.")
            else:
                self.logger.info("Model does not support set_grad_checkpointing; skipping.")

        self.logger.info("Two-Stage Trainer Initialized")
        self.logger.info(f"    Stage 1: {self.stage1_epochs} epochs (balanced)")
        self.logger.info(f"    Stage 2: {self.stage2_epochs} epochs (mixed)")

    def _build_teacher_model(self, kd_cfg: Dict) -> nn.Module:
        """
        Membangun teacher dari config:
        - kd_cfg['teacher_model'] (mis. "convnext_base.fb_in22k")
        - kd_cfg['teacher_checkpoint'] (pth offline)
        Model teacher diset eval dan di-float32 untuk stabilitas KD.
        """
        t_name = str(kd_cfg.get('teacher_model', self.model_name))
        t_ckpt = str(kd_cfg['teacher_checkpoint'])

        # Gunakan factory yang sama, tetapi panggil dengan CONFIG, bukan argumen terpisah.
        # Ambil config global, override bagian model untuk teacher.
        teacher_config = dict(self.config)
        teacher_model_cfg = dict(self.config.get('model', {}))
        teacher_model_cfg['model_name'] = t_name
        teacher_model_cfg['num_classes'] = int(teacher_model_cfg.get('num_classes', 5))
        # Offline policy: gunakan checkpoint teacher sebagai local_weights_path
        teacher_model_cfg['local_weights_path'] = t_ckpt
        teacher_config['model'] = teacher_model_cfg

        t_model = create_model_from_config(teacher_config, logger=self.logger)

        if not os.path.exists(t_ckpt):
            raise FileNotFoundError(f"Teacher checkpoint not found: {t_ckpt}")
        state = torch.load(t_ckpt, map_location='cpu')
        # state dapat berupa raw state_dict atau checkpoint dict
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        t_model.load_state_dict(state, strict=False)
        t_model.to(self.device)
        for p in t_model.parameters():
            p.requires_grad = False
        t_model.eval()
        self.logger.info(f"Teacher loaded from: {t_ckpt}")
        return t_model


    # ---------------------- Stage setups ----------------------
    def setup_stage1(self, train_manifest: pd.DataFrame):
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 1: FORCE LEARNING ALL CLASSES")
        self.logger.info("="*80)

        stage1_cfg = self.train_config.get('stage1', {})
        max_wr = float(stage1_cfg.get('max_weight_ratio', 2.5))
        class_weights = calculate_safe_class_weights(train_manifest, max_weight_ratio=max_wr)
        class_weights = class_weights.to(self.device, dtype=torch.float32)

        rl = (self.config.get('robust_loss') or {})
        rl_type = str(rl.get('type', 'cross_entropy')).lower()
        params = rl.get('params', {}) or {}
        ls = float(stage1_cfg.get('label_smoothing', params.get('label_smoothing', 0.0)))

        if rl_type in ('bi_tempered', 'bitempered', 'bi-tempered'):
            t1 = float(params.get('t1', 0.7)); t2 = float(params.get('t2', 1.2))
            self.criterion = BiTemperedLoss(t1=t1, t2=t2, label_smoothing=ls)
        elif rl_type in ('symmetric_ce', 'symmetricce', 'sce'):
            alpha = float(params.get('alpha', 0.1)); beta = float(params.get('beta', 1.0))
            self.criterion = SymmetricCrossEntropy(alpha=alpha, beta=beta, weight=class_weights, label_smoothing=ls)
        elif rl_type in ('gce', 'generalized_ce', 'generalized_cross_entropy'):
            q = float(params.get('q', 0.7))
            self.criterion = GeneralizedCrossEntropy(q=q, weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)

        lr = float(stage1_cfg['learning_rate'])
        wd = float(self.train_config['weight_decay'])
        backbone_scale = float(stage1_cfg.get('backbone_lr_scale', 0.3))
        head_keywords = ('fc', 'classifier', 'head')
        backbone_params, head_params = [], []
        for n, p in self.model.named_parameters():
            (head_params if any(k in n for k in head_keywords) else backbone_params).append(p)
        self.optimizer = torch.optim.AdamW(
            [{'params': backbone_params, 'lr': max(lr * backbone_scale, 1e-7), 'tag': 'backbone'},
             {'params': head_params,     'lr': lr,                              'tag': 'head'}],
            weight_decay=wd
        )

        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = int(stage1_cfg.get('warmup_steps', 0))
        if warmup_steps > 0:
            def lr_lambda(step):
                return float(step + 1) / float(max(1, warmup_steps)) if step < warmup_steps else 1.0
            self.warmup = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.warmup = None

        self.max_grad_norm = float(self.train_config.get('max_grad_norm', 5.0))
        self.expl_thresh = float(self.train_config.get('grad_explosion_threshold', 25.0))
        self.expl_decay = float(self.train_config.get('grad_explosion_lr_decay', 0.5))
        self.expl_patience = int(self.train_config.get('grad_explosion_patience', 50))
        self._expl_count = 0
        self._global_step = 0

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )

    def setup_stage2(self, train_manifest: pd.DataFrame):
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 2: OPTIMIZE PERFORMANCE")
        self.logger.info("="*80)

        stage2_cfg = self.train_config.get('stage2', {})
        max_wr = float(stage2_cfg.get('max_weight_ratio', 3.0))
        class_weights = calculate_safe_class_weights(train_manifest, max_weight_ratio=max_wr)
        class_weights = class_weights.to(self.device, dtype=torch.float32)

        rl = (self.config.get('robust_loss') or {})
        rl_type = str(rl.get('type', 'cross_entropy')).lower()
        params = rl.get('params', {}) or {}
        ls = float(stage2_cfg.get('label_smoothing', params.get('label_smoothing', 0.0)))

        if rl_type in ('bi_tempered', 'bitempered', 'bi-tempered'):
            t1 = float(params.get('t1', 0.7)); t2 = float(params.get('t2', 1.2))
            self.criterion = BiTemperedLoss(t1=t1, t2=t2, label_smoothing=ls)
        elif rl_type in ('symmetric_ce', 'symmetricce', 'sce'):
            alpha = float(params.get('alpha', 0.1)); beta = float(params.get('beta', 1.0))
            self.criterion = SymmetricCrossEntropy(alpha=alpha, beta=beta, weight=class_weights, label_smoothing=ls)
        elif rl_type in ('gce', 'generalized_ce', 'generalized_cross_entropy'):
            q = float(params.get('q', 0.7))
            self.criterion = GeneralizedCrossEntropy(q=q, weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)

        lr = float(stage2_cfg['learning_rate'])
        wd = float(self.train_config['weight_decay'])
        backbone_scale2 = float(stage2_cfg.get('backbone_lr_scale', 0.5))
        head_keywords = ('fc', 'classifier', 'head')
        backbone_params, head_params = [], []
        for n, p in self.model.named_parameters():
            (head_params if any(k in n for k in head_keywords) else backbone_params).append(p)
        self.optimizer = torch.optim.AdamW(
            [{'params': backbone_params, 'lr': max(lr * backbone_scale2, 1e-7), 'tag': 'backbone'},
             {'params': head_params,     'lr': lr,                               'tag': 'head'}],
            weight_decay=wd
        )

        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = int(stage2_cfg.get('warmup_steps', 0))
        self.warmup = LambdaLR(self.optimizer, lambda step: (float(step + 1) / float(max(1, warmup_steps)))
                               if (warmup_steps > 0 and step < warmup_steps) else 1.0) if warmup_steps > 0 else None

        self.max_grad_norm = float(self.train_config.get('max_grad_norm', 5.0))
        self.expl_thresh = float(self.train_config.get('grad_explosion_threshold', 25.0))
        self.expl_decay = float(self.train_config.get('grad_explosion_lr_decay', 0.5))
        self.expl_patience = int(self.train_config.get('grad_explosion_patience', 50))
        self._expl_count = 0
        self._global_step = 0

        pat = int(self.train_config.get('scheduler_patience', 3))
        fac = float(self.train_config.get('scheduler_factor', 0.5))
        minlr = float(self.train_config.get('min_lr', 1e-7))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=fac, patience=pat, min_lr=minlr, verbose=True
        )

    # ---------------------- Train / Validate ----------------------
    def _compute_loss_with_kd(self, student_logits: torch.Tensor, labels: torch.Tensor,
                              images: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Gabungkan CE (self.criterion) dengan KD bila diaktifkan.
        loss = α * CE + (1-α) * T^2 * KL(soft_t || soft_s)
        """
        # CE (selalu)
        loss_ce = self.criterion(student_logits, labels)

        if not self.kd_enabled or epoch < self.kd_after_epoch or self.teacher_model is None:
            return loss_ce

        with torch.no_grad():
            t_logits = self.teacher_model(images)
            t_logits = t_logits.float().clamp_(-20, 20)

        T = self.kd_T
        s_logp = F.log_softmax(student_logits / T, dim=1)
        t_prob = F.softmax(t_logits / T, dim=1)
        loss_kd = F.kl_div(s_logp, t_prob, reduction='batchmean') * (T * T)

        return self.kd_alpha * loss_ce + (1.0 - self.kd_alpha) * loss_kd

    def train_epoch(self, train_loader, epoch: int, stage: int,
                    is_iterator: bool = False, steps_per_epoch: int = None) -> Dict:
        self.model.train()
        self.train_monitor.reset()

        if is_iterator:
            assert steps_per_epoch and steps_per_epoch > 0, "steps_per_epoch wajib saat is_iterator=True"
            num_batches = steps_per_epoch
        else:
            num_batches = len(train_loader)

        print_freq = max(1, num_batches // 20)

        if is_iterator:
            get_next = train_loader
            for batch_idx in range(num_batches):
                images, labels, metadata = next(get_next)
                try:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    if self.train_config.get('channels_last', True):
                        images = images.to(memory_format=torch.channels_last)

                    accum_default = int(self.train_config.get('gradient_accumulation_steps', 1))
                    accum_stage2  = int(self.train_config.get('stage2', {}).get('gradient_accumulation_steps', accum_default))
                    accum_steps   = accum_stage2 if stage == 2 else accum_default
                    if (batch_idx % accum_steps) == 0:
                        self.optimizer.zero_grad(set_to_none=True)

                    amp_enabled = self.train_config.get('mixed_precision', True)
                    if stage == 2:
                        amp_enabled = bool(self.train_config.get('mixed_precision_stage2', amp_enabled))
                    with autocast(enabled=amp_enabled, dtype=self.amp_dtype):
                        s_logits = self.model(images)
                    s_logits = s_logits.float().clamp_(-20, 20)
                    loss = self._compute_loss_with_kd(s_logits, labels, images, epoch) / accum_steps
                    if not torch.isfinite(loss):
                        self.logger.warning("⚠️  Train loss is NaN/Inf, decaying LR and skipping step.")
                        for g in self.optimizer.param_groups:
                            if g.get('tag') == 'backbone':
                                g['lr'] = max(g['lr'] * 0.5, 1e-7)
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if ((batch_idx + 1) % accum_steps) == 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.train_monitor.update_gradient(float(grad_norm), explosion_threshold=self.expl_thresh)

                        if (not torch.isfinite(grad_norm)) or (grad_norm > self.expl_thresh):
                            self._expl_count += 1
                            for g in self.optimizer.param_groups:
                                if g.get('tag') == 'backbone':
                                    g['lr'] = max(g['lr'] * self.expl_decay, 1e-7)
                            self.optimizer.zero_grad(set_to_none=True)
                        else:
                            if self.scaler is not None:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            self._expl_count = max(0, self._expl_count - 1)

                        if self.warmup is not None:
                            self.warmup.step()
                        self._global_step += 1

                    if self._expl_count > self.expl_patience:
                        self.logger.error("🔴 Too many gradient explosions – aborting epoch.")
                        return None

                    with torch.no_grad():
                        probs = torch.softmax(s_logits, dim=1)
                        confs, preds = probs.max(1)
                        self.train_monitor.update_batch(preds, labels, loss.item() * accum_steps, None, confs, probs=probs)

                    if (batch_idx + 1) % print_freq == 0:
                        m = self.train_monitor.get_metrics()
                        print(
                            f"\r  Stage {stage} Epoch {epoch} [{batch_idx+1}/{num_batches}] - "
                            f"Loss: {m.get('avg_loss',0):.4f}, BA: {m.get('balanced_accuracy',0):.3f}, "
                            f"Grad: {m.get('avg_gradient_norm',0):.2f}",
                            end=""
                        )

                    should_stop, reason = self.train_monitor.should_stop_training()
                    if should_stop:
                        self.logger.error(f"\n🔴 Stopping training: {reason}")
                        return None

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        self.logger.warning("CUDA OOM: skip batch & torch.cuda.empty_cache()")
                        torch.cuda.empty_cache()
                        continue
                    raise
        else:
            for batch_idx, (images, labels, metadata) in enumerate(train_loader):
                try:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    if self.train_config.get('channels_last', True):
                        images = images.to(memory_format=torch.channels_last)

                    accum_default = int(self.train_config.get('gradient_accumulation_steps', 1))
                    accum_stage2  = int(self.train_config.get('stage2', {}).get('gradient_accumulation_steps', accum_default))
                    accum_steps   = accum_stage2 if stage == 2 else accum_default
                    if (batch_idx % accum_steps) == 0:
                        self.optimizer.zero_grad(set_to_none=True)

                    amp_enabled = self.train_config.get('mixed_precision', True)
                    if stage == 2:
                        amp_enabled = bool(self.train_config.get('mixed_precision_stage2', amp_enabled))
                    with autocast(enabled=amp_enabled, dtype=self.amp_dtype):
                        s_logits = self.model(images)
                    s_logits = s_logits.float().clamp_(-20, 20)
                    loss = self._compute_loss_with_kd(s_logits, labels, images, epoch) / accum_steps

                    if not torch.isfinite(loss):
                        self.logger.warning("⚠️  Train loss is NaN/Inf, decaying LR and skipping step.")
                        for g in self.optimizer.param_groups:
                            g['lr'] = max(g['lr'] * 0.5, 1e-7)
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if ((batch_idx + 1) % accum_steps) == 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.train_monitor.update_gradient(float(grad_norm), explosion_threshold=self.expl_thresh)

                        if (not torch.isfinite(grad_norm)) or (grad_norm > self.expl_thresh):
                            self._expl_count += 1
                            self.logger.warning(f"⚠️  Gradient explosion: {grad_norm:.2f}")
                            for g in self.optimizer.param_groups:
                                if g.get('tag') == 'backbone':
                                    g['lr'] = max(g['lr'] * self.expl_decay, 1e-7)
                            self.optimizer.zero_grad(set_to_none=True)
                        else:
                            if self.scaler is not None:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            # zero grad next cycle
                            if self.warmup is not None:
                                self.warmup.step()
                            self._global_step += 1

                    if self._expl_count > self.expl_patience:
                        self.logger.error("🔴 Too many gradient explosions – aborting epoch.")
                        return None

                    with torch.no_grad():
                        probs = torch.softmax(s_logits, dim=1)
                        confs, preds = probs.max(1)
                        self.train_monitor.update_batch(preds, labels, loss.item() * accum_steps, None, confs, probs=probs)

                    if (batch_idx + 1) % print_freq == 0:
                        m = self.train_monitor.get_metrics()
                        print(
                            f"\r  Stage {stage} Epoch {epoch} [{batch_idx+1}/{num_batches}] - "
                            f"Loss: {m.get('avg_loss',0):.4f}, BA: {m.get('balanced_accuracy',0):.3f}, "
                            f"Grad: {m.get('avg_gradient_norm',0):.2f}",
                            end=""
                        )

                    should_stop, reason = self.train_monitor.should_stop_training()
                    if should_stop:
                        self.logger.error(f"\n🔴 Stopping training: {reason}")
                        return None

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        self.logger.warning("CUDA OOM: skip batch & torch.cuda.empty_cache()")
                        torch.cuda.empty_cache()
                        continue
                    raise

        print()
        return self.train_monitor.get_metrics()

    def validate(self, val_loader: DataLoader, current_epoch: int = 0) -> Dict:
        self.model.eval()
        self.val_monitor.reset()
        with torch.no_grad():
            for images, labels, metadata in val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                amp_enabled = self.train_config.get('mixed_precision', True)
                with autocast(enabled=amp_enabled, dtype=self.amp_dtype):
                    logits = self.model(images)
                logits = logits.float().clamp_(-20, 20)
                loss = self.criterion(logits, labels)
                if not torch.isfinite(loss):
                    self.logger.warning("⚠️  Val loss is NaN/Inf – skipping this batch.")
                    continue
                probs = torch.softmax(logits, dim=1)
                confs, preds = probs.max(1)
                self.val_monitor.update_batch(preds, labels, float(loss.item()), None, confs, probs=probs)
        return self.val_monitor.get_metrics()

    def evaluate(self, test_loader: DataLoader, return_raw: bool = True) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        all_logits, all_probs, all_preds, all_labels, all_losses = [], [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch[:2]  # support (img, lbl) or (img, lbl, meta)
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                amp_enabled = self.train_config.get('mixed_precision', True)
                with autocast(enabled=amp_enabled, dtype=self.amp_dtype):
                    logits = self.model(images)
                logits = logits.float().clamp_(-20, 20)
                loss = self.criterion(logits, labels)
                if torch.isfinite(loss):
                    all_losses.append(loss.item())
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_logits.append(logits.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,5), dtype=np.float32)
        all_probs  = np.concatenate(all_probs,  axis=0) if all_probs  else np.zeros((0,5), dtype=np.float32)
        all_preds  = np.concatenate(all_preds,  axis=0) if all_preds  else np.zeros((0,),  dtype=np.int64)
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,),  dtype=np.int64)

        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        metrics = {
            'loss': np.mean(all_losses) if all_losses else float('nan'),
            'accuracy': float(accuracy_score(all_labels, all_preds)) if len(all_labels) else 0.0,
            'balanced_accuracy': float(balanced_accuracy_score(all_labels, all_preds)) if len(all_labels) else 0.0,
            'f1_score': float(f1_score(all_labels, all_preds, average='macro')) if len(all_labels) else 0.0,
            'num_samples': int(len(all_labels))
        }
        if len(all_labels):
            f1_per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(5)))
            for i, f1v in enumerate(f1_per_class):
                metrics[f'f1_class_{i}'] = float(f1v)
        return (metrics, all_logits, all_probs, all_labels) if return_raw else (metrics, None, None, None)

    def save_checkpoint(self, epoch: int, stage: int, metrics: Dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else {},
            'best_val_ba': self.best_val_ba,
            'metrics': metrics
        }
        checkpoint_dir = self.config['paths']['checkpoints_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        latest_path = os.path.join(checkpoint_dir, f'{self.model_name}_latest.pth')
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = os.path.join(checkpoint_dir, f'best_{self.model_name}.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"⭐⭐⭐ New best model! Val BA: {metrics['balanced_accuracy']:.3f}")

    def train_full_pipeline(self, train_manifest: pd.DataFrame,
                            val_manifest: pd.DataFrame) -> Dict:
        def _set_backbone_requires_grad(model, enable: bool):
            for n, p in model.named_parameters():
                if any(k in n for k in ['fc', 'classifier', 'head']):
                    p.requires_grad = True
                else:
                    p.requires_grad = enable

        freeze_epochs = int(self.train_config.get('stage1', {}).get('freeze_backbone_epochs', 0))
        if freeze_epochs > 0:
            _set_backbone_requires_grad(self.model, False)
            self.logger.info(f"🧊 Freezing backbone for first {freeze_epochs} epoch(s)")

        # ===== STAGE 1 =====
        self.logger.info("\n" + "#"*80)
        self.logger.info("STARTING STAGE 1: FORCE LEARNING ALL CLASSES")
        self.logger.info("#"*80)

        self.setup_stage1(train_manifest)

        train_dataset = FundusDataset(train_manifest,
                                      transform=get_train_transforms(self.config, self.model_name),
                                      is_training=True)
        val_dataset = FundusDataset(val_manifest,
                                    transform=get_val_transforms(self.model_name, self.config),
                                    is_training=False)

        train_labels = (train_manifest['grade'] if 'grade' in train_manifest.columns else train_manifest['label']).values
        bs1 = int(self.train_config['stage1']['batch_size'])
        num_classes = 5
        if bs1 < num_classes or (bs1 % num_classes != 0):
            raise ValueError(f"Stage-1 batch_size must be >= {num_classes} and divisible by {num_classes} (got {bs1}).")

        train_loader_stage1 = DataLoader(
            train_dataset,
            batch_sampler=StratifiedBatchSampler(dataset_labels=train_labels,
                                                 batch_size=bs1, num_classes=num_classes, drop_last=True),
            num_workers=self.train_config['num_workers'],
            pin_memory=self.train_config.get('pin_memory', True),
            persistent_workers=self.train_config.get('persistent_workers', True)
        )
        val_bs = int(self.train_config.get('val_batch_size', self.train_config['stage2']['batch_size']))
        val_loader = DataLoader(
            val_dataset, batch_size=val_bs, shuffle=False,
            num_workers=self.train_config['num_workers'],
            pin_memory=self.train_config.get('pin_memory', True),
            persistent_workers=self.train_config.get('persistent_workers', True)
        )

        stage1_history = {'train': [], 'val': []}
        for epoch in range(1, self.stage1_epochs + 1):
            if freeze_epochs > 0 and epoch > freeze_epochs:
                _set_backbone_requires_grad(self.model, True)
                self.logger.info(f"🔥 Unfreezing backbone at epoch {epoch}")
                freeze_epochs = 0

            print(f"\n{'='*80}")
            print(f"STAGE 1 - EPOCH {epoch}/{self.stage1_epochs}")
            print(f"{'='*80}")

            train_metrics = self.train_epoch(train_loader_stage1, epoch, stage=1)
            if train_metrics is None:
                self.logger.error("Training failed!")
                return {'success': False}

            val_metrics = self.validate(val_loader, current_epoch=epoch)
            self.scheduler.step()

            self.train_monitor.print_epoch_summary(epoch, "TRAIN")
            self.val_monitor.print_epoch_summary(epoch, "VAL")

            stage1_history['train'].append(train_metrics)
            stage1_history['val'].append(val_metrics)

            val_ba = val_metrics['balanced_accuracy']
            is_best = val_ba > self.best_val_ba
            if is_best:
                self.best_val_ba = val_ba
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, 1, val_metrics, is_best)

            if epoch >= 5 and val_metrics.get('min_class_f1', 0.0) > 0.2:
                self.logger.info("\n✅ Stage 1 Success: All classes F1 > 0.2")
                if epoch < self.stage1_epochs - 5:
                    self.logger.info("   Moving to Stage 2 early")
                    break

        # ===== STAGE 2 =====
        self.logger.info("\n" + "#"*80)
        self.logger.info("STARTING STAGE 2: OPTIMIZE PERFORMANCE")
        self.logger.info("#"*80)

        self.setup_stage2(train_manifest)

        bs2   = int(self.train_config['stage2']['batch_size'])
        if bs2 % 5 != 0:
            fixed = max(5, (bs2 // 5) * 5)
            self.logger.warning(f"Stage-2 batch_size {bs2} bukan kelipatan 5 — diganti ke {fixed}.")
            bs2 = fixed

        loader_random = DataLoader(
            train_dataset, batch_size=bs2, shuffle=True,
            num_workers=self.train_config['num_workers'],
            pin_memory=True, drop_last=True
        )
        train_labels = (train_manifest['grade'] if 'grade' in train_manifest.columns else train_manifest['label']).values
        loader_strat = DataLoader(
            train_dataset,
            batch_sampler=StratifiedBatchSampler(dataset_labels=train_labels,
                                                 batch_size=bs2, num_classes=5, drop_last=True),
            num_workers=self.train_config['num_workers'],
            pin_memory=True
        )

        def mixed_batch_iter(mix_p_local: float):
            it_r = iter(loader_random); it_s = iter(loader_strat)
            while True:
                use_strat = (random.random() < mix_p_local)
                try:
                    yield next(it_s if use_strat else it_r)
                except StopIteration:
                    if use_strat:
                        it_s = iter(loader_strat); yield next(it_s)
                    else:
                        it_r = iter(loader_random); yield next(it_r)

        stage2_history = {'train': [], 'val': []}
        steps_per_epoch = len(loader_random)

        for epoch in range(1, self.stage2_epochs + 1):
            print(f"\n{'='*80}")
            print(f"STAGE 2 - EPOCH {epoch}/{self.stage2_epochs}")
            print(f"{'='*80}")

            p_start = float(self.train_config.get('stage2', {}).get('sampler_mixture_p_start', 1.0))
            p_end   = float(self.train_config.get('stage2', {}).get('sampler_mixture_p',       0.30))
            anneal_E= int(self.train_config.get('stage2', {}).get('sampler_mixture_anneal_epochs', self.stage2_epochs))
            if anneal_E <= 0:
                mix_p_epoch = p_end
            else:
                ratio = min(max(epoch, 0) / float(anneal_E), 1.0)
                cos_t = 0.5 * (1.0 + math.cos(math.pi * ratio))
                mix_p_epoch = p_end + (p_start - p_end) * cos_t
            _lo, _hi = (p_start, p_end) if p_start <= p_end else (p_end, p_start)
            mix_p_epoch = float(min(max(mix_p_epoch, _lo), _hi))
            mix_p_epoch = float(min(max(mix_p_epoch, 0.0), 1.0))
            self.logger.info(
                f"[Stage2] mix_p_epoch={mix_p_epoch:.3f} (p_start={p_start}, p_end={p_end}, anneal_E={anneal_E}, epoch={epoch})"
            )

            train_metrics = self.train_epoch(
                mixed_batch_iter(mix_p_epoch), epoch, stage=2,
                is_iterator=True, steps_per_epoch=steps_per_epoch
            )
            if train_metrics is None:
                self.logger.error("Training failed!")
                return {'success': False}

            val_metrics = self.validate(val_loader, current_epoch=epoch)
            self.scheduler.step(val_metrics['balanced_accuracy'])

            self.train_monitor.print_epoch_summary(epoch, "TRAIN")
            self.val_monitor.print_epoch_summary(epoch, "VAL")

            stage2_history['train'].append(train_metrics)
            stage2_history['val'].append(val_metrics)

            val_ba = val_metrics['balanced_accuracy']
            is_best = val_ba > self.best_val_ba
            if is_best:
                self.best_val_ba = val_ba
                self.best_epoch = f"Stage2-{epoch}"
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, 2, val_metrics, is_best)

            if self.patience_counter >= self.train_config['early_stop_patience']:
                self.logger.info(f"\n⏹️ Early stopping triggered (patience={self.patience_counter})")
                break

        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Best Val BA: {self.best_val_ba:.3f} (Epoch {self.best_epoch})")

        return {
            'success': True,
            'best_val_ba': self.best_val_ba,
            'stage1_history': stage1_history,
            'stage2_history': stage2_history
        }
