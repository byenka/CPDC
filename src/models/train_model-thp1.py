#!/usr/bin/env python
"""
CoWePS v2.4 - Generic Model Trainer untuk "Ensemble A"

KUNCI v2.4:
- Pelatih generik yang bekerja untuk SOTA (ConvNeXt, DINOv2, CLIP-ViT)
- Menggunakan DRDataset dengan masking 512x512
- Implementasi kalibrasi suhu (T-scaling) setelah training
- LOG CoWePS 4-serangkai per-epoch: BA (adil), F1 (akurat), ECE (jujur), Entropy (tenang)

Author: CoWePS v2.4 Implementation Team
"""

import os
import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings('ignore')

# Tambahkan project root ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import create_model_from_config
from src.data.data_processing import DRDataset


# ============================================================================
# TEMPERATURE CALIBRATION (ECE + T-scaling)
# ============================================================================
class ECELoss(nn.Module):
    """
    Expected Calibration Error (ECE) – Guo et al., 2017.
    """
    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=1)
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        # Buat n_bins+1 batas [0.0, 1.0] di device yang sama dengan logits
        bin_boundaries = torch.linspace(0.0, 1.0, steps=self.n_bins + 1, device=logits.device)
        ece = torch.zeros(1, device=logits.device)

        # Hitung ECE per-bin
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (confidences > lo) & (confidences <= hi)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                acc_in_bin = accuracies[in_bin].float().mean()
                conf_in_bin = confidences[in_bin].mean()
                ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
        return ece


@torch.no_grad()
def _collect_logits_labels(model: nn.Module, loader: DataLoader, device: torch.device, logger: logging.Logger | None):
    if logger:
        logger.info("Collecting validation logits...")
    model.eval()
    logits_list, labels_list = [], []
    for images, labels in tqdm(loader, desc="Collecting logits", disable=logger is None):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        logits_list.append(logits.cpu())
        labels_list.append(labels)
    logits_all = torch.cat(logits_list).to(device)
    labels_all = torch.cat(labels_list).to(device)
    if logger:
        logger.info(f"Total validation samples: {len(labels_all)}")
    return logits_all, labels_all


def calibrate_temperature(model: nn.Module, val_loader: DataLoader, device: torch.device, logger: logging.Logger | None):
    """
    Cari T optimal yang meminimalkan NLL; laporkan ECE sebelum/sesudah.
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("TEMPERATURE CALIBRATION (T-scaling)")
        logger.info("="*80)

    logits_all, labels_all = _collect_logits_labels(model, val_loader, device, logger)

    temperature = nn.Parameter(torch.ones(1, device=device))
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = ECELoss()

    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_func():
        optimizer.zero_grad(set_to_none=True)
        loss = nll_criterion(logits_all / temperature, labels_all)
        loss.backward()
        return loss

    optimizer.step(eval_func)

    T_opt = float(temperature.item())
    with torch.no_grad():
        ece_before = float(ece_criterion(logits_all, labels_all, temperature=1.0).item())
        ece_after  = float(ece_criterion(logits_all, labels_all, temperature=T_opt).item())

    if logger:
        logger.info("\n" + "="*80)
        logger.info("CALIBRATION RESULTS")
        logger.info("="*80)
        logger.info(f"Optimal Temperature: {T_opt:.4f}")
        logger.info(f"ECE before: {ece_before:.4f}")
        logger.info(f"ECE after : {ece_after:.4f}")
        logger.info(f"ECE improvement: {(ece_before - ece_after):.4f}")

    return T_opt


# ============================================================================
# UTIL: CoWePS metrics per-epoch
# ============================================================================
def _entropy_from_logits(logits: torch.Tensor) -> float:
    """Mean entropy over samples (natural log)."""
    # Entropy aman jika logits float32; hindari Half di CPU.
    logits = logits.to(dtype=torch.float32)
    probs = torch.softmax(logits, dim=1)
    ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)  # per-sample
    return float(ent.mean().item())



def _compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    # Pastikan dtype & device valid untuk softmax:
    # - logits: float32
    # - labels: long
    # - device: samakan (GPU kalau tersedia, atau CPU) agar kernel tersedia.
    target_device = labels.device
    logits = logits.to(device=target_device, dtype=torch.float32, non_blocking=True)
    labels = labels.to(device=target_device, dtype=torch.long, non_blocking=True)

    ece = ECELoss(n_bins=n_bins)
    with torch.no_grad():
        val = float(ece(logits, labels, temperature=1.0).item())
    return val



# ============================================================================
# TRAINER
# ============================================================================
class Trainer:
    """
    Generic trainer untuk Ensemble A models (v2.4)
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        logger: logging.Logger | None = None,
        class_weights: torch.Tensor | None = None
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

        # Optimizer
        opt_name = tr_cfg['optimizer']
        lr = float(tr_cfg['learning_rate'])
        wd = float(tr_cfg['weight_decay'])
        if opt_name == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Scheduler (opsional) + optional warmup
        sch_name = tr_cfg.get('scheduler', 'CosineAnnealingLR')
        self.warmup_epochs = int(tr_cfg.get('warmup_epochs', 0))

        if sch_name == 'CosineAnnealingLR':
            sch_params = tr_cfg.get('scheduler_params', {})

            # Scheduler utama: CosineAnnealingLR
            # Jika ada warmup, T_max diperkecil ke (num_epochs - warmup_epochs)
            t_max_main = int(sch_params.get('T_max', self.num_epochs))
            if self.warmup_epochs > 0:
                t_max_main = max(1, self.num_epochs - self.warmup_epochs)

            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max_main,
                eta_min=float(sch_params.get('eta_min', 1e-6))
            )

            if self.warmup_epochs > 0:
                # Warmup: LR naik linier dari 0 → 1 selama warmup_epochs
                def warmup_lambda(epoch: int) -> float:
                    # epoch di sini dimulai dari 0
                    if epoch < self.warmup_epochs:
                        return float(epoch + 1) / float(self.warmup_epochs)
                    return 1.0

                warmup_scheduler = optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=warmup_lambda
                )

                # Gabungkan warmup + main ke dalam satu scheduler
                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[self.warmup_epochs]
                )
            else:
                self.scheduler = main_scheduler
        else:
            self.scheduler = None


        # Loss (label smoothing + optional class weights)
        label_smoothing = float(tr_cfg.get('label_smoothing', 0.0))
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking best
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_val_ba = 0.0
        self.best_val_f1 = 0.0

        if self.logger:
            self.logger.info("\n" + "="*80)
            self.logger.info("TRAINER INITIALIZED")
            self.logger.info("="*80)
            self.logger.info(f"Scheduler: {sch_name} (warmup_epochs={self.warmup_epochs})")
            self.logger.info(f"Optimizer: {opt_name} (LR: {lr}, WD: {wd})")
            self.logger.info(f"Scheduler: {sch_name}")
            self.logger.info(f"Mixed Precision: {self.use_amp}")
            self.logger.info(f"Grad Accum: {self.grad_accum_steps} steps")
            self.logger.info(f"Total Epochs: {self.num_epochs}")

    def _forward_loss(self, images, labels):
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return outputs, loss

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        # Kumpulkan logits untuk ECE/Entropy (aman: kelas=5 → ringan)
        logits_accum, labels_accum = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        self.optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with autocast():
                    outputs, loss = self._forward_loss(images, labels)
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                outputs, loss = self._forward_loss(images, labels)
                loss = loss / self.grad_accum_steps
                loss.backward()
                if (step + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * self.grad_accum_steps
            # Kumpulkan pred & label
            all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            # Kumpulkan logits untuk ECE/Entropy (pastikan float32 agar aman di CPU/GPU)
            logits_accum.append(outputs.detach().to(device=self.device, dtype=torch.float32))
            labels_accum.append(labels.detach().to(device=self.device, dtype=torch.long))

            pbar.set_postfix({'loss': running_loss / (step + 1)})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_ba = balanced_accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')

        # ECE & Entropy
        logits_all = torch.cat(logits_accum, dim=0)
        labels_all = torch.cat(labels_accum, dim=0)
        epoch_ece = _compute_ece(logits_all, labels_all, n_bins=15)
        epoch_ent = _entropy_from_logits(logits_all)

        return epoch_loss, epoch_ba, epoch_f1, epoch_ece, epoch_ent

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        logits_accum, labels_accum = [], []

        for images, labels in tqdm(self.val_loader, desc="Validation"):
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

        logits_all = torch.cat(logits_accum, dim=0)
        labels_all = torch.cat(labels_accum, dim=0)
        val_ece = _compute_ece(logits_all, labels_all, n_bins=15)
        val_ent = _entropy_from_logits(logits_all)

        return val_loss, val_ba, val_f1, val_ece, val_ent

    def train(self):
        if self.logger:
            self.logger.info("\n" + "="*80)
            self.logger.info("TRAINING START")
            self.logger.info("="*80)

        for epoch in range(self.num_epochs):
            tr_loss, tr_ba, tr_f1, tr_ece, tr_ent = self.train_epoch(epoch)
            val_loss, val_ba, val_f1, val_ece, val_ent = self.validate()

            if self.logger:
                self.logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
                # CoWePS 4-serangkai
                self.logger.info(f"  Train Loss: {tr_loss:.4f} | BA(adil): {tr_ba:.4f} | F1(akurat): {tr_f1:.4f} | "
                                 f"ECE(jujur): {tr_ece:.4f} | Entropy(tenang): {tr_ent:.4f}")
                self.logger.info(f"  Val   Loss: {val_loss:.4f} | BA(adil): {val_ba:.4f} | F1(akurat): {val_f1:.4f} | "
                                 f"ECE(jujur): {val_ece:.4f} | Entropy(tenang): {val_ent:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs} | "
                      f"Train: loss={tr_loss:.4f}, BA={tr_ba:.4f}, F1={tr_f1:.4f}, ECE={tr_ece:.4f}, H={tr_ent:.4f} | "
                      f"Val: loss={val_loss:.4f}, BA={val_ba:.4f}, F1={val_f1:.4f}, ECE={val_ece:.4f}, H={val_ent:.4f}")

            # Best on val_loss (lebih kecil = lebih baik)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.best_val_ba = val_ba
                self.best_val_f1 = val_f1

                out_dir = self.config['model']['output_dir']
                os.makedirs(out_dir, exist_ok=True)
                ckpt_path = os.path.join(out_dir, self.config['model']['checkpoint_name'])
                torch.save(self.model.state_dict(), ckpt_path)
                if self.logger:
                    self.logger.info(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
                else:
                    print(f"  ✓ Best model saved (val_loss: {val_loss:.4f}) -> {ckpt_path}")

            if self.scheduler:
                self.scheduler.step()

        if self.logger:
            self.logger.info("\n" + "="*80)
            self.logger.info("TRAINING COMPLETE")
            self.logger.info("="*80)
            self.logger.info(f"Best epoch   : {self.best_epoch}")
            self.logger.info(f"Best val loss: {self.best_val_loss:.4f}")
            self.logger.info(f"Best val BA  : {self.best_val_ba:.4f}")
            self.logger.info(f"Best val F1  : {self.best_val_f1:.4f}")
        else:
            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80)
            print(f"Best epoch   : {self.best_epoch}")
            print(f"Best val loss: {self.best_val_loss:.4f}")
            print(f"Best val BA  : {self.best_val_ba:.4f}")
            print(f"Best val F1  : {self.best_val_f1:.4f}")


# ============================================================================
# MAIN TRAIN FUNCTION
# ============================================================================
def train(config_path: str):
    """
    Workflow:
      1) Load config (model + base_config)
      2) Build datasets/dataloaders (DRDataset 512x512 + mask)
      3) Build model (timm/local weights)
      4) Train (best by val_loss)
      5) Load best checkpoint
      6) T-scaling calibration
      7) Save T_optimal
    """
    print("\n" + "="*80)
    print("CoWePS v2.4 - Generic Model Trainer")
    print("="*80)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_config_path = Path(config_path).parent / 'base_config.yaml'
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Merge yang diperlukan
    config['paths'] = base_config['paths']
    config['random_seed'] = base_config.get('random_seed', 42)

    print(f"\nModel: {config['model']['model_name']}")
    print(f"Config loaded from: {config_path}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Optional logger
    logger = None
    log_cfg = config.get('logging', {})
    log_dir = log_cfg.get('log_dir')
    if log_cfg.get('save_logs', False) and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir,
            f"training_{Path(config_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logger = logging.getLogger(f"training_{Path(config_path).stem}")
        logger.setLevel(logging.INFO)
        logger.handlers = []  # reset
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_path)
        sh = logging.StreamHandler(sys.stdout)
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
        logger.info("Logger initialized")

    # Seed
    torch.manual_seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_seed'])

    # Datasets & manifests (prefer YAML, fallback ke gold_standard)
    processed_dir = config['paths']['processed_dir']

    # Ambil preferensi dari base_config (bukan dari model config)
    manifests   = base_config.get('manifests', {}) or {}
    data_alias  = base_config.get('data', {}) or {}

    train_manifest = (
        manifests.get('train')
        or data_alias.get('train_manifest_path')
        or os.path.join(processed_dir, 'gold_standard_train.csv')
    )
    val_manifest = (
        manifests.get('validate')
        or data_alias.get('validation_manifest_path')
        or os.path.join(processed_dir, 'gold_standard_validate.csv')
    )

    print("\nLoading datasets...")
    print(f"  Train manifest: {train_manifest}")
    print(f"  Val manifest:   {val_manifest}")

    # Safety check yang ramah
    if not os.path.exists(train_manifest):
        raise FileNotFoundError(f"Train manifest not found: {train_manifest}")
    if not os.path.exists(val_manifest):
        raise FileNotFoundError(f"Val manifest not found: {val_manifest}")


    train_dataset = DRDataset(train_manifest, config, mode='train')
    val_dataset   = DRDataset(val_manifest, config, mode='val')

    # Sampler berbasis imbalance (weak_label_class)
    df_train = pd.read_csv(train_manifest)
    label_col = 'weak_label_class' if 'weak_label_class' in df_train.columns else 'label'
    counts = df_train[label_col].value_counts().sort_index()
    freq = counts.values / counts.values.sum()
    inv = 1.0 / np.clip(freq, 1e-6, 1.0)
    inv = inv / inv.mean()
    inv = np.clip(inv, 1.0, 3.0)  # cap agar tidak ekstrem

    sample_weights = df_train[label_col].map(lambda y: inv[int(y)]).values.astype(np.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    batch_size = int(config['training']['batch_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("\nDatasets loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    # ================================================================
    # LR LINEAR SCALING RULE (berbasis effective batch size)
    # ================================================================
    tr_cfg = config['training']

    # Ambil setting saat ini
    bs_now = int(tr_cfg['batch_size'])
    ga_now = int(tr_cfg.get('gradient_accumulation_steps', 1))
    eff_now = bs_now * ga_now

    # Ambil konfigurasi basis dari YAML (fallback: pakai setting saat ini)
    lr_lin = tr_cfg.get('lr_linear', {}) or {}
    enabled = bool(lr_lin.get('enabled', True))  # default: aktif
    bs_base = int(lr_lin.get('base_batch_size', bs_now))
    ga_base = int(lr_lin.get('base_grad_accum', ga_now))
    eff_base = max(1, bs_base * ga_base)

    base_lr = float(tr_cfg.get('learning_rate', 1e-4))
    factor = eff_now / eff_base
    cap = lr_lin.get('cap_max_factor', None)
    if cap is not None:
        try:
            cap = float(cap)
            factor = min(factor, cap)
        except Exception:
            pass  # jika cap tidak valid, abaikan

    lr_eff = base_lr * factor

    # Rounding opsional agar rapi di log
    round_to = float(lr_lin.get('round_to', 0.0))
    if round_to and round_to > 0:
        lr_eff = round(lr_eff / round_to) * round_to

    # Simpan ke config → dipakai Trainer.__init__()
    tr_cfg['learning_rate_effective'] = lr_eff

    # Logging ke console & logger (jika ada)
    msg_lr = (f"[LR Linear Rule] base_lr={base_lr:g} | base_eff={eff_base} "
              f"(bs={bs_base},ga={ga_base}) → now_eff={eff_now} (bs={bs_now},ga={ga_now}) "
              f"→ factor={factor:.4g} → effective_lr={lr_eff:g}")
    print(msg_lr)
    if logger:
        logger.info(msg_lr)

    # ================================================================
    # Model & Trainer
    # ================================================================
    print("\nCreating model...")
    model = create_model_from_config(config)

    # Class weights tensor (untuk CrossEntropyLoss)
    class_weights = torch.tensor(inv, dtype=torch.float32, device=device)

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config, device,
                      logger=logger, class_weights=class_weights)

    # Train
    trainer.train()

    # Load best checkpoint (state_dict)
    print("\nLoading best checkpoint untuk calibration...")
    ckpt_path = os.path.join(config['model']['output_dir'], config['model']['checkpoint_name'])
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"✓ Loaded: {ckpt_path}")

    # Calibrate temperature
    T_optimal = calibrate_temperature(model, val_loader, device, logger=logger)

    # Save T_optimal
    calibration_path = config['model']['calibration_path']
    os.makedirs(os.path.dirname(calibration_path), exist_ok=True)
    torch.save(T_optimal, calibration_path)

    print(f"\n✓ Temperature saved: {calibration_path}")
    print(f"✓ Training complete untuk {config['model']['model_name']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Ensemble A model")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to model config (e.g., configs/convnext_config.yaml)")
    args = parser.parse_args()
    train(args.config)
