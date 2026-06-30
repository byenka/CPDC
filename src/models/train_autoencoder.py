#!/usr/bin/env python
"""
CoWePS v2.4 - Autoencoder Training Script (Ensemble B)

Training loop untuk autoencoder yang belajar merekonstruksi gambar retina
berkualitas tinggi.

KUNCI v2.4:
- WAJIB menggunakan DRDataset dari src/data/data_processing.py
- Gambar sudah 512x512 dan dimasker (image * mask)
- Loss: MSELoss (reconstruction error)
- Target: Model learns to perfectly reconstruct high-quality images

Author: CoWePS v2.4 Implementation Team
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_autoencoder import create_autoencoder
from src.data.data_processing import DRDataset


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def save_reconstruction_samples(model, val_loader, device, save_path, num_samples=4):
    """
    Save sample reconstructions untuk visual inspection
    
    Args:
        model: Trained autoencoder
        val_loader: Validation DataLoader
        device: torch device
        save_path: Path untuk save figure
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get one batch (DRDataset bisa return >2 elemen → ambil index 0 sebagai image)
    batch = next(iter(val_loader))
    if isinstance(batch, (list, tuple)):
        images = batch[0]
    else:
        images = batch

    images = images[:num_samples].to(device)
    
    with torch.no_grad():
        reconstructed = model(images)
    
    # Move to CPU dan denormalize untuk visualization
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    # Move to CPU dan denormalize untuk visualization
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    
    for i in range(num_samples):
        # Original
        img_orig = images[i].permute(1, 2, 0).numpy()
        # Denormalize (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_orig = std * img_orig + mean
        img_orig = np.clip(img_orig, 0, 1)
        
        # Reconstructed
        img_recon = reconstructed[i].permute(1, 2, 0).numpy()
        img_recon = std * img_recon + mean
        img_recon = np.clip(img_recon, 0, 1)
        
        # Plot
        axes[i, 0].imshow(img_orig)
        axes[i, 0].set_title(f'Original {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img_recon)
        axes[i, 1].set_title(f'Reconstructed {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved reconstruction samples: {save_path}")


# ============================================================================
# AUTOENCODER TRAINER
# ============================================================================

class AutoencoderTrainer:
    """
    Trainer untuk Autoencoder (Ensemble B)
    """
    
    def __init__(self, model, train_loader, val_loader, config, device, logger=None):
        """
        Initialize trainer
        
        Args:
            model: Autoencoder model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Configuration dictionary
            device: torch device
            logger: Logger instance (optional)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # Training config
        training_config = config['training']
        self.num_epochs = training_config['num_epochs']
        self.use_amp = training_config.get('use_amp', True)
        
        # Optimizer
        optimizer_name = training_config['optimizer']
        lr = training_config['learning_rate']
        weight_decay = training_config['weight_decay']
        
        if optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Scheduler
        scheduler_name = training_config.get('scheduler', 'ReduceLROnPlateau')
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler_params = training_config.get('scheduler_params', {})
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_params.get('mode', 'min'),
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5),
                min_lr=scheduler_params.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Visualization
        self.visualize_every_n_epochs = config.get('validation', {}).get('visualize_every_n_epochs', 5)
        self.reconstruction_dir = config.get('logging', {}).get('reconstruction_dir', 'outputs/figures/autoencoder_reconstructions')
        os.makedirs(self.reconstruction_dir, exist_ok=True)
        
        # Gradient clipping
        self.max_grad_norm = training_config.get('max_grad_norm', 1.0)
        
        if self.logger:
            self.logger.info("\n" + "="*80)
            self.logger.info("AUTOENCODER TRAINER INITIALIZED")
            self.logger.info("="*80)
            self.logger.info(f"Optimizer: {optimizer_name} (LR: {lr}, WD: {weight_decay})")
            self.logger.info(f"Scheduler: {scheduler_name}")
            self.logger.info(f"Loss: MSELoss (reconstruction error)")
            self.logger.info(f"Mixed Precision: {self.use_amp}")
            self.logger.info(f"Total Epochs: {self.num_epochs}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch in pbar:
            # DRDataset bisa return (images, labels, meta, ...) → ambil index 0 sebagai image
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass dengan AMP
            if self.use_amp:
                with autocast():
                    reconstructed = self.model(images)
                    loss = self.criterion(reconstructed, images)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstructed = self.model(images)
                loss = self.criterion(reconstructed, images)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
        
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    
    def validate(self):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # DRDataset bisa return (images, labels, meta, ...)
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)
                
                reconstructed = self.model(images)
                loss = self.criterion(reconstructed, images)
                
                running_loss += loss.item()
        
        val_loss = running_loss / len(self.val_loader)
        return val_loss

    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("AUTOENCODER TRAINING START")
        print("="*80)
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Log
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                
                # Save checkpoint
                output_dir = self.config['model']['output_dir']
                os.makedirs(output_dir, exist_ok=True)
                checkpoint_path = os.path.join(output_dir, self.config['model']['checkpoint_name'])
                
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ Best model saved (val_loss: {val_loss:.6f})")
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Visualize reconstructions
            if (epoch + 1) % self.visualize_every_n_epochs == 0:
                save_path = os.path.join(self.reconstruction_dir, f'epoch_{epoch+1}_reconstructions.png')
                save_reconstruction_samples(self.model, self.val_loader, self.device, save_path)
        
        print("\n" + "="*80)
        print("AUTOENCODER TRAINING COMPLETE")
        print("="*80)
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val loss: {self.best_val_loss:.6f}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_autoencoder(config_path: str, base_config_path: str | None = None):
    """
    Main training function untuk Autoencoder (Ensemble B)
    
    KUNCI v2.4: 
    - WAJIB menggunakan DRDataset dengan masking 512x512
    - Autoencoder dilatih untuk merekonstruksi gambar berkualitas tinggi
    - Reconstruction error akan digunakan untuk Q-score di Fase 3
    
    Args:
        config_path: Path ke autoencoder config file
        base_config_path: (optional) Path ke base config (mis. base_config_coweps2.yaml)
    """
    print("\n" + "="*80)
    print("CoWePS v2.4 - Autoencoder Training (Ensemble B)")
    print("="*80)
    
    # Load config autoencoder
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Tentukan base_config yang dipakai
    if base_config_path is None:
        # default behaviour v2.4: pakai base_config.yaml di folder yang sama
        base_config_path_resolved = Path(config_path).parent / 'base_config.yaml'
    else:
        base_config_path_resolved = Path(base_config_path)
    
    if not base_config_path_resolved.exists():
        raise FileNotFoundError(f"Base config tidak ditemukan: {base_config_path_resolved}")
    
    with open(base_config_path_resolved, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Merge configs
    config['paths'] = base_config['paths']
    config['random_seed'] = base_config.get('random_seed', 42)

    
    print(f"\nArchitecture: {config['model']['architecture']}")
    print(f"Config loaded from: {config_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Set random seed
    torch.manual_seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])
    
    # KRITIS: Create datasets menggunakan DRDataset (WAJIB untuk konsistensi dengan Ensemble A)
    processed_dir = config['paths']['processed_dir']
    train_manifest = os.path.join(processed_dir, 'gold_standard_train.csv')
    val_manifest = os.path.join(processed_dir, 'gold_standard_validate.csv')
    
    print(f"\nLoading datasets (USING DRDataset with 512x512 masking)...")
    print(f"  Train manifest: {train_manifest}")
    print(f"  Val manifest: {val_manifest}")
    
    # KUNCI v2.4: WAJIB menggunakan DRDataset yang sudah implement masking
    train_dataset = DRDataset(train_manifest, config, mode='train')
    val_dataset = DRDataset(val_manifest, config, mode='val')
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nDatasets loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Batch size: {batch_size}")
    
    # Create model
    print(f"\nCreating autoencoder...")
    model = create_autoencoder(config)
    
    # Create trainer
    trainer = AutoencoderTrainer(model, train_loader, val_loader, config, device)
    
    # Train
    trainer.train()
    
    # Save final model (best checkpoint)
    final_model_path = config['model'].get('final_model_path', 'models/autoencoder/autoencoder_final.pth')
    best_checkpoint = os.path.join(config['model']['output_dir'], config['model']['checkpoint_name'])
    
    # Load best checkpoint
    model.load_state_dict(torch.load(best_checkpoint))
    
    # Save as final model
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\n✓ Final model saved: {final_model_path}")
    print(f"✓ Training complete untuk Autoencoder (Ensemble B)")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Autoencoder (Ensemble B)")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/autoencoder_config.yaml',
        help="Path to autoencoder config"
    )
    parser.add_argument(
        '--base-config',
        type=str,
        default=None,
        help="Path ke base config (mis. configs/base_config_coweps2.yaml)"
    )
    args = parser.parse_args()
    
    train_autoencoder(args.config, args.base_config)
