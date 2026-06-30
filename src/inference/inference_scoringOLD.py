#!/usr/bin/env python
"""
CoWePS v2.4 - Inference & Scoring Module (AI Ganda)

FASE 3: Operasi Bedah pada Inferensi

KUNCI v2.4:
- Ensemble A (3 SOTA models): Generate C-score (Confidence) dengan T-scaling
- Ensemble B (Autoencoder): Generate Q-score (Quality) via reconstruction error
- ROC Calibration: Find optimal Q-threshold using Youden's J
- Output: full_inference_results.csv dengan C_score, Q_score (binary), Pred_Class

Author: CoWePS v2.4 Implementation Team
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import create_model_from_config
from src.models.model_autoencoder import UNet
from src.data.data_processing import DRDataset


# ============================================================================
# HELPER: RECONSTRUCTION ERROR
# ============================================================================

def get_reconstruction_error(ae_model, images, device):
    """
    Compute reconstruction error (MSE) per image in batch
    
    Args:
        ae_model: Autoencoder model
        images: Batch of images (B, 3, 512, 512)
        device: torch device
    
    Returns:
        error_per_image: MSE error per image (B,) as numpy array
    """
    images = images.to(device)
    
    with torch.no_grad():
        reconstructed = ae_model(images)
        # Compute MSE per image (average over C, H, W)
        error_per_image = F.mse_loss(reconstructed, images, reduction='none').mean(dim=[1, 2, 3])
    
    return error_per_image.cpu().numpy()


# ============================================================================
# ROC CALIBRATION (KUNCI v2.3 - Youden's J)
# ============================================================================

def find_q_threshold_roc(ae_model, base_config, device, logger=None):
    """
    Find optimal Q-score threshold using ROC analysis (Youden's J)
    
    KUNCI v2.3: Menggantikan calibrate_quality_model yang naif.
    
    Method:
    1. Load "Good" images (gold_standard_validate.csv) → label=1
    2. Load "Bad" images (Folder 5 ungradable) → label=0
    3. Compute reconstruction errors for both
    4. Find threshold that maximizes Youden's J = TPR - FPR
    
    Args:
        ae_model: Trained autoencoder
        base_config: Base configuration dictionary
        device: torch device
        logger: Logger instance (optional)
    
    Returns:
        optimal_threshold: Optimal reconstruction error threshold
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("ROC CALIBRATION FOR Q-SCORE")
        logger.info("="*80)
    else:
        print("\n" + "="*80)
        print("ROC CALIBRATION FOR Q-SCORE")
        print("="*80)
    
    ae_model.eval()
    
    all_errors = []
    all_labels = []
    
    # ========================================================================
    # LOAD "GOOD" IMAGES (Label = 1)
    # ========================================================================
    
    if logger:
        logger.info("\n1. Loading 'Good' quality images...")
    else:
        print("\n1. Loading 'Good' quality images...")
    
    good_manifest_path = os.path.join(
        base_config['paths']['processed_dir'],
        'gold_standard_validate.csv'
    )
    
    if not os.path.exists(good_manifest_path):
        raise FileNotFoundError(f"Good images manifest not found: {good_manifest_path}")
    
    good_dataset = DRDataset(good_manifest_path, base_config, mode='val')
    good_loader = DataLoader(
        good_dataset,
        batch_size=base_config.get('inference', {}).get('batch_size', 8),
        shuffle=False,
        num_workers=4
    )
    
    if logger:
        logger.info(f"   Loaded: {len(good_dataset)} 'Good' images")
    else:
        print(f"   Loaded: {len(good_dataset)} 'Good' images")
    
    for images, _ in tqdm(good_loader, desc="Processing 'Good' images"):
        errors = get_reconstruction_error(ae_model, images, device)
        all_errors.extend(errors)
        all_labels.extend([1] * len(errors))  # 1 = Good quality
    
    # ========================================================================
    # LOAD "BAD" IMAGES (Label = 0)
    # ========================================================================
    
    if logger:
        logger.info("\n2. Loading 'Bad' quality images (Folder 5)...")
    else:
        print("\n2. Loading 'Bad' quality images (Folder 5)...")
    
    bad_data_path = base_config['paths']['data_root']
    bad_folder = os.path.join(bad_data_path, '5_ungradable_oia')
    
    if not os.path.exists(bad_folder):
        raise FileNotFoundError(f"Bad images folder not found: {bad_folder}")
    
    # Create temporary manifest for bad images
    import glob
    bad_images = glob.glob(os.path.join(bad_folder, '*.png'))
    bad_images = [f for f in bad_images if '_mask.png' not in f]
    
    bad_manifest_data = []
    for img_path in bad_images:
        mask_path = img_path.replace('.png', '_mask.png')
        if os.path.exists(mask_path):
            bad_manifest_data.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'weak_label_class': 5  # Ungradable
            })
    
    bad_manifest_df = pd.DataFrame(bad_manifest_data)
    bad_manifest_temp = '/tmp/bad_images_temp.csv'
    bad_manifest_df.to_csv(bad_manifest_temp, index=False)
    
    bad_dataset = DRDataset(bad_manifest_temp, base_config, mode='val')
    bad_loader = DataLoader(
        bad_dataset,
        batch_size=base_config.get('inference', {}).get('batch_size', 8),
        shuffle=False,
        num_workers=4
    )
    
    if logger:
        logger.info(f"   Loaded: {len(bad_dataset)} 'Bad' images")
    else:
        print(f"   Loaded: {len(bad_dataset)} 'Bad' images")
    
    for images, _ in tqdm(bad_loader, desc="Processing 'Bad' images"):
        errors = get_reconstruction_error(ae_model, images, device)
        all_errors.extend(errors)
        all_labels.extend([0] * len(errors))  # 0 = Bad quality
    
    # ========================================================================
    # ROC ANALYSIS
    # ========================================================================
    
    if logger:
        logger.info("\n3. Computing ROC curve...")
    else:
        print("\n3. Computing ROC curve...")
    
    # Convert to numpy
    errors_array = np.array(all_errors)
    labels_array = np.array(all_labels)
    
    # CRITICAL: Invert errors for ROC
    # Low error = Good (1), High error = Bad (0)
    # ROC expects high score = positive class
    # So we use: scores = -errors (negative errors)
    scores = -errors_array
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels_array, scores)
    
    # Find optimal threshold using Youden's J
    # Youden's J = TPR - FPR (maximize this)
    youdens_j = tpr - fpr
    optimal_idx = np.argmax(youdens_j)
    
    optimal_score_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    # Convert back to error threshold
    optimal_error_threshold = -optimal_score_threshold
    
    # Compute AUC
    roc_auc = auc(fpr, tpr)
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    if logger:
        logger.info("\n" + "="*80)
        logger.info("ROC CALIBRATION RESULTS")
        logger.info("="*80)
        logger.info(f"Total 'Good' images: {sum(labels_array == 1)}")
        logger.info(f"Total 'Bad' images: {sum(labels_array == 0)}")
        logger.info(f"\nOptimal Error Threshold: {optimal_error_threshold:.6f}")
        logger.info(f"  Images with error < {optimal_error_threshold:.6f} → Q_score = 1 (Good)")
        logger.info(f"  Images with error ≥ {optimal_error_threshold:.6f} → Q_score = 0 (Bad)")
        logger.info(f"\nPerformance at Optimal Threshold:")
        logger.info(f"  TPR (Sensitivity): {optimal_tpr:.4f}")
        logger.info(f"  FPR (1 - Specificity): {optimal_fpr:.4f}")
        logger.info(f"  Youden's J: {youdens_j[optimal_idx]:.4f}")
        logger.info(f"  AUC-ROC: {roc_auc:.4f}")
    else:
        print("\n" + "="*80)
        print("ROC CALIBRATION RESULTS")
        print("="*80)
        print(f"Total 'Good' images: {sum(labels_array == 1)}")
        print(f"Total 'Bad' images: {sum(labels_array == 0)}")
        print(f"\nOptimal Error Threshold: {optimal_error_threshold:.6f}")
        print(f"  Images with error < {optimal_error_threshold:.6f} → Q_score = 1 (Good)")
        print(f"  Images with error ≥ {optimal_error_threshold:.6f} → Q_score = 0 (Bad)")
        print(f"\nPerformance at Optimal Threshold:")
        print(f"  TPR (Sensitivity): {optimal_tpr:.4f}")
        print(f"  FPR (1 - Specificity): {optimal_fpr:.4f}")
        print(f"  Youden's J: {youdens_j[optimal_idx]:.4f}")
        print(f"  AUC-ROC: {roc_auc:.4f}")
    
    return optimal_error_threshold


# ============================================================================
# LOAD ENSEMBLE A (3 SOTA MODELS + T_OPTIMAL)
# ============================================================================

def load_ensemble_a(config_paths, device, logger=None):
    """
    Load Ensemble A (3 SOTA models) and their calibrated temperatures
    
    KUNCI v2.3: Load models + T_optimal from Fase 1
    
    Args:
        config_paths: List of 3 config paths (ConvNeXt, DINOv2, CLIP-ViT)
        device: torch device
        logger: Logger instance (optional)
    
    Returns:
        ensemble_a: List of 3 models
        temperatures: List of 3 T_optimal values
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("LOADING ENSEMBLE A (KLINISI SOTA)")
        logger.info("="*80)
    else:
        print("\n" + "="*80)
        print("LOADING ENSEMBLE A (KLINISI SOTA)")
        print("="*80)
    
    ensemble_a = []
    temperatures = []
    
    for i, config_path in enumerate(config_paths, 1):
        if logger:
            logger.info(f"\n{i}. Loading model from: {config_path}")
        else:
            print(f"\n{i}. Loading model from: {config_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load base config untuk paths
        base_config_path = Path(config_path).parent / 'base_config.yaml'
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configs
        config['paths'] = base_config['paths']
        
        # Create model
        model = create_model_from_config(config, logger=logger)
        
        # Load checkpoint
        checkpoint_path = os.path.join(
            config['model']['output_dir'],
            config['model']['checkpoint_name']
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device).eval()
        
        if logger:
            logger.info(f"   ✓ Model loaded: {checkpoint_path}")
        else:
            print(f"   ✓ Model loaded: {checkpoint_path}")
        
        # Load calibrated temperature
        T_path = config['model']['calibration_path']
        
        if not os.path.exists(T_path):
            raise FileNotFoundError(f"Temperature file not found: {T_path}")
        
        T_data = torch.load(T_path, map_location='cpu')
        
        # Handle different T_optimal formats
        if isinstance(T_data, dict):
            T_optimal = T_data['temperature']
        else:
            T_optimal = T_data
        
        if logger:
            logger.info(f"   ✓ Temperature loaded: T = {T_optimal:.4f}")
        else:
            print(f"   ✓ Temperature loaded: T = {T_optimal:.4f}")
        
        ensemble_a.append(model)
        temperatures.append(T_optimal)
    
    if logger:
        logger.info(f"\n✓ Ensemble A loaded: {len(ensemble_a)} models")
    else:
        print(f"\n✓ Ensemble A loaded: {len(ensemble_a)} models")
    
    return ensemble_a, temperatures


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def run_full_inference(base_config_path='configs/base_config.yaml',
                      ensemble_a_configs=None,
                      ae_config_path='configs/autoencoder_config.yaml',
                      logger=None):
    """
    Run full CoWePS inference pipeline (AI Ganda v2.4)
    
    WORKFLOW:
    1. Load base configuration
    2. Load Ensemble A (3 SOTA models + temperatures)
    3. Load Ensemble B (Autoencoder)
    4. Calibrate Q-threshold using ROC (Youden's J)
    5. Run inference on ALL 54k images
    6. Generate C-score (confidence with T-scaling)
    7. Generate Q-score (binary: 0 or 1)
    8. Save results to full_inference_results.csv
    
    Args:
        base_config_path: Path to base config
        ensemble_a_configs: List of 3 config paths for Ensemble A
        ae_config_path: Path to autoencoder config
        logger: Logger instance (optional)
    
    Returns:
        results_df: DataFrame with inference results
    """
    print("\n" + "="*80)
    print("CoWePS v2.4 - FULL INFERENCE PIPELINE (AI GANDA)")
    print("="*80)
    
    # Default Ensemble A configs if not provided
    if ensemble_a_configs is None:
        ensemble_a_configs = [
            'configs/convnext_config.yaml',
            'configs/dinov2_config.yaml',
            'configs/clipvit_config.yaml'
        ]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # ========================================================================
    # 1. LOAD CONFIGURATIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("1. LOADING CONFIGURATIONS")
    print("="*80)
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    print(f"✓ Base config loaded: {base_config_path}")
    
    with open(ae_config_path, 'r') as f:
        ae_config = yaml.safe_load(f)
    
    # Merge base paths into ae_config
    ae_config['paths'] = base_config['paths']
    print(f"✓ Autoencoder config loaded: {ae_config_path}")
    
    # ========================================================================
    # 2. LOAD ENSEMBLE A (KLINISI SOTA)
    # ========================================================================
    
    ensemble_a, temperatures = load_ensemble_a(ensemble_a_configs, device, logger)
    
    # ========================================================================
    # 3. LOAD ENSEMBLE B (TEKNISI KUALITAS)
    # ========================================================================
    
    print("\n" + "="*80)
    print("LOADING ENSEMBLE B (TEKNISI KUALITAS)")
    print("="*80)
    
    ae_model = UNet(n_channels=3, n_classes=3, bilinear=True).to(device).eval()
    
    ae_checkpoint = os.path.join(
        ae_config['model']['output_dir'],
        ae_config['model']['checkpoint_name']
    )
    
    if not os.path.exists(ae_checkpoint):
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_checkpoint}")
    
    ae_model.load_state_dict(torch.load(ae_checkpoint, map_location=device))
    print(f"✓ Autoencoder loaded: {ae_checkpoint}")
    
    # ========================================================================
    # 4. CALIBRATE Q-THRESHOLD (ROC / YOUDEN'S J)
    # ========================================================================
    
    Q_threshold = find_q_threshold_roc(ae_model, base_config, device, logger)
    
    # ========================================================================
    # 5. PREPARE MAIN DATA LOADER (ALL 54K IMAGES)
    # ========================================================================
    
    print("\n" + "="*80)
    print("5. PREPARING MAIN DATA LOADER")
    print("="*80)
    
    master_list_path = os.path.join(
        base_config['paths']['processed_dir'],
        'master_list.csv'
    )
    
    if not os.path.exists(master_list_path):
        raise FileNotFoundError(f"Master list not found: {master_list_path}")
    
    main_dataset = DRDataset(master_list_path, base_config, mode='val')
    main_loader = DataLoader(
        main_dataset,
        batch_size=base_config.get('inference', {}).get('batch_size', 8),
        shuffle=False,
        num_workers=4
    )
    
    print(f"✓ Main dataset loaded: {len(main_dataset)} images")
    print(f"✓ Batch size: {main_loader.batch_size}")
    
    # ========================================================================
    # 6. RUN MAIN INFERENCE LOOP
    # ========================================================================
    
    print("\n" + "="*80)
    print("6. RUNNING FULL INFERENCE (AI GANDA)")
    print("="*80)
    
    results = []
    
    # Get manifest for image paths
    manifest_df = pd.read_csv(master_list_path)
    
    batch_idx = 0
    with torch.no_grad():
        for images, labels in tqdm(main_loader, desc="CoWePS Inference"):
            images = images.to(device)
            batch_size_actual = images.shape[0]
            
            # ================================================================
            # ENSEMBLE A (KLINISI) - C-SCORE WITH T-SCALING
            # ================================================================
            
            all_scaled_logits = []
            
            for model, T in zip(ensemble_a, temperatures):
                # Forward pass
                logits = model(images)
                
                # Apply T-scaling (KUNCI v2.3)
                scaled_logits = logits / T
                
                all_scaled_logits.append(scaled_logits.unsqueeze(0))
            
            # Average scaled logits
            avg_scaled_logits = torch.mean(torch.cat(all_scaled_logits, dim=0), dim=0)
            
            # Get C-score (confidence) and Pred_Class
            C_probs = torch.softmax(avg_scaled_logits, dim=1)
            C_score_tensor, Pred_Class_tensor = torch.max(C_probs, dim=1)
            
            # ================================================================
            # ENSEMBLE B (TEKNISI) - Q-SCORE VIA RECONSTRUCTION ERROR
            # ================================================================
            
            raw_error_batch = get_reconstruction_error(ae_model, images, device)
            
            # ================================================================
            # COMBINE RESULTS
            # ================================================================
            
            for i in range(batch_size_actual):
                # Get image path from manifest
                img_idx = batch_idx * main_loader.batch_size + i
                img_path = manifest_df.iloc[img_idx]['image_path']
                
                # Extract scores
                C_score = C_score_tensor[i].item()
                Pred_Class = Pred_Class_tensor[i].item()
                raw_error = raw_error_batch[i]
                
                # Convert error to binary Q_score
                Q_score = 1 if raw_error < Q_threshold else 0
                
                results.append({
                    'image_path': img_path,
                    'C_score': C_score,
                    'Q_score': Q_score,  # Binary: 0 or 1
                    'Pred_Class': int(Pred_Class),
                    'reconstruction_error': raw_error  # For debugging
                })
            
            batch_idx += 1
    
    # ========================================================================
    # 7. SAVE RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("7. SAVING RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    
    # Create scores directory if not exists
    scores_dir = base_config['paths']['scores_dir']
    os.makedirs(scores_dir, exist_ok=True)
    
    output_path = os.path.join(scores_dir, 'full_inference_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Results saved: {output_path}")
    print(f"✓ Total images processed: {len(results_df):,}")
    
    # Print summary statistics
    print(f"\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    print(f"Total images: {len(results_df):,}")
    print(f"\nQ-score Distribution:")
    print(f"  Q=1 (Good quality): {(results_df['Q_score']==1).sum():,} ({(results_df['Q_score']==1).sum()/len(results_df)*100:.1f}%)")
    print(f"  Q=0 (Poor quality): {(results_df['Q_score']==0).sum():,} ({(results_df['Q_score']==0).sum()/len(results_df)*100:.1f}%)")
    print(f"\nPredicted Class Distribution:")
    for cls in range(5):
        count = (results_df['Pred_Class'] == cls).sum()
        print(f"  Class {cls}: {count:,} images")
    print(f"\nC-score Statistics:")
    print(f"  Mean: {results_df['C_score'].mean():.4f}")
    print(f"  Median: {results_df['C_score'].median():.4f}")
    print(f"  Min: {results_df['C_score'].min():.4f}")
    print(f"  Max: {results_df['C_score'].max():.4f}")
    
    print("\n" + "="*80)
    print("✓ FASE 3 COMPLETE")
    print("="*80)
    
    return results_df


if __name__ == "__main__":
    """
    Run inference as standalone script
    
    Usage:
        python src/inference/inference_scoring.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CoWePS full inference")
    parser.add_argument('--base-config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--ae-config', type=str, default='configs/autoencoder_config.yaml')
    args = parser.parse_args()
    
    # Run inference
    results_df = run_full_inference(
        base_config_path=args.base_config,
        ae_config_path=args.ae_config
    )
    
    print("\n✓ Inference complete!")
    print(f"✓ Results: data/scores/full_inference_results.csv")