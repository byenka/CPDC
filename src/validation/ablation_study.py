"""
CoWePS V2.4 Ablation Study Module
Fase 5: Scientific Validation via Ablation Study

This module conducts rigorous ablation experiments to empirically prove
the value of CoWePS methodology by comparing 4 different training datasets:
1. Baseline (Random sampling - patient-disjoint)
2. C-Only (Confidence score only)
3. Q-Only (Quality score only)  
4. CoWePS Full (Both C and Q scores combined)

All experiments use:
- Same test set (gold_standard_validate.csv)
- Same model architecture (e.g., ConvNeXt)
- Same training protocol
- Calibration metrics (ECE, Brier Score) - V2.4

Author: CoWePS V2.4 Implementation
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processing import DRDataset
from src.models.model_factory import create_model_from_config
from src.models.trainer_base import TwoStageTrainer
from src.core.utils import (
    load_config,
    setup_logging,
    calculate_ece,
    calculate_brier_score
)
from torch.utils.data import DataLoader


# ============================================================================
# DATASET CREATION - V2.4 PATIENT-DISJOINT
# ============================================================================

def create_ablation_datasets(config: Dict, 
                            scores_df: pd.DataFrame,
                            master_list_df: pd.DataFrame, 
                            test_df: pd.DataFrame,
                            logger) -> Dict[str, str]:
    """
    Create 4 ablation dataset manifests with patient-disjoint baseline
    
    Critical V2.4 requirement: Baseline must be patient-disjoint from test set
    to prevent data leakage.
    
    Args:
        config: Configuration dictionary
        scores_df: Scored candidates from Fase 3 (full_inference_results.csv)
        master_list_df: Master list with patient_id from Fase 0
        test_df: Test set manifest (gold_standard_validate.csv)
        logger: Logger instance
    
    Returns:
        Dictionary mapping dataset names to file paths
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING ABLATION DATASETS")
    logger.info("="*80)
    
    TARGET_PER_CLASS = config['sampling'].get('target_per_class', 1038)
    NUM_CLASSES = config.get('model', {}).get('num_classes', 5)
    processed_dir = config['paths']['processed_dir']
    
    os.makedirs(processed_dir, exist_ok=True)
    
    dataset_paths = {}
    
    # ========================================================================
    # DATASET 1: BASELINE (Random Sampling - Patient-Disjoint)
    # ========================================================================
    logger.info("\n--- Creating Baseline Dataset (Patient-Disjoint) ---")
    
    # Step 1: Get patient IDs in test set
    if 'patient_id' not in test_df.columns:
        logger.error("❌ Test set missing 'patient_id' column!")
        logger.error("Cannot ensure patient-disjoint baseline. Aborting.")
        raise ValueError("Test set must have 'patient_id' column for anti-leakage")
    
    test_patient_ids = set(test_df['patient_id'].unique())
    logger.info(f"Test set contains {len(test_patient_ids)} unique patients")
    
    # Step 2: Remove all images from test patients from master_list
    if 'patient_id' not in master_list_df.columns:
        logger.error("❌ Master list missing 'patient_id' column!")
        raise ValueError("Master list must have 'patient_id' column")
    
    master_list_safe = master_list_df[~master_list_df['patient_id'].isin(test_patient_ids)].copy()
    
    logger.info(f"Master list: {len(master_list_df)} images")
    logger.info(f"After removing test patients: {len(master_list_safe)} images")
    logger.info(f"Removed: {len(master_list_df) - len(master_list_safe)} images")
    
    # Step 3: Filter to valid classes and sample
    valid_classes = master_list_safe['weak_label_class'].isin(range(NUM_CLASSES))
    master_list_safe = master_list_safe[valid_classes]
    
    target_total = TARGET_PER_CLASS * NUM_CLASSES
    
    if len(master_list_safe) < target_total:
        logger.warning(f"⚠️  Not enough samples! Available: {len(master_list_safe)}, Target: {target_total}")
        logger.warning("Using all available samples for baseline")
        baseline_df = master_list_safe
    else:
        # Stratified sampling
        try:
            baseline_df, _ = train_test_split(
                master_list_safe,
                train_size=target_total,
                stratify=master_list_safe['weak_label_class'],
                random_state=42
            )
        except ValueError as e:
            logger.warning(f"Stratified sampling failed: {e}")
            logger.warning("Using random sampling instead")
            baseline_df = master_list_safe.sample(n=target_total, random_state=42)
    
    # Save baseline
    baseline_path = os.path.join(processed_dir, 'ablation_baseline.csv')
    baseline_df.to_csv(baseline_path, index=False)
    dataset_paths['baseline'] = baseline_path
    
    logger.info(f"✓ Baseline dataset created: {len(baseline_df)} samples")
    logger.info(f"  Class distribution:")
    for class_id in range(NUM_CLASSES):
        count = (baseline_df['weak_label_class'] == class_id).sum()
        logger.info(f"    Class {class_id}: {count}")
    
    # ========================================================================
    # DATASET 2: C-ONLY (Confidence Score Only)
    # ========================================================================
    logger.info("\n--- Creating C-Only Dataset ---")
    
    c_only_list = []
    for class_id in range(NUM_CLASSES):
        # Filter by predicted class
        class_candidates = scores_df[scores_df['Pred_Class'] == class_id].copy()
        
        # Sort by C_score descending
        class_candidates_sorted = class_candidates.sort_values(
            by='C_score', 
            ascending=False
        )
        
        # Select top TARGET_PER_CLASS
        selection = class_candidates_sorted.head(TARGET_PER_CLASS)
        c_only_list.append(selection)
        
        logger.info(f"  Class {class_id}: Selected {len(selection)}/{TARGET_PER_CLASS}")
    
    c_only_df = pd.concat(c_only_list, ignore_index=True)
    c_only_path = os.path.join(processed_dir, 'ablation_c_only.csv')
    c_only_df.to_csv(c_only_path, index=False)
    dataset_paths['c_only'] = c_only_path
    
    logger.info(f"✓ C-Only dataset created: {len(c_only_df)} samples")
    
    # ========================================================================
    # DATASET 3: Q-ONLY (Quality Score Only)
    # ========================================================================
    logger.info("\n--- Creating Q-Only Dataset ---")
    
    # Filter to samples that passed quality check
    q_passed_df = scores_df[scores_df['Q_score'] == 1].copy()
    logger.info(f"Quality-passed candidates: {len(q_passed_df)}")
    
    q_only_list = []
    for class_id in range(NUM_CLASSES):
        class_candidates = q_passed_df[q_passed_df['Pred_Class'] == class_id].copy()
        
        # Random sample if more than target, otherwise take all
        if len(class_candidates) > TARGET_PER_CLASS:
            selection = class_candidates.sample(n=TARGET_PER_CLASS, random_state=42)
        else:
            selection = class_candidates
            if len(selection) < TARGET_PER_CLASS:
                logger.warning(f"  ⚠️  Class {class_id}: Only {len(selection)}/{TARGET_PER_CLASS} available")
        
        q_only_list.append(selection)
        logger.info(f"  Class {class_id}: Selected {len(selection)}/{TARGET_PER_CLASS}")
    
    q_only_df = pd.concat(q_only_list, ignore_index=True)
    q_only_path = os.path.join(processed_dir, 'ablation_q_only.csv')
    q_only_df.to_csv(q_only_path, index=False)
    dataset_paths['q_only'] = q_only_path
    
    logger.info(f"✓ Q-Only dataset created: {len(q_only_df)} samples")
    
    # ========================================================================
    # DATASET 4: COWEPS FULL (From Fase 4)
    # ========================================================================
    logger.info("\n--- CoWePS Full Dataset ---")
    
    coweps_path = os.path.join(config['paths']['final_dir'], 'coweps_final_dataset.csv')
    if not os.path.exists(coweps_path):
        logger.error(f"❌ CoWePS full dataset not found: {coweps_path}")
        logger.error("Please run Fase 4 first!")
        raise FileNotFoundError(f"CoWePS dataset not found: {coweps_path}")
    
    dataset_paths['coweps_full'] = coweps_path
    coweps_df = pd.read_csv(coweps_path)
    logger.info(f"✓ CoWePS Full dataset: {len(coweps_df)} samples")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("DATASET CREATION COMPLETE")
    logger.info("="*80)
    for name, path in dataset_paths.items():
        df = pd.read_csv(path)
        logger.info(f"  {name:15s}: {len(df):5d} samples → {path}")
    
    return dataset_paths


# ============================================================================
# SINGLE EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(train_manifest_path: str,
                         test_manifest_path: str,
                         experiment_name: str,
                         config: Dict,
                         logger) -> Dict:
    """
    Run single ablation experiment: train model and evaluate
    
    Args:
        train_manifest_path: Path to training manifest CSV
        test_manifest_path: Path to test manifest CSV
        experiment_name: Name of the experiment
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Dictionary with experiment results including calibration metrics
    """
    logger.info("\n" + "="*80)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info("="*80)
    
    # ========================================================================
    # 1. Load Test Model Configuration
    # ========================================================================
    # Use a standard model for all experiments (e.g., ConvNeXt)
    test_model_config_path = config.get('ablation', {}).get('test_model_config_path', 
                                                             'configs/efficient_config.yaml')
    
    logger.info(f"Test model config: {test_model_config_path}")
    
    if not os.path.exists(test_model_config_path):
        logger.error(f"❌ Model config not found: {test_model_config_path}")
        raise FileNotFoundError(f"Model config not found: {test_model_config_path}")
    
    with open(test_model_config_path, 'r') as f:
        test_model_config = yaml.safe_load(f)
    
    # ========================================================================
    # 2. Create Model
    # ========================================================================
    logger.info("Creating model...")
    model = create_model_from_config(test_model_config)
    
    # ========================================================================
    # 3. Load Datasets
    # ========================================================================
    logger.info(f"Loading training data from: {train_manifest_path}")
    train_df = pd.read_csv(train_manifest_path)
    logger.info(f"Training samples: {len(train_df)}")
    
    logger.info(f"Loading test data from: {test_manifest_path}")
    test_df = pd.read_csv(test_manifest_path)
    logger.info(f"Test samples: {len(test_df)}")
    
    # ========================================================================
    # 4. Create DataLoaders
    # ========================================================================
    # Note: DRDataset will handle image loading and preprocessing (512x512 + masking)
    from torch.utils.data import DataLoader
    
    batch_size = config.get('training', {}).get('batch_size', 16)
    
    # For ablation, we use simple validation split from train
    train_split_df, val_split_df = train_test_split(
        train_df, 
        test_size=0.15, 
        stratify=train_df.get('weak_label_class', train_df.get('Pred_Class')),
        random_state=42
    )
    
    logger.info(f"Train split: {len(train_split_df)}, Val split: {len(val_split_df)}")
    
    # ========================================================================
    # 5. Train Model
    # ========================================================================
    logger.info("Starting training...")
    
    # Merge configs
    trainer_config = {**config, **test_model_config}
    trainer_config['model_name'] = f"ablation_{experiment_name}"
    
    # Create trainer
    trainer = TwoStageTrainer(
        model=model,
        config=trainer_config
    )
    
    # Run training
    try:
        training_results = trainer.train_full_pipeline(
            train_manifest=train_split_df,
            val_manifest=val_split_df
        )
        logger.info(f"Training completed: Best Val BA = {training_results.get('best_val_ba', 0):.3f}")
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # ========================================================================
    # 6. Evaluate on Test Set
    # ========================================================================
    logger.info("Evaluating on test set...")
    
    # Create test loader
    test_dataset = DRDataset(manifest_path=test_manifest_path, mode='val')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate with raw outputs for calibration metrics
    metrics, all_logits, all_probs, all_labels = trainer.evaluate(
        test_loader, 
        return_raw=True
    )
    
    # ========================================================================
    # 7. Calculate Calibration Metrics (V2.4)
    # ========================================================================
    logger.info("Calculating calibration metrics...")
    
    metrics['ece'] = calculate_ece(all_logits, all_labels)
    metrics['brier_score'] = calculate_brier_score(all_probs, all_labels)
    
    # Add experiment metadata
    metrics['experiment_name'] = experiment_name
    metrics['train_samples'] = len(train_split_df)
    metrics['test_samples'] = len(test_df)
    
    # ========================================================================
    # 8. Log Results
    # ========================================================================
    logger.info("\n" + "-"*80)
    logger.info(f"RESULTS: {experiment_name}")
    logger.info("-"*80)
    logger.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  F1 Score (macro):  {metrics['f1_score']:.4f}")
    logger.info(f"  ECE:               {metrics['ece']:.5f}")
    logger.info(f"  Brier Score:       {metrics['brier_score']:.5f}")
    logger.info("-"*80)
    
    return metrics


# ============================================================================
# MAIN ABLATION STUDY ORCHESTRATOR
# ============================================================================

def run_ablation_study(config_path: str) -> Dict:
    """
    Run complete ablation study with 4 experiments
    
    Args:
        config_path: Path to base configuration file
    
    Returns:
        Dictionary with all experiment results
    """
    # ========================================================================
    # 1. Load Configuration and Setup
    # ========================================================================
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(
        config['paths']['logs_dir'],
        'ablation_study_v24',
        config
    )
    
    logger.info("\n" + "="*80)
    logger.info("FASE 5: COWEPS V2.4 ABLATION STUDY")
    logger.info("Scientific Validation via Comparative Experiments")
    logger.info("="*80)
    logger.info(f"Configuration: {config_path}")
    
    # ========================================================================
    # 2. Load Supporting Data
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("LOADING SUPPORTING DATA")
    logger.info("="*80)
    
    # Scores from Fase 3
    scores_path = os.path.join(config['paths']['scores_dir'], 'full_inference_results.csv')
    if not os.path.exists(scores_path):
        logger.error(f"❌ Scores file not found: {scores_path}")
        logger.error("Please run Fase 3 (Inference & Scoring) first!")
        raise FileNotFoundError(f"Scores file not found: {scores_path}")
    
    scores_df = pd.read_csv(scores_path)
    logger.info(f"✓ Loaded scores: {len(scores_df)} candidates")
    
    # Master list from Fase 0
    master_list_path = os.path.join(config['paths']['processed_dir'], 'master_list.csv')
    if not os.path.exists(master_list_path):
        logger.error(f"❌ Master list not found: {master_list_path}")
        raise FileNotFoundError(f"Master list not found: {master_list_path}")
    
    master_list_df = pd.read_csv(master_list_path)
    logger.info(f"✓ Loaded master list: {len(master_list_df)} images")
    
    # Test set (gold standard validation)
    test_path = os.path.join(config['paths']['processed_dir'], 'gold_standard_validate.csv')
    if not os.path.exists(test_path):
        logger.error(f"❌ Test set not found: {test_path}")
        logger.error("Please run Fase 0 (Data Processing) first!")
        raise FileNotFoundError(f"Test set not found: {test_path}")
    
    test_df = pd.read_csv(test_path)
    logger.info(f"✓ Loaded test set: {len(test_df)} images")
    
    # ========================================================================
    # 3. Create Ablation Datasets
    # ========================================================================
    dataset_paths = create_ablation_datasets(
        config=config,
        scores_df=scores_df,
        master_list_df=master_list_df,
        test_df=test_df,
        logger=logger
    )
    
    # ========================================================================
    # 4. Run 4 Experiments
    # ========================================================================
    all_results = []
    
    experiments = [
        ('baseline', 'Baseline (Random Patient-Disjoint)'),
        ('c_only', 'C-Only (Confidence)'),
        ('q_only', 'Q-Only (Quality)'),
        ('coweps_full', 'CoWePS Full (C + Q)')
    ]
    
    for dataset_key, experiment_name in experiments:
        try:
            metrics = run_single_experiment(
                train_manifest_path=dataset_paths[dataset_key],
                test_manifest_path=test_path,
                experiment_name=experiment_name,
                config=config,
                logger=logger
            )
            all_results.append(metrics)
        except Exception as e:
            logger.error(f"❌ Experiment '{experiment_name}' failed: {str(e)}")
            logger.error("Continuing with remaining experiments...")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # 5. Save Results
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)
    
    results_df = pd.DataFrame(all_results)
    
    # Save as CSV
    reports_dir = config['paths'].get('reports_dir', 'outputs/reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    csv_path = os.path.join(reports_dir, 'ablation_study_results.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Results saved to: {csv_path}")
    
    # Save as Markdown
    md_path = os.path.join(reports_dir, 'ablation_study_results.md')
    
    # Create markdown report
    with open(md_path, 'w') as f:
        f.write("# CoWePS V2.4 Ablation Study Results\n\n")
        f.write("## Experiment Comparison\n\n")
        
        # Select key metrics for table
        key_metrics = [
            'experiment_name', 
            'accuracy', 
            'balanced_accuracy', 
            'f1_score', 
            'ece', 
            'brier_score'
        ]
        
        results_subset = results_df[key_metrics].copy()
        
        # Format numbers
        for col in ['accuracy', 'balanced_accuracy', 'f1_score']:
            results_subset[col] = results_subset[col].apply(lambda x: f"{x:.4f}")
        for col in ['ece', 'brier_score']:
            results_subset[col] = results_subset[col].apply(lambda x: f"{x:.5f}")
        
        # Write table
        f.write(results_subset.to_markdown(index=False))
        f.write("\n\n")
        
        # Add interpretation
        f.write("## Interpretation\n\n")
        f.write("**Lower ECE and Brier Score indicate better calibration.**\n\n")
        f.write("- **Baseline**: Random sampling (patient-disjoint from test set)\n")
        f.write("- **C-Only**: Selected by confidence score only\n")
        f.write("- **Q-Only**: Selected by quality score only\n")
        f.write("- **CoWePS Full**: Combined C and Q score selection\n\n")
        
        # Find best performer
        best_ba_idx = results_df['balanced_accuracy'].idxmax()
        best_exp = results_df.loc[best_ba_idx, 'experiment_name']
        best_ba = results_df.loc[best_ba_idx, 'balanced_accuracy']
        
        f.write(f"**Best Balanced Accuracy**: {best_exp} ({best_ba:.4f})\n\n")
        
        # Calibration analysis
        best_ece_idx = results_df['ece'].idxmin()
        best_ece_exp = results_df.loc[best_ece_idx, 'experiment_name']
        best_ece = results_df.loc[best_ece_idx, 'ece']
        
        f.write(f"**Best Calibration (ECE)**: {best_ece_exp} ({best_ece:.5f})\n")
    
    logger.info(f"✓ Markdown report saved to: {md_path}")
    
    # ========================================================================
    # 6. Final Summary
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("FASE 5 COMPLETE - ABLATION STUDY SUMMARY")
    logger.info("="*80)
    
    summary_cols = ['experiment_name', 'balanced_accuracy', 'f1_score', 'ece', 'brier_score']
    logger.info("\n" + results_df[summary_cols].to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("✅ FASE 5 COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    return {
        'success': True,
        'results': all_results,
        'results_csv': csv_path,
        'results_md': md_path
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CoWePS V2.4 Ablation Study')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to base configuration file'
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CoWePS V2.4 Ablation Study")
    print("Fase 5: Scientific Validation")
    print("="*80 + "\n")
    
    try:
        results = run_ablation_study(args.config)
        
        if results['success']:
            print("\n✅ Ablation study completed successfully!")
            print(f"Results saved to:")
            print(f"  - {results['results_csv']}")
            print(f"  - {results['results_md']}")
        else:
            print("\n❌ Ablation study failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)