"""
CoWePS V2.4 Progressive Sampling Module
Fase 4: Filter & Sort Sampling Pipeline

Philosophy: "Quality over Quantity"
- NO arbitrary Tier logic
- Simple binary Q_score filter (pass/fail)
- Sort only by C_score (confidence)
- Select top TARGET_PER_CLASS per class

Input:  data/scores/full_inference_results.csv (from Fase 3)
Output: data/final/coweps_final_dataset.csv

Author: CoWePS V2.4 Implementation
"""

import os
import sys
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.utils import (
    load_config,
    setup_logging,
)


# ============================================================================
# MAIN PROGRESSIVE SAMPLING FUNCTION - V2.4
# ============================================================================

def run_progressive_sampling(config_path: str = 'configs/base_config.yaml') -> Dict:
    """
    Run CoWePS V2.4 sampling pipeline (Filter & Sort)
    
    Pipeline:
    1. Filter: Keep only Q_score == 1 (passed quality check)
    2. Sort: Order by C_score (confidence) descending per class
    3. Select: Take top TARGET_PER_CLASS samples for each class
    
    Args:
        config_path: Path to configuration file or config dict
    
    Returns:
        Results dictionary with success status and metadata
    """
    # ========================================================================
    # STEP 1: Load Configuration
    # ========================================================================
    if isinstance(config_path, dict):
        config = config_path
        config_file = 'config_dict'
    else:
        config = load_config(config_path)
        config_file = config_path
    
    # Setup logging
    logger = setup_logging(
        config['paths']['logs_dir'],
        'progressive_sampling_v24',
        config
    )
    
    try:
        logger.info("\n" + "="*80)
        logger.info("FASE 4: COWEPS V2.4 PROGRESSIVE SAMPLING")
        logger.info("Filter & Sort Pipeline")
        logger.info("="*80)
        logger.info(f"Config: {config_file}")
        
        # Extract configuration parameters
        scores_dir = config['paths']['scores_dir']
        final_dir = config['paths']['final_dir']
        TARGET_PER_CLASS = config['sampling'].get('target_per_class', 1038)
        NUM_CLASSES = config.get('model', {}).get('num_classes', 5)
        
        # Input file from Fase 3
        scores_file = os.path.join(scores_dir, 'full_inference_results.csv')
        
        # Output file for Fase 4
        final_output_file = os.path.join(final_dir, 'coweps_final_dataset.csv')
        
        # Ensure output directory exists
        os.makedirs(final_dir, exist_ok=True)
        
        logger.info(f"\nConfiguration:")
        logger.info(f"  Input:  {scores_file}")
        logger.info(f"  Output: {final_output_file}")
        logger.info(f"  Target per class: {TARGET_PER_CLASS}")
        logger.info(f"  Number of classes: {NUM_CLASSES}")
        
        # ====================================================================
        # STEP 2: Load Scoring Results from Fase 3
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("LOADING FASE 3 RESULTS")
        logger.info("="*80)
        
        if not os.path.exists(scores_file):
            error_msg = f"ERROR: Scores file not found: {scores_file}"
            logger.error(error_msg)
            logger.error("Please run Fase 3 (Inference & Scoring) first!")
            return {
                'success': False,
                'error': error_msg
            }
        
        try:
            df = pd.read_csv(scores_file)
            logger.info(f"✓ Loaded {len(df)} scored candidates")
        except Exception as e:
            error_msg = f"Failed to load scores file: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
        
        # Validate required columns
        required_cols = ['image_path', 'C_score', 'Pred_Class', 'Q_score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
        
        logger.info(f"✓ All required columns present")
        logger.info(f"\nInitial statistics:")
        logger.info(f"  Total candidates: {len(df)}")
        logger.info(f"  C_score range: [{df['C_score'].min():.3f}, {df['C_score'].max():.3f}]")
        logger.info(f"  Q_score distribution:")
        logger.info(f"    Pass (Q_score=1): {(df['Q_score']==1).sum()}")
        logger.info(f"    Fail (Q_score=0): {(df['Q_score']==0).sum()}")
        
        # ====================================================================
        # STEP 3: FILTER - Apply Quality Gate (Q_score == 1)
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 1: QUALITY FILTER")
        logger.info("="*80)
        logger.info("Filtering candidates with Q_score == 1 (passed quality check)")
        
        initial_count = len(df)
        df_passed = df[df['Q_score'] == 1].copy()
        rejected_count = initial_count - len(df_passed)
        
        logger.info(f"\nFilter results:")
        logger.info(f"  Rejected (Q_score=0): {rejected_count} ({100*rejected_count/initial_count:.1f}%)")
        logger.info(f"  Passed   (Q_score=1): {len(df_passed)} ({100*len(df_passed)/initial_count:.1f}%)")
        
        if len(df_passed) == 0:
            error_msg = "No candidates passed quality filter!"
            logger.error(f"\n❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
        
        # Show passed candidates by class
        logger.info(f"\nPassed candidates by class:")
        for class_id in range(NUM_CLASSES):
            class_passed = len(df_passed[df_passed['Pred_Class'] == class_id])
            logger.info(f"  Class {class_id}: {class_passed} candidates")
        
        # ====================================================================
        # STEP 4: SORT & SELECT - Per Class Selection
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 2: SORT & SELECT")
        logger.info("="*80)
        logger.info(f"Selecting top {TARGET_PER_CLASS} samples per class")
        logger.info(f"Sorting criteria: C_score (confidence) - descending")
        
        final_selection_list = []
        selection_summary = {}
        
        for class_id in range(NUM_CLASSES):
            logger.info(f"\n--- Processing Class {class_id} ---")
            
            # Isolate candidates for this class
            class_candidates = df_passed[df_passed['Pred_Class'] == class_id]
            logger.info(f"  Available candidates: {len(class_candidates)}")
            
            if len(class_candidates) == 0:
                logger.warning(f"  ⚠️  WARNING: No candidates for Class {class_id}!")
                selection_summary[class_id] = {
                    'selected': 0,
                    'available': 0,
                    'target': TARGET_PER_CLASS,
                    'shortage': TARGET_PER_CLASS
                }
                continue
            
            # ================================================================
            # CORE SORTING LOGIC V2.4
            # ================================================================
            # Sort ONLY by C_score (confidence) - highest first
            # No more multi-level sorting with Tier!
            sorted_candidates = class_candidates.sort_values(
                by=['C_score'],
                ascending=False
            )
            
            # Select top TARGET_PER_CLASS
            selection = sorted_candidates.head(TARGET_PER_CLASS)
            final_selection_list.append(selection)
            
            # Statistics
            selected_count = len(selection)
            shortage = max(0, TARGET_PER_CLASS - selected_count)
            
            logger.info(f"  Selected: {selected_count}/{TARGET_PER_CLASS}")
            if shortage > 0:
                logger.warning(f"  ⚠️  Shortage: {shortage} samples")
            
            logger.info(f"  C_score range: [{selection['C_score'].min():.3f}, {selection['C_score'].max():.3f}]")
            logger.info(f"  Mean C_score: {selection['C_score'].mean():.3f}")
            
            selection_summary[class_id] = {
                'selected': selected_count,
                'available': len(class_candidates),
                'target': TARGET_PER_CLASS,
                'shortage': shortage,
                'c_score_min': float(selection['C_score'].min()),
                'c_score_max': float(selection['C_score'].max()),
                'c_score_mean': float(selection['C_score'].mean())
            }
        
        # ====================================================================
        # STEP 5: Combine and Save Final Dataset
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 3: FINAL ASSEMBLY")
        logger.info("="*80)
        
        if not final_selection_list:
            error_msg = "No samples selected for any class!"
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
        
        final_dataset_df = pd.concat(final_selection_list, ignore_index=True)
        
        # Save to CSV
        final_dataset_df.to_csv(final_output_file, index=False)
        logger.info(f"✓ Saved final dataset to: {final_output_file}")
        
        # ====================================================================
        # STEP 6: Generate Statistics and Summary
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("FASE 4 COMPLETE - FINAL SUMMARY")
        logger.info("="*80)
        
        total_selected = len(final_dataset_df)
        total_target = TARGET_PER_CLASS * NUM_CLASSES
        total_shortage = total_target - total_selected
        
        logger.info(f"\n📊 Overall Statistics:")
        logger.info(f"  Total selected: {total_selected}/{total_target}")
        if total_shortage > 0:
            logger.warning(f"  Total shortage: {total_shortage}")
        logger.info(f"  Overall C_score: {final_dataset_df['C_score'].mean():.3f} ± {final_dataset_df['C_score'].std():.3f}")
        
        logger.info(f"\n📋 Class Distribution:")
        class_dist = final_dataset_df['Pred_Class'].value_counts().sort_index()
        for class_id, count in class_dist.items():
            shortage = TARGET_PER_CLASS - count
            status = "✓" if shortage == 0 else "⚠️"
            logger.info(f"  Class {class_id}: {count:4d}/{TARGET_PER_CLASS} {status}")
        
        # Prepare metadata
        metadata = {
            'total_samples': total_selected,
            'target_total': total_target,
            'target_per_class': TARGET_PER_CLASS,
            'total_shortage': total_shortage,
            'class_distribution': class_dist.to_dict(),
            'selection_summary': selection_summary,
            'overall_c_score_mean': float(final_dataset_df['C_score'].mean()),
            'overall_c_score_std': float(final_dataset_df['C_score'].std()),
            'overall_c_score_min': float(final_dataset_df['C_score'].min()),
            'overall_c_score_max': float(final_dataset_df['C_score'].max())
        }
        
        # Save metadata
        metadata_file = os.path.join(final_dir, 'coweps_final_metadata.json')
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved metadata to: {metadata_file}")
        
        logger.info("\n" + "="*80)
        logger.info("✅ FASE 4 COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return {
            'success': True,
            'final_path': final_output_file,
            'metadata': metadata,
            'total_samples': total_selected,
            'samples_per_class': dict(class_dist)
        }
        
    except Exception as e:
        logger.error(f"\n❌ Error in Fase 4 progressive sampling: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run progressive sampling as standalone script
    
    Usage:
        python src/sampling/progressive_sampling.py
        python src/sampling/progressive_sampling.py --config configs/base_config.yaml
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='CoWePS V2.4 Progressive Sampling')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CoWePS V2.4 Progressive Sampling Pipeline")
    print("Fase 4: Filter & Sort Selection")
    print("="*80 + "\n")
    
    results = run_progressive_sampling(args.config)
    
    if results['success']:
        print("\n✅ Progressive sampling completed successfully!")
        print(f"Total samples selected: {results['total_samples']}")
        print(f"Results saved to: {results['final_path']}")
        print(f"Check logs in: outputs/logs/")
    else:
        print("\n❌ Progressive sampling failed!")
        if 'error' in results:
            print(f"Error: {results['error']}")
        sys.exit(1)