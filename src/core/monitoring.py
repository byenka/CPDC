"""
Training Health Monitoring Utilities
=====================================
Comprehensive monitoring tools to prevent training failures

Features:
- Real-time gradient health monitoring
- Per-class performance tracking
- Early failure detection
- Training diagnostics and visualization

Author: CoWePS Fixed Implementation
"""

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# GRADIENT HEALTH MONITOR
# ============================================================================

class GradientHealthMonitor:
    """
    Monitor gradient health to prevent NaN and explosions
    """
    
    def __init__(self, model: nn.Module, logger=None,
             warn_threshold: float = 10.0, crit_threshold: float = 100.0):
        """
        Initialize gradient monitor
        
        Args:
            model: PyTorch model to monitor
            logger: Logger instance
        """
        self.model = model
        self.logger = logger
        self.warn_threshold = float(warn_threshold)
        self.crit_threshold = float(crit_threshold)
        self.gradient_history = defaultdict(list)
        self.explosion_count = 0
        self.nan_count = 0
    
    def check_gradients(self) -> Dict[str, float]:
        """
        Check gradient health across all layers
        
        Returns:
            Dictionary with gradient statistics
        """
        total_norm = 0.0
        layer_norms = {}
        has_nan = False
        has_inf = False
        max_grad = 0.0
        min_grad = float('inf')
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Check for NaN/Inf
                if torch.isnan(grad).any():
                    has_nan = True
                    if self.logger:
                        self.logger.error(f"ðŸ”´ NaN gradient in {name}")
                
                if torch.isinf(grad).any():
                    has_inf = True
                    if self.logger:
                        self.logger.error(f"ðŸ”´ Inf gradient in {name}")
                
                # Calculate norm
                param_norm = grad.norm(2).item()
                total_norm += param_norm ** 2
                layer_norms[name] = param_norm
                
                # Track max/min
                max_grad = max(max_grad, param_norm)
                min_grad = min(min_grad, param_norm) if param_norm > 0 else min_grad
        
        total_norm = total_norm ** 0.5
        
        # Update counters
        if has_nan:
            self.nan_count += 1
        if total_norm > 10.0:
            self.explosion_count += 1
        
        # Store history
        self.gradient_history['total_norm'].append(total_norm)
        self.gradient_history['max_grad'].append(max_grad)
        
        stats = {
            'total_norm': total_norm,
            'max_grad': max_grad,
            'min_grad': min_grad,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'explosion_count': self.explosion_count,
            'nan_count': self.nan_count,
            'gradient_ratio': max_grad / (min_grad + 1e-8)
        }
        
        # Alert on issues
        if total_norm > self.warn_threshold:
            if self.logger:
                self.logger.warning(f"âš ï¸  Large gradient norm: {total_norm:.2f}")

        if total_norm > self.crit_threshold:
            if self.logger:
                self.logger.error(f"ðŸ”´ CRITICAL: Gradient explosion! Norm={total_norm:.2f}")

        
        return stats
    
    def get_layer_gradient_stats(self) -> pd.DataFrame:
        """
        Get detailed gradient statistics per layer
        
        Returns:
            DataFrame with layer-wise gradient info
        """
        layer_data = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                layer_data.append({
                    'layer': name,
                    'grad_mean': grad.mean().item(),
                    'grad_std': grad.std().item(),
                    'grad_norm': grad.norm(2).item(),
                    'grad_max': grad.max().item(),
                    'grad_min': grad.min().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item()
                })
        
        return pd.DataFrame(layer_data)
    
    def plot_gradient_history(self, save_path: Optional[str] = None):
        """
        Plot gradient norm history
        
        Args:
            save_path: Path to save figure
        """
        if not self.gradient_history['total_norm']:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Total norm over time
        axes[0].plot(self.gradient_history['total_norm'])
        axes[0].axhline(y=10, color='orange', linestyle='--', label='Warning (10)')
        axes[0].axhline(y=100, color='red', linestyle='--', label='Critical (100)')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Gradient Norm')
        axes[0].set_title('Gradient Norm History')
        axes[0].legend()
        axes[0].set_yscale('log')
        
        # Distribution of norms
        axes[1].hist(self.gradient_history['total_norm'], bins=50, edgecolor='black')
        axes[1].axvline(x=10, color='orange', linestyle='--', label='Warning')
        axes[1].axvline(x=100, color='red', linestyle='--', label='Critical')
        axes[1].set_xlabel('Gradient Norm')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Gradient Norm Distribution')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()


# ============================================================================
# CLASS BALANCE MONITOR
# ============================================================================

class ClassBalanceMonitor:
    """
    Monitor class-wise performance and detect learning failures
    """
    
    def __init__(self, num_classes: int = 5, logger=None):
        """
        Initialize class balance monitor
        
        Args:
            num_classes: Number of classes
            logger: Logger instance
        """
        self.num_classes = num_classes
        self.logger = logger
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.labels = []
        self.batch_distributions = []
    
    def update(self, predictions: np.ndarray, labels: np.ndarray):
        """
        Update with batch predictions
        
        Args:
            predictions: Predicted classes
            labels: True labels
        """
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        
        # Track batch distribution
        batch_dist = np.bincount(labels, minlength=self.num_classes)
        self.batch_distributions.append(batch_dist)
    
    def get_class_metrics(self) -> pd.DataFrame:
        """
        Calculate per-class metrics
        
        Returns:
            DataFrame with per-class performance
        """
        if not self.predictions:
            return pd.DataFrame()
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        class_data = []
        
        for c in range(self.num_classes):
            # Binary classification for this class
            y_true = (labels == c).astype(int)
            y_pred = (predictions == c).astype(int)
            
            # Calculate metrics
            support = y_true.sum()
            
            if support > 0:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
            else:
                precision = recall = f1 = 0.0
            
            # Prediction count
            pred_count = (predictions == c).sum()
            
            class_data.append({
                'class': c,
                'support': support,
                'predictions': pred_count,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'learning': f1 > 0.01  # Is class being learned?
            })
        
        df = pd.DataFrame(class_data)
        
        # Add status column
        df['status'] = df.apply(
            lambda x: 'âœ…' if x['f1_score'] > 0.3 
            else 'âš ï¸' if x['f1_score'] > 0.01 
            else 'ðŸ”´', 
            axis=1
        )
        
        return df
    
    def check_learning_health(self) -> Dict[str, Any]:
        """
        Check if model is learning all classes properly
        
        Returns:
            Dictionary with health status
        """
        if not self.predictions:
            return {'healthy': True, 'issues': []}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        issues = []
        
        # Check 1: Is model predicting all classes?
        predicted_classes = np.unique(predictions)
        if len(predicted_classes) < self.num_classes:
            missing = set(range(self.num_classes)) - set(predicted_classes)
            issues.append(f"Not predicting classes: {missing}")
        
        # Check 2: Extreme class imbalance in predictions?
        pred_dist = np.bincount(predictions, minlength=self.num_classes)
        pred_pct = pred_dist / len(predictions)
        
        if pred_pct.max() > 0.8:
            dominant_class = pred_pct.argmax()
            issues.append(f"Over-predicting class {dominant_class} ({pred_pct[dominant_class]:.1%})")
        
        # Check 3: Any class with zero F1?
        class_metrics = self.get_class_metrics()
        zero_f1_classes = class_metrics[class_metrics['f1_score'] == 0]['class'].tolist()
        
        if zero_f1_classes:
            issues.append(f"Zero F1 for classes: {zero_f1_classes}")
        
        # Check 4: Balanced accuracy
        ba = balanced_accuracy_score(labels, predictions)
        if ba < 0.25:
            issues.append(f"Very low balanced accuracy: {ba:.3f}")
        
        # Overall health
        healthy = len(issues) == 0
        
        if not healthy and self.logger:
            self.logger.warning("âš ï¸  Learning health issues detected:")
            for issue in issues:
                self.logger.warning(f"   - {issue}")
        
        return {
            'healthy': healthy,
            'issues': issues,
            'balanced_accuracy': ba,
            'prediction_distribution': pred_pct.tolist()
        }
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save figure
        """
        if not self.predictions:
            return
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   cbar=True, ax=ax1, square=True)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title('Confusion Matrix (Counts)')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   cbar=True, ax=ax2, square=True)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title('Confusion Matrix (Normalized)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()


# ============================================================================
# BATCH STATISTICS TRACKER
# ============================================================================

class BatchStatisticsTracker:
    """
    Track batch-level statistics for debugging
    """
    
    def __init__(self, num_classes: int = 5):
        """
        Initialize batch tracker
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset statistics"""
        self.batch_sizes = []
        self.batch_class_counts = []
        self.batch_losses = []
        self.batch_accuracies = []
        self.batch_gradient_norms = []
    
    def update(self, batch_size: int, class_counts: np.ndarray, 
              loss: float, accuracy: float, gradient_norm: float = None):
        """
        Update with batch statistics
        
        Args:
            batch_size: Size of batch
            class_counts: Count per class in batch
            loss: Batch loss
            accuracy: Batch accuracy
            gradient_norm: Gradient norm
        """
        self.batch_sizes.append(batch_size)
        self.batch_class_counts.append(class_counts)
        self.batch_losses.append(loss)
        self.batch_accuracies.append(accuracy)
        
        if gradient_norm is not None:
            self.batch_gradient_norms.append(gradient_norm)
    
    def get_batch_balance_stats(self) -> Dict[str, float]:
        """
        Calculate batch balance statistics
        
        Returns:
            Dictionary with balance metrics
        """
        if not self.batch_class_counts:
            return {}
        
        # Calculate entropy for each batch (measure of balance)
        entropies = []
        
        for counts in self.batch_class_counts:
            total = counts.sum()
            if total > 0:
                probs = counts / total
                # Add small epsilon to avoid log(0)
                probs = np.clip(probs, 1e-10, 1.0)
                entropy = -np.sum(probs * np.log(probs))
                entropies.append(entropy)
        
        # Perfect balance entropy
        perfect_entropy = -np.log(1.0 / self.num_classes)
        
        # Calculate stats
        avg_entropy = np.mean(entropies)
        balance_score = avg_entropy / perfect_entropy  # 1.0 = perfect balance
        
        # Count perfectly balanced batches
        perfect_batches = 0
        for counts in self.batch_class_counts:
            if len(np.unique(counts)) == 1 and counts[0] > 0:
                perfect_batches += 1
        
        return {
            'avg_batch_entropy': avg_entropy,
            'balance_score': balance_score,
            'perfect_batches': perfect_batches,
            'total_batches': len(self.batch_class_counts),
            'perfect_batch_ratio': perfect_batches / len(self.batch_class_counts)
        }


# ============================================================================
# EARLY STOPPING MONITOR
# ============================================================================

class EarlyStoppingMonitor:
    """
    Advanced early stopping with multiple criteria
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 min_f1_per_class: float = 0.1, logger=None):
        """
        Initialize early stopping monitor
        
        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum improvement
            min_f1_per_class: Minimum F1 required for all classes
            logger: Logger instance
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_f1_per_class = min_f1_per_class
        self.logger = logger
        
        self.best_score = 0.0
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
    
    def check(self, val_ba: float, per_class_f1: List[float], epoch: int) -> bool:
        """
        Check if should stop training
        
        Args:
            val_ba: Validation balanced accuracy
            per_class_f1: F1 score per class
            epoch: Current epoch
        
        Returns:
            True if should stop
        """
        # Check minimum F1 constraint
        if min(per_class_f1) < self.min_f1_per_class and epoch > 10:
            if self.logger:
                self.logger.warning(
                    f"âš ï¸  Some classes below minimum F1 threshold ({self.min_f1_per_class})"
                )
        
        # Check improvement
        if val_ba > self.best_score + self.min_delta:
            # Improvement!
            self.best_score = val_ba
            self.best_epoch = epoch
            self.counter = 0
            
            if self.logger:
                self.logger.info(f"âœ… New best model! Val BA: {val_ba:.3f}")
        else:
            # No improvement
            self.counter += 1
            
            if self.logger:
                self.logger.info(
                    f"No improvement for {self.counter} epochs "
                    f"(best: {self.best_score:.3f} at epoch {self.best_epoch})"
                )
        
        # Check if should stop
        if self.counter >= self.patience:
            self.should_stop = True
            
            if self.logger:
                self.logger.info(f"â¹ Early stopping triggered!")
        
        return self.should_stop


# ============================================================================
# TRAINING DIAGNOSTICS REPORT
# ============================================================================

def generate_training_report(history: Dict, model_name: str, 
                            save_path: Optional[str] = None) -> str:
    """
    Generate comprehensive training report
    
    Args:
        history: Training history dictionary
        model_name: Name of model
        save_path: Path to save report
    
    Returns:
        Report as string
    """
    report = f"""
{'='*80}
TRAINING DIAGNOSTICS REPORT
Model: {model_name}
{'='*80}

"""
    
    # Stage 1 Summary
    if 'stage1_history' in history:
        stage1 = history['stage1_history']
        
        report += "STAGE 1: FORCE LEARNING ALL CLASSES\n"
        report += "-" * 40 + "\n\n"
        
        # Final metrics
        if stage1['val']:
            final_val = stage1['val'][-1]
            
            report += f"Final Stage 1 Metrics:\n"
            report += f"  Balanced Accuracy: {final_val['balanced_accuracy']:.3f}\n"
            report += f"  Min Class F1: {final_val['min_class_f1']:.3f}\n"
            report += f"  All Classes Learning: {final_val['all_classes_learning']}\n\n"
            
            # Per-class F1
            report += "Per-Class F1 Scores:\n"
            for i, f1 in enumerate(final_val['per_class_f1']):
                status = "âœ…" if f1 > 0.3 else "âš ï¸" if f1 > 0.01 else "ðŸ”´"
                report += f"  Grade {i}: {f1:.3f} {status}\n"
            
            report += "\n"
    
    # Stage 2 Summary
    if 'stage2_history' in history:
        stage2 = history['stage2_history']
        
        report += "STAGE 2: OPTIMIZE PERFORMANCE\n"
        report += "-" * 40 + "\n\n"
        
        # Final metrics
        if stage2['val']:
            final_val = stage2['val'][-1]
            
            report += f"Final Stage 2 Metrics:\n"
            report += f"  Balanced Accuracy: {final_val['balanced_accuracy']:.3f}\n"
            report += f"  Overall Accuracy: {final_val['accuracy']:.3f}\n"
            report += f"  Average Loss: {final_val['avg_loss']:.4f}\n\n"
            
            # Per-class performance
            report += "Final Per-Class F1 Scores:\n"
            for i, f1 in enumerate(final_val['per_class_f1']):
                status = "âœ…" if f1 > 0.5 else "âš ï¸" if f1 > 0.3 else "ðŸ”´"
                report += f"  Grade {i}: {f1:.3f} {status}\n"
    
    # Best overall
    report += "\n" + "="*80 + "\n"
    report += f"BEST VALIDATION BA: {history.get('best_val_ba', 0.0):.3f}\n"
    report += "="*80 + "\n"
    
    # Save if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report


# ============================================================================
# PLOT TRAINING CURVES
# ============================================================================

def plot_training_curves(history: Dict, model_name: str, 
                         save_path: Optional[str] = None):
    """
    Plot comprehensive training curves
    
    Args:
        history: Training history
        model_name: Model name
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Combine stage histories
    train_ba = []
    val_ba = []
    train_loss = []
    val_loss = []
    
    if 'stage1_history' in history:
        for t, v in zip(history['stage1_history']['train'], 
                       history['stage1_history']['val']):
            train_ba.append(t['balanced_accuracy'])
            val_ba.append(v['balanced_accuracy'])
            train_loss.append(t['avg_loss'])
            val_loss.append(v['avg_loss'])
    
    if 'stage2_history' in history:
        for t, v in zip(history['stage2_history']['train'],
                       history['stage2_history']['val']):
            train_ba.append(t['balanced_accuracy'])
            val_ba.append(v['balanced_accuracy'])
            train_loss.append(t['avg_loss'])
            val_loss.append(v['avg_loss'])
    
    epochs = range(1, len(train_ba) + 1)
    
    # Plot 1: Balanced Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_ba, 'b-', label='Train BA')
    plt.plot(epochs, val_ba, 'r-', label='Val BA')
    
    if 'stage1_history' in history:
        stage1_end = len(history['stage1_history']['train'])
        plt.axvline(x=stage1_end, color='gray', linestyle='--', alpha=0.5)
        plt.text(stage1_end, 0.5, 'Stage 2', rotation=90)
    
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'{model_name} - Balanced Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    
    if 'stage1_history' in history:
        plt.axvline(x=stage1_end, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Per-Class F1 Evolution
    plt.subplot(2, 2, 3)
    
    if 'stage2_history' in history and history['stage2_history']['val']:
        # Extract per-class F1 over time
        for class_idx in range(5):
            class_f1 = []
            
            if 'stage1_history' in history:
                for v in history['stage1_history']['val']:
                    class_f1.append(v['per_class_f1'][class_idx])
            
            for v in history['stage2_history']['val']:
                class_f1.append(v['per_class_f1'][class_idx])
            
            plt.plot(range(1, len(class_f1)+1), class_f1, 
                    label=f'Grade {class_idx}')
        
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Per-Class F1 Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Gradient Health
    plt.subplot(2, 2, 4)
    
    if 'stage2_history' in history and history['stage2_history']['train']:
        grad_norms = []
        
        if 'stage1_history' in history:
            for t in history['stage1_history']['train']:
                grad_norms.append(t.get('avg_gradient_norm', 0))
        
        for t in history['stage2_history']['train']:
            grad_norms.append(t.get('avg_gradient_norm', 0))
        
        plt.plot(range(1, len(grad_norms)+1), grad_norms)
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Safe')
        plt.axhline(y=10.0, color='orange', linestyle='--', alpha=0.5, label='Warning')
        plt.axhline(y=100.0, color='red', linestyle='--', alpha=0.5, label='Critical')
        
        plt.xlabel('Epoch')
        plt.ylabel('Average Gradient Norm')
        plt.title('Gradient Health')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} Training Diagnostics', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Training Health Monitoring Utilities")
    print("="*80)
    
    print("\nAvailable monitors:")
    print("  âœ… GradientHealthMonitor - Prevent gradient explosions")
    print("  âœ… ClassBalanceMonitor - Track per-class learning")
    print("  âœ… BatchStatisticsTracker - Monitor batch balance")
    print("  âœ… EarlyStoppingMonitor - Advanced stopping criteria")
    
    print("\nDiagnostic functions:")
    print("  âœ… generate_training_report() - Comprehensive report")
    print("  âœ… plot_training_curves() - Visualization")
    
    print("\nMonitoring utilities loaded successfully!")