# utils.py - FIXED VERSION
"""
Utility Functions for CoWePS Pipeline - TIER 1 & 2 ENHANCED

Enhanced utilities with:
- Comprehensive metrics visualization
- Advanced logging and monitoring
- Checkpoint management improvements
- Training stability tracking
- Publication-quality figure generation
"""

import os
import yaml
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None
from typing import Dict, List, Optional, Any, Tuple
import torch
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LOGGING SETUP - ENHANCED
# ============================================================================

def setup_logging(log_dir: str, phase_name: str, config: Dict) -> logging.Logger:
    """
    Setup logging configuration for a specific phase
    
    Args:
        log_dir: Directory to save log files
        phase_name: Name of the current phase
        config: Configuration dictionary
    
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{phase_name}_{timestamp}.log')
    
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_format = config.get('logging', {}).get('format', 
                                                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(phase_name)
    logger.setLevel(getattr(logging, log_level))
    
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized for {phase_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


class MetricsLogger:
    """Enhanced logger for tracking training metrics"""
    
    def __init__(self, log_dir: str, model_name: str):
        """
        Initialize metrics logger
        
        Args:
            log_dir: Directory to save metrics
            model_name: Name of the model
        """
        self.log_dir = log_dir
        self.model_name = model_name
        self.metrics_file = os.path.join(log_dir, f'{model_name}_metrics.json')
        
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'train_balanced_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_balanced_acc': [],
            'val_f1_per_class': [],
            'learning_rates': [],
            'gradient_norms': [],
            'epoch_times': []
        }
    
    def log_epoch(self, epoch: int, metrics: Dict):
        """Log metrics for one epoch"""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def save(self):
        """Save metrics to JSON file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def load(self):
        """Load metrics from JSON file"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        return self.metrics


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config(config_path: str = 'base_config.yaml') -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config.yaml file
    
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file: {e}")


def create_directories(config: Dict) -> None:
    """
    Create all necessary directories from config
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    
    os.makedirs(paths.get('processed_dir', 'data/processed'), exist_ok=True)
    os.makedirs(paths.get('scores_dir', 'data/scores'), exist_ok=True)
    os.makedirs(paths.get('final_dir', 'data/final'), exist_ok=True)
    os.makedirs(paths.get('checkpoints_dir', 'models/checkpoints'), exist_ok=True)
    os.makedirs(paths.get('figures_dir', 'outputs/figures'), exist_ok=True)
    os.makedirs(paths.get('reports_dir', 'outputs/reports'), exist_ok=True)
    os.makedirs(paths.get('logs_dir', 'outputs/logs'), exist_ok=True)


# ============================================================================
# FILE I/O OPERATIONS
# ============================================================================

def save_dataframe(df: pd.DataFrame, filepath: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Save DataFrame to both CSV and pickle format
    
    Args:
        df: DataFrame to save
        filepath: Base filepath (without extension)
        logger: Logger instance (optional)
    """
    csv_path = f"{filepath}.csv"
    df.to_csv(csv_path, index=False)
    if logger:
        logger.info(f"Saved CSV: {csv_path} ({len(df)} rows)")
    
    pkl_path = f"{filepath}.pkl"
    df.to_pickle(pkl_path)
    if logger:
        logger.info(f"Saved pickle: {pkl_path}")


def load_dataframe(filepath: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load DataFrame from pickle (faster) or CSV (fallback)
    
    Smart path handling: accepts both with and without extension
    - Input: 'data/manifest' or 'data/manifest.csv' → both work!
    
    Args:
        filepath: Filepath (with or without .csv/.pkl extension)
        logger: Logger instance (optional)
    
    Returns:
        Loaded DataFrame
    """
    # Smart extension handling
    base_path = filepath
    
    # Remove extension if present
    if filepath.endswith('.csv'):
        base_path = filepath[:-4]  # Remove '.csv'
    elif filepath.endswith('.pkl'):
        base_path = filepath[:-4]  # Remove '.pkl'
    
    # Try pickle first (faster)
    pkl_path = f"{base_path}.pkl"
    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
        if logger:
            logger.info(f"Loaded from pickle: {pkl_path} ({len(df)} rows)")
        return df
    
    # Fallback to CSV
    csv_path = f"{base_path}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if logger:
            logger.info(f"Loaded from CSV: {csv_path} ({len(df)} rows)")
        return df
    
    # If still not found, also try the original path as-is (edge case)
    if os.path.exists(filepath):
        if filepath.endswith('.pkl'):
            df = pd.read_pickle(filepath)
            if logger:
                logger.info(f"Loaded from pickle: {filepath} ({len(df)} rows)")
            return df
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            if logger:
                logger.info(f"Loaded from CSV: {filepath} ({len(df)} rows)")
            return df
    
    raise FileNotFoundError(
        f"File not found. Tried:\n"
        f"  - {pkl_path}\n"
        f"  - {csv_path}\n"
        f"  - {filepath}"
    )

def save_json(data: Dict, filepath: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        logger: Logger instance (optional)
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    if logger:
        logger.info(f"Saved JSON: {filepath}")


def load_json(filepath: str, logger: Optional[logging.Logger] = None) -> Dict:
    """
    Load JSON file
    
    Args:
        filepath: JSON file path
        logger: Logger instance (optional)
    
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if logger:
        logger.info(f"Loaded JSON: {filepath}")
    
    return data


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def get_device(config: Dict, logger: Optional[logging.Logger] = None) -> torch.device:
    """
    Get PyTorch device (CUDA or CPU)
    
    Args:
        config: Configuration dictionary
        logger: Logger instance (optional)
    
    Returns:
        PyTorch device
    """
    device_name = config.get('device', 'cuda')
    
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        if logger:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        if logger:
            if device_name == 'cuda':
                logger.warning("CUDA not available, using CPU")
            else:
                logger.info("Using CPU")
    
    return device


def set_random_seed(seed: int, logger: Optional[logging.Logger] = None) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
        logger: Logger instance (optional)
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if logger:
        logger.info(f"Random seed set to: {seed}")


# ============================================================================
# BASIC VISUALIZATION UTILITIES (FOR BACKWARD COMPATIBILITY)
# ============================================================================

def plot_distribution(data: pd.Series, title: str, xlabel: str, ylabel: str,
                     save_path: Optional[str] = None, 
                     logger: Optional[logging.Logger] = None) -> None:
    """
    Plot distribution as bar chart (BACKWARD COMPATIBLE)
    
    Args:
        data: Pandas Series with value counts
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (optional)
        logger: Logger instance (optional)
    """
    plt.figure(figsize=(10, 6))
    
    ax = data.plot(kind='bar', color='steelblue', edgecolor='black', linewidth=1.5)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    
    for i, v in enumerate(data.values):
        ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Saved figure: {save_path}")
    
    plt.close()


def plot_histogram(data: np.ndarray, title: str, xlabel: str, 
                   bins: int = 50, save_path: Optional[str] = None,
                   logger: Optional[logging.Logger] = None) -> None:
    """
    Plot histogram
    
    Args:
        data: Numpy array of values
        title: Plot title
        xlabel: X-axis label
        bins: Number of bins
        save_path: Path to save figure (optional)
        logger: Logger instance (optional)
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(data, bins=bins, color='green', alpha=0.7, edgecolor='black')
    
    mean_val = np.mean(data)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_val:.3f}')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Saved figure: {save_path}")
    
    plt.close()


# ============================================================================
# ENHANCED VISUALIZATION UTILITIES
# ============================================================================

def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str],
                         title: str = 'Confusion Matrix',
                         save_path: Optional[str] = None,
                         logger: Optional[logging.Logger] = None,
                         normalize: bool = True) -> None:
    """
    Plot confusion matrix with enhanced visualization
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure (optional)
        logger: Logger instance (optional)
        normalize: Whether to normalize the matrix
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
    else:
        conf_matrix_norm = conf_matrix
    
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                square=True, linewidths=0.5)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Saved confusion matrix: {save_path}")
    
    plt.close()


def plot_training_history(history: Dict, model_name: str,
                          save_path: Optional[str] = None,
                          logger: Optional[logging.Logger] = None) -> None:
    """
    Plot comprehensive training history
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
        save_path: Path to save figure
        logger: Logger instance
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontweight='bold')
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Balanced Accuracy
    if 'train_balanced_acc' in history and len(history['train_balanced_acc']) > 0:
        axes[0, 2].plot(epochs, history['train_balanced_acc'], 'b-', label='Train BA', linewidth=2)
        axes[0, 2].plot(epochs, history['val_balanced_acc'], 'r-', label='Val BA', linewidth=2)
        axes[0, 2].set_xlabel('Epoch', fontweight='bold')
        axes[0, 2].set_ylabel('Balanced Accuracy (%)', fontweight='bold')
        axes[0, 2].set_title('Balanced Accuracy Curves', fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontweight='bold')
    axes[1, 0].set_ylabel('Learning Rate', fontweight='bold')
    axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)
    
    # F1 Scores per Class
    if 'val_f1_per_class' in history and len(history['val_f1_per_class']) > 0:
        f1_data = np.array(history['val_f1_per_class'])
        num_classes = f1_data.shape[1]
        for i in range(num_classes):
            axes[1, 1].plot(epochs, f1_data[:, i], label=f'Grade {i}', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontweight='bold')
        axes[1, 1].set_ylabel('F1 Score', fontweight='bold')
        axes[1, 1].set_title('F1 Score per Class', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    # Gradient Norms (if available)
    if 'gradient_norms' in history and len(history['gradient_norms']) > 0:
        axes[1, 2].plot(epochs, history['gradient_norms'], 'purple', linewidth=2)
        axes[1, 2].axhline(y=10, color='r', linestyle='--', label='Warning Threshold')
        axes[1, 2].set_xlabel('Epoch', fontweight='bold')
        axes[1, 2].set_ylabel('Gradient Norm', fontweight='bold')
        axes[1, 2].set_title('Gradient Magnitude', fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Saved training history: {save_path}")
    
    plt.close()


def plot_class_distribution(data: pd.Series, title: str,
                           save_path: Optional[str] = None,
                           logger: Optional[logging.Logger] = None) -> None:
    """
    Plot class distribution with enhanced styling
    
    Args:
        data: Pandas Series with value counts
        title: Plot title
        save_path: Path to save figure
        logger: Logger instance
    """
    plt.figure(figsize=(12, 7))
    
    colors = sns.color_palette("husl", len(data))
    ax = data.plot(kind='bar', color=colors, edgecolor='black', linewidth=1.5)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    for i, (idx, v) in enumerate(data.items()):
        percentage = 100 * v / data.sum()
        ax.text(i, v, f'{v}\n({percentage:.1f}%)', 
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Saved distribution plot: {save_path}")
    
    plt.close()


def plot_per_class_metrics(metrics: Dict, class_names: List[str],
                          save_path: Optional[str] = None,
                          logger: Optional[logging.Logger] = None) -> None:
    """
    Plot per-class precision, recall, F1 scores
    
    Args:
        metrics: Dictionary with per-class metrics
        class_names: List of class names
        save_path: Path to save figure
        logger: Logger instance
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    
    x = np.arange(len(class_names))
    width = 0.6
    
    # Precision
    if 'precision' in metrics:
        axes[0].bar(x, metrics['precision'], width, color='skyblue', edgecolor='black')
        axes[0].set_ylabel('Precision', fontweight='bold')
        axes[0].set_title('Precision per Class', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_names)
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(metrics['precision']):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Recall
    if 'recall' in metrics:
        axes[1].bar(x, metrics['recall'], width, color='lightcoral', edgecolor='black')
        axes[1].set_ylabel('Recall', fontweight='bold')
        axes[1].set_title('Recall per Class', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_names)
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(metrics['recall']):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # F1 Score
    if 'f1' in metrics:
        axes[2].bar(x, metrics['f1'], width, color='lightgreen', edgecolor='black')
        axes[2].set_ylabel('F1 Score', fontweight='bold')
        axes[2].set_title('F1 Score per Class', fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(class_names)
        axes[2].set_ylim([0, 1.1])
        axes[2].grid(axis='y', alpha=0.3)
        for i, v in enumerate(metrics['f1']):
            axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Saved per-class metrics: {save_path}")
    
    plt.close()


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Enhanced checkpoint management for model training"""
    
    def __init__(self, checkpoint_dir: str, model_name: str, 
                 keep_best_n: int = 3, logger: Optional[logging.Logger] = None):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Name of the model
            keep_best_n: Number of best checkpoints to keep
            logger: Logger instance
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.keep_best_n = keep_best_n
        self.logger = logger
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.checkpoints = []
    
    def save_checkpoint(self, epoch: int, model_state: Dict, 
                       optimizer_state: Dict, metric: float,
                       is_best: bool = False) -> str:
        """
        Save checkpoint with metadata
        
        Args:
            epoch: Current epoch
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            metric: Metric value (for ranking)
            is_best: Whether this is the best checkpoint
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'metric': metric,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'{self.model_name}_epoch_{epoch}.pth'
        )
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir,
                f'best_{self.model_name}.pth'
            )
            torch.save(checkpoint, best_path)
            if self.logger:
                self.logger.info(f"Saved best checkpoint: {best_path}")
        
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'metric': metric
        })
        
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Keep only the best N checkpoints"""
        if len(self.checkpoints) <= self.keep_best_n:
            return
        
        self.checkpoints.sort(key=lambda x: x['metric'], reverse=True)
        
        to_remove = self.checkpoints[self.keep_best_n:]
        
        for ckpt in to_remove:
            if os.path.exists(ckpt['path']):
                os.remove(ckpt['path'])
                if self.logger:
                    self.logger.info(f"Removed old checkpoint: {ckpt['path']}")
        
        self.checkpoints = self.checkpoints[:self.keep_best_n]
    
    def load_best_checkpoint(self) -> Optional[Dict]:
        """Load the best checkpoint"""
        best_path = os.path.join(
            self.checkpoint_dir,
            f'best_{self.model_name}.pth'
        )
        
        if not os.path.exists(best_path):
            if self.logger:
                self.logger.warning(f"No best checkpoint found: {best_path}")
            return None
        
        checkpoint = torch.load(best_path)
        if self.logger:
            self.logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load specific checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        if self.logger:
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return checkpoint


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total: int, description: str, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize progress tracker
        
        Args:
            total: Total number of items
            description: Description of operation
            logger: Logger instance (optional)
        """
        self.total = total
        self.description = description
        self.logger = logger
        self.current = 0
        self.start_time = datetime.now()
        
        if logger:
            logger.info(f"Starting: {description} (Total: {total})")
    
    def update(self, n: int = 1) -> None:
        """
        Update progress
        
        Args:
            n: Number of items completed
        """
        self.current += n
        
        if self.logger and self.current % max(1, self.total // 10) == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            
            self.logger.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({100*self.current/self.total:.1f}%) - "
                f"Rate: {rate:.1f} items/s - "
                f"ETA: {remaining:.0f}s"
            )
    
    def finish(self) -> None:
        """Mark operation as complete"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.logger:
            self.logger.info(
                f"Completed: {self.description} - "
                f"Total time: {elapsed:.1f}s - "
                f"Average rate: {self.total/elapsed:.1f} items/s"
            )


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_paths(config: Dict, logger: Optional[logging.Logger] = None) -> bool:
    """
    Validate that required paths exist
    
    Args:
        config: Configuration dictionary
        logger: Logger instance (optional)
    
    Returns:
        True if all paths valid, False otherwise
    """
    paths = config.get('paths', {})
    data_root = paths.get('data_root', 'data/raw')
    
    if not os.path.exists(data_root):
        if logger:
            logger.error(f"Data root directory not found: {data_root}")
        return False
    
    required_grades = [0, 1, 2, 3, 4, 5]
    missing_folders = []
    
    for grade in required_grades:
        grade_folder = os.path.join(data_root, str(grade))
        if not os.path.exists(grade_folder):
            missing_folders.append(grade)
    
    if missing_folders:
        if logger:
            logger.error(f"Missing grade folders: {missing_folders}")
        return False
    
    if logger:
        logger.info("All required paths validated successfully")
    
    return True


# ============================================================================
# STATISTICS UTILITIES
# ============================================================================

def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics
    
    Args:
        data: Numpy array
    
    Returns:
        Dictionary with statistics
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'q10': float(np.percentile(data, 10)),
        'q90': float(np.percentile(data, 90))
    }


def print_statistics(stats: Dict[str, float], name: str, 
                    logger: logging.Logger) -> None:
    """
    Print statistics in formatted way
    
    Args:
        stats: Statistics dictionary
        name: Name of the metric
        logger: Logger instance
    """
    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Mean:   {stats['mean']:.4f}")
    logger.info(f"  Std:    {stats['std']:.4f}")
    logger.info(f"  Min:    {stats['min']:.4f}")
    logger.info(f"  Max:    {stats['max']:.4f}")
    logger.info(f"  Median: {stats['median']:.4f}")
    logger.info(f"  Q25:    {stats['q25']:.4f}")
    logger.info(f"  Q75:    {stats['q75']:.4f}")


def calculate_class_balance_metrics(class_counts: np.ndarray) -> Dict[str, float]:
    """
    Calculate class balance metrics
    
    Args:
        class_counts: Array of counts per class
    
    Returns:
        Dictionary with balance metrics
    """
    sorted_counts = np.sort(class_counts)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
    
    cv = np.std(class_counts) / np.mean(class_counts)
    
    metrics = {
        'gini_coefficient': float(gini),
        'coefficient_of_variation': float(cv),
        'min_count': int(np.min(class_counts)),
        'max_count': int(np.max(class_counts)),
        'mean_count': float(np.mean(class_counts)),
        'imbalance_ratio': float(np.max(class_counts) / np.min(class_counts))
    }
    
    return metrics


# ============================================================================
# CALIBRATION METRICS - V2.4 FASE 5
# ============================================================================

def calculate_ece(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Calculate Expected Calibration Error (ECE)
    
    ECE measures the difference between predicted confidence and actual accuracy.
    Lower ECE indicates better calibration.
    
    Args:
        logits: Raw model outputs (N, num_classes) - will be converted to probabilities
        labels: True class labels (N,)
        n_bins: Number of bins for calibration calculation
    
    Returns:
        ECE score (float) - lower is better, range [0, 1]
    
    Reference:
        Guo et al. (2017) "On Calibration of Modern Neural Networks"
    """
    # Convert logits to probabilities using softmax
    if logits.ndim == 2:  # Multi-class
        from scipy.special import softmax
        probs = softmax(logits, axis=1)
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:  # Binary or pre-computed probabilities
        confidences = logits
        predictions = (logits > 0.5).astype(int)
    
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Calculate accuracy per sample
    accuracies = (predictions == labels).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Average confidence and accuracy in this bin
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            # Add weighted difference to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def calculate_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Brier Score for multi-class classification
    
    Brier Score measures the mean squared difference between predicted 
    probabilities and actual outcomes. Lower is better.
    
    Args:
        probs: Predicted probabilities (N, num_classes) - already softmax-ed
        labels: True class labels (N,)
    
    Returns:
        Brier score (float) - lower is better, range [0, 2] for multi-class
    
    Reference:
        Brier (1950) "Verification of Forecasts Expressed in Terms of Probability"
    """
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Handle both pre-softmaxed probs and logits
    if probs.ndim == 2:
        # If max value > 1, assume these are logits and need softmax
        if np.max(probs) > 1.0:
            from scipy.special import softmax
            probs = softmax(probs, axis=1)
    else:
        raise ValueError("probs must be 2D array (N, num_classes)")
    
    # Create one-hot encoded labels
    num_classes = probs.shape[1]
    one_hot_labels = np.zeros_like(probs)
    one_hot_labels[np.arange(len(labels)), labels] = 1
    
    # Calculate Brier Score
    # BS = (1/N) * sum((p_i - y_i)^2) for all classes and samples
    brier_score = np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1))
    
    return float(brier_score)


# ============================================================================
# REPORT GENERATION UTILITIES
# ============================================================================

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: List[str]) -> str:
    """
    Generate comprehensive classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Report string
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   digits=3)
    return report


def save_training_summary(model_name: str, history: Dict, 
                         final_metrics: Dict, save_dir: str,
                         logger: Optional[logging.Logger] = None) -> str:
    """
    Save comprehensive training summary
    
    Args:
        model_name: Name of the model
        history: Training history
        final_metrics: Final evaluation metrics
        save_dir: Directory to save summary
        logger: Logger instance
    
    Returns:
        Path to saved summary
    """
    summary = {
        'model_name': model_name,
        'training_completed': datetime.now().isoformat(),
        'total_epochs': len(history['train_loss']),
        'best_epoch': int(np.argmax(history['val_balanced_acc'])) + 1 if 'val_balanced_acc' in history else 0,
        'best_val_balanced_acc': float(np.max(history['val_balanced_acc'])) if 'val_balanced_acc' in history else 0.0,
        'final_metrics': final_metrics,
        'history': history
    }
    
    summary_path = os.path.join(save_dir, f'{model_name}_summary.json')
    save_json(summary, summary_path, logger)
    
    return summary_path


if __name__ == "__main__":
    print("Testing CoWePS Enhanced Utilities...")
    
    try:
        config = load_config('base_config.yaml')
        print("âœ“ Config loaded successfully")
    except Exception as e:
        print(f"âœ— Config loading failed: {e}")
    
    logger = setup_logging('outputs/logs', 'test_utils', config)
    logger.info("Test log message")
    print("âœ“ Logging works")
    
    device = get_device(config, logger)
    print(f"âœ“ Device: {device}")
    
    print("\nAll enhanced utility tests passed!")