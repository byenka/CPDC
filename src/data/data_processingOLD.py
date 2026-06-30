"""
CoWePS v2.4 Data Processing Module
Menyediakan DRDataset class untuk loading gambar dengan masking.

Author: CoWePS v2.4 Implementation Team
"""

import random
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class DRDataset(TorchDataset):
    """
    Dataset untuk Diabetic Retinopathy dengan implementasi masker (v2.4)
    
    KUNCI v2.4:
    - Input: 512x512 (WAJIB)
    - Menerapkan mask * image untuk menghilangkan background
    - Augmentasi disinkronkan antara image dan mask
    """
    def __init__(self, manifest_path: str, config: dict, mode: str = 'train'):
        """
        Initialize DRDataset
        
        Args:
            manifest_path: Path ke CSV manifest (dari Fase 0)
            config: Configuration dictionary
            mode: 'train', 'val', atau 'test'
        """
        self.manifest = pd.read_csv(manifest_path)
        self.config = config if isinstance(config, dict) else {}
        self.mode = mode
        self.image_size = 512  # WAJIB 512x512 untuk v2.4
        
        # Validasi manifest columns
        required_cols = ['image_path', 'mask_path', 'weak_label_class']
        for col in required_cols:
            if col not in self.manifest.columns:
                raise ValueError(f"Manifest missing required column: {col}")
        
        print(f"DRDataset initialized: {len(self.manifest)} samples ({mode} mode)")
        
        # === Transform konsisten (resize -> tensor -> normalize) ===
        # === Transform konsisten (resize -> tensor -> normalize) ===
        # Untuk menjaga sinkronisasi dengan mask, augmentasi GEOMETRIK (flip/rotasi)
        # tetap dilakukan di __getitem__ dengan F.hflip / F.rotate.
        # Di sini kita hanya tambahkan augmentasi FOTOMETRIK (ColorJitter + Blur) untuk mode 'train'.
        if self.mode == 'train':
            self.image_transform = transforms.Compose([
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.05
                ),
                transforms.ToTensor(),
                # Blur ringan; jangan terlalu agresif supaya detail lesi tidak hilang total
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            # Val/test: hanya resize + normalize, TANPA jitter/blur
            self.image_transform = transforms.Compose([
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        # Mask transformations (NEAREST untuk binary mask)
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()  # hasil (1,H,W) dengan nilai 0..1
        ])
        
        # Augmentation flag
        self.use_augmentation = (mode == 'train')
        
        # Ambil parameter augment dari config (fallback aman)
        aug = self.config.get('augmentation', {})
        self.p_hflip  = float(aug.get('horizontal_flip_prob', 0.5))
        # Fundus: default vflip 0.0 agar tidak merusak orientasi superior-inferior
        self.p_vflip  = float(aug.get('vertical_flip_prob', 0.0))
        self.rot_deg  = float(aug.get('rotation_degrees', 7.0))


    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        """
        Get item with masking applied
        
        Returns:
            image_masked: Tensor (3, 512, 512) - gambar yang sudah dimasker
            label: int - kelas DR (0-4)
        """
        # Load paths
        row = self.manifest.iloc[idx]
        img_path  = row['image_path']
        mask_path = row['mask_path']
        label     = int(row['weak_label_class'])
        
        # Load image and mask (PIL)
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")  # Grayscale
        
        # === SINKRONISASI AUGMENTASI (TRAIN SAJA) ===
        if self.use_augmentation:
            # 1) Horizontal flip
            if torch.rand(1).item() < self.p_hflip:
                image = F.hflip(image)
                mask  = F.hflip(mask)
            # 2) Vertical flip (opsional - default dimatikan)
            if self.p_vflip > 0.0 and torch.rand(1).item() < self.p_vflip:
                image = F.vflip(image)
                mask  = F.vflip(mask)
            # 3) Rotasi ringan ±rot_deg
            if self.rot_deg and self.rot_deg > 0:
                angle = random.uniform(-self.rot_deg, self.rot_deg)
                image = F.rotate(image, angle,
                                 interpolation=transforms.InterpolationMode.BICUBIC,
                                 expand=False)
                mask  = F.rotate(mask,  angle,
                                 interpolation=transforms.InterpolationMode.NEAREST,
                                 expand=False)

        # === Apply transformations ===
        image = self.image_transform(image)   # (3, 512, 512), normalized
        mask  = self.mask_transform(mask)     # (1, 512, 512), [0, 1]
        
        # KUNCI v2.4: Apply mask to image (broadcast 1xHxW -> 3xHxW)
        image_masked = image * mask
        
        return image_masked, label


# ============================================================================
# BACKWARD COMPATIBILITY (if needed for old scripts)
# ============================================================================
def run_data_processing(config_path='base_config.yaml'):
    """
    DEPRECATED: Use scripts/run_0_create_gold_standard.py instead
    
    This function is kept for backward compatibility only.
    """
    raise DeprecationWarning(
        "run_data_processing() is deprecated in v2.4. "
        "Please use: python scripts/run_0_create_gold_standard.py"
    )


if __name__ == "__main__":
    """
    Run data processing as standalone script
    
    Usage:
        python src/data_processing.py
    """
    print("\n" + "="*80)
    print("CoWePS Data Processing Pipeline")
    print("Phases 0, 1, 2: Setup, Manifest, Split Assignment")
    print("="*80 + "\n")
    try:
        run_data_processing('base_config.yaml')
    except DeprecationWarning as e:
        print(str(e))
