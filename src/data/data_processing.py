import os
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

# ---------------------------
# Helper parsing (robust)
# ---------------------------
def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _to_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)

def _to_tuple2(x, default=(0.0, 0.0)):
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return (float(x[0]), float(x[1]))
    return default

def _odd_kernel(k: int) -> int:
    k = int(k)
    if k <= 0:
        return 3
    return k if (k % 2 == 1) else (k + 1)

class DRDataset(Dataset):
    def __init__(self, manifest_path, config, mode='train'):
        self.manifest = pd.read_csv(manifest_path)
        self.config = config
        self.mode = mode

        train_cfg = (config.get('training', {}) or {})

        # Label column resolution (keep consistent with train_model.py)
        # Priority: explicit config -> label -> weak_label_class -> grade
        preferred = train_cfg.get('label_column')
        if isinstance(preferred, str) and preferred.strip():
            preferred = preferred.strip()
            if preferred not in self.manifest.columns:
                raise ValueError(
                    f"Configured training.label_column='{preferred}' not found in manifest columns: {list(self.manifest.columns)}"
                )
            self.label_col = preferred
        else:
            if 'label' in self.manifest.columns:
                self.label_col = 'label'
            elif 'weak_label_class' in self.manifest.columns:
                self.label_col = 'weak_label_class'
            elif 'grade' in self.manifest.columns:
                self.label_col = 'grade'
            else:
                raise ValueError(
                    f"Manifest has no supported label column. Expected one of ['label','weak_label_class','grade'] but got: {list(self.manifest.columns)}"
                )

        # Guard: if both common label columns exist, they must match.
        if 'label' in self.manifest.columns and 'weak_label_class' in self.manifest.columns:
            a = pd.to_numeric(self.manifest['label'], errors='coerce')
            b = pd.to_numeric(self.manifest['weak_label_class'], errors='coerce')
            mismatch = (a.notna() & b.notna() & (a.astype('Int64') != b.astype('Int64'))).sum()
            if mismatch > 0:
                raise ValueError(
                    f"Manifest label mismatch: {mismatch} rows have different values between 'label' and 'weak_label_class'. "
                    "Set training.label_column explicitly or fix the manifest."
                )
        self.image_size = _to_int(train_cfg.get('image_size', config.get('image_size', 512)), 512)

        norm_cfg = (config.get('normalization', {}) or {})
        self.normalize_mean = train_cfg.get('normalize_mean', norm_cfg.get('mean', [0.485, 0.456, 0.406]))
        self.normalize_std = train_cfg.get('normalize_std', norm_cfg.get('std', [0.229, 0.224, 0.225]))

        # Augmentation config (from YAML)
        aug = (config.get('augmentation', {}) or {})
        self.use_augmentation = (mode == 'train')

        # Geometric aug params (used in __getitem__)
        self.p_hflip = _to_float(aug.get('horizontal_flip_prob', 0.5), 0.5)
        self.p_vflip = _to_float(aug.get('vertical_flip_prob', 0.0), 0.0)
        self.rot_deg = _to_float(aug.get('rotation_degrees', 0.0), 0.0)

        # ---------------------------
        # Build image transform (photometric)
        # ---------------------------
        t_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]

        if self.use_augmentation:
            # Color jitter from YAML (if present)
            cj = (aug.get('color_jitter', {}) or {})
            if len(cj) > 0:
                t_list.append(
                    transforms.ColorJitter(
                        brightness=_to_float(cj.get('brightness', 0.0), 0.0),
                        contrast=_to_float(cj.get('contrast', 0.0), 0.0),
                        saturation=_to_float(cj.get('saturation', 0.0), 0.0),
                        hue=_to_float(cj.get('hue', 0.0), 0.0),
                    )
                )

        t_list.append(transforms.ToTensor())

        if self.use_augmentation:
            # Gaussian blur from YAML (probabilistic)
            gb = (aug.get('gaussian_blur', {}) or {})
            gb_p = _to_float(gb.get('probability', 0.0), 0.0)
            if gb_p > 0.0:
                k = _odd_kernel(_to_int(gb.get('kernel_size', 3), 3))
                sigma = gb.get('sigma', (0.1, 1.0))
                sigma = _to_tuple2(sigma, default=(0.1, 1.0))
                blur = transforms.GaussianBlur(kernel_size=k, sigma=sigma)
                t_list.append(transforms.RandomApply([blur], p=gb_p))

        # Normalize always
        t_list.append(transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std))

        if self.use_augmentation:
            # Random erasing from YAML (tensor domain) — only active if probability > 0
            re_cfg = (aug.get('random_erasing', {}) or {})
            re_p = _to_float(re_cfg.get('probability', 0.0), 0.0)
            if re_p > 0.0:
                re_scale = _to_tuple2(re_cfg.get('scale', (0.02, 0.15)), default=(0.02, 0.15))
                re_ratio = _to_tuple2(re_cfg.get('ratio', (0.3, 3.3)), default=(0.3, 3.3))
                re_value = re_cfg.get('value', 0)
                t_list.append(
                    transforms.RandomErasing(
                        p=re_p,
                        scale=re_scale,
                        ratio=re_ratio,
                        value=re_value
                    )
                )

        self.image_transform = transforms.Compose(t_list)

        # Mask transform (no photometric aug)
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        image_path = row['image_path']
        mask_path = row.get('mask_path', '')
        # Ensure label is an int (pandas may parse as float/object)
        label_raw = row.get(self.label_col, 0)
        try:
            label = int(label_raw)
        except Exception:
            label = int(_to_int(label_raw, 0))
        source = row.get('source', 'NA')

        image = Image.open(image_path).convert('RGB')

        mask = None
        if isinstance(mask_path, str) and mask_path.strip() and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')

        # ---------------------------
        # Geometric augmentation (train only)
        # ---------------------------
        if self.use_augmentation:
            if torch.rand(1).item() < self.p_hflip:
                image = F.hflip(image)
                if mask is not None:
                    mask = F.hflip(mask)

            if torch.rand(1).item() < self.p_vflip:
                image = F.vflip(image)
                if mask is not None:
                    mask = F.vflip(mask)

            if self.rot_deg > 0:
                angle = random.uniform(-self.rot_deg, self.rot_deg)
                image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
                if mask is not None:
                    mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        # Photometric transforms
        image = self.image_transform(image)
        if mask is not None:
            mask = self.mask_transform(mask)
        else:
            # Use an all-ones mask when missing so masking is a no-op
            mask = torch.ones((1, self.image_size, self.image_size), dtype=image.dtype)

        # Apply mask (broadcast 1xHxW -> 3xHxW)
        image_masked = image * mask

        # Backward-compatible return format for existing trainers/evaluators
        meta = {
            'mask': mask,
            'source': source,
            'image_path': image_path,
        }
        return image_masked, torch.tensor(label, dtype=torch.long), meta
