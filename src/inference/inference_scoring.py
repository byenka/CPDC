#!/usr/bin/env python
"""
CoWePS v2.5 - Inference & Scoring Module (AI Ganda)

FASE 3 (modular, tier-aware):
- Ensemble A (1..N models): C-score (confidence) dengan T-scaling (per-model)
- Ensemble B (Autoencoder, opsional): Q-score via reconstruction error + ROC (Youden J)
- Metrik tambahan: entropy, margin (dari probabilitas terkalibrasi)
- Output utama: full_inference_results_<mode>.csv
- Output tiering (jika diaktifkan di base_config): tier_A.csv, tier_B.csv, tier_C.csv

Kebijakan offline:
- ModLoader tidak melakukan unduhan; bobot .pth wajib lokal via config masing-masing model.
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

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import create_model_from_config
from src.models.model_autoencoder import UNet
from src.data.data_processing import DRDataset


# =============================================================================
# Small utilities
# =============================================================================

def _apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Apply temperature scaling (scalar T) to logits."""
    return logits / float(T)

def _probs_entropy_margin(probs: torch.Tensor):
    """
    Compute entropy and margin from probability tensor (B, C).
    Returns:
        entropy: (B,)  [-sum p log p]
        margin : (B,)  [p_top - p_second]
    """
    # numerical safety
    eps = 1e-8
    ent = -(probs * (probs + eps).log()).sum(dim=1)
    top2 = torch.topk(probs, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return ent, margin

def _resolve_manifest_for_mode(base_config, mode: str, logger=None) -> str:
    """
    Return CSV manifest path depending on mode:
      - master        : processed/master_list.csv
      - train         : manifests.train
      - val           : manifests.validate
      - gold_standard : concat(train, validate) -> temp CSV
    """
    processed = base_config['paths']['processed_dir']
    manifests = base_config.get('manifests', {})
    log = (logger.info if logger else None)

    def _emit(msg: str) -> None:
        if log is not None:
            log(msg)

    if mode == 'master':
        p = os.path.join(processed, 'master_list.csv')
        _emit(f"[manifest] mode=master -> {p} (source=paths.processed_dir/master_list.csv)")
        return p
    elif mode == 'train':
        p = manifests.get('train') or os.path.join(processed, 'gold_standard_train.csv')
        src = 'manifests.train' if manifests.get('train') else 'fallback processed/gold_standard_train.csv'
        _emit(f"[manifest] mode=train -> {p} (source={src})")
        return p
    elif mode == 'val':
        # Plan B: prefer explicit label-validation manifest (classification) if provided.
        p = manifests.get('validate_label') or manifests.get('validate') or os.path.join(processed, 'gold_standard_validate.csv')
        if manifests.get('validate_label'):
            src = 'manifests.validate_label'
        elif manifests.get('validate'):
            src = 'manifests.validate (legacy)'
        else:
            src = 'fallback processed/gold_standard_validate.csv'
        _emit(f"[manifest] mode=val -> {p} (source={src})")
        return p
    elif mode == 'gold_standard':
        train_csv = manifests.get('train') or os.path.join(processed, 'gold_standard_train.csv')
        val_csv   = manifests.get('validate_label') or manifests.get('validate') or os.path.join(processed, 'gold_standard_validate.csv')

        if manifests.get('train'):
            train_src = 'manifests.train'
        else:
            train_src = 'fallback processed/gold_standard_train.csv'

        if manifests.get('validate_label'):
            val_src = 'manifests.validate_label'
        elif manifests.get('validate'):
            val_src = 'manifests.validate (legacy)'
        else:
            val_src = 'fallback processed/gold_standard_validate.csv'

        _emit(f"[manifest] mode=gold_standard train -> {train_csv} (source={train_src})")
        _emit(f"[manifest] mode=gold_standard val   -> {val_csv} (source={val_src})")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            raise FileNotFoundError("gold_standard mode needs both train & validate CSVs")
        df = pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv)], ignore_index=True)
        tmp = '/tmp/gold_standard_concat.csv'
        df.to_csv(tmp, index=False)
        _emit(f"[manifest] mode=gold_standard concat -> {tmp}")
        return tmp
    else:
        raise ValueError(f"Unknown mode: {mode}")

# =============================================================================
# Autoencoder (Q-score) helpers
# =============================================================================

def get_reconstruction_error(ae_model, images, device):
    """
    Compute reconstruction error (MSE) per image in batch
    Returns numpy array shape (B,)
    """
    images = images.to(device)
    with torch.no_grad():
        reconstructed = ae_model(images)
        error_per_image = F.mse_loss(reconstructed, images, reduction='none').mean(dim=[1, 2, 3])
    return error_per_image.cpu().numpy()

def find_q_threshold_roc(ae_model, base_config, device, logger=None):
    """
    Find optimal Q-score threshold using ROC (Youden's J)
    Good = gold_standard_validate.csv ; Bad = paths.ungradable_folder
    """
    log = (logger.info if logger else print)
    log("\n" + "="*80)
    log("ROC CALIBRATION FOR Q-SCORE")
    log("="*80)

    ae_model.eval()

    all_errors, all_labels = [], []

    # 1) GOOD images
    log("\n1. Loading 'Good' quality images...")
    # Plan B: prefer explicit quality-validation manifest (GOOD set) if provided.
    manifests = base_config.get('manifests', {}) or {}
    good_manifest_path = (
        manifests.get('validate_quality')
        or manifests.get('validate')
        or os.path.join(base_config['paths']['processed_dir'], 'gold_standard_validate.csv')
    )
    if manifests.get('validate_quality'):
        good_src = 'manifests.validate_quality'
    elif manifests.get('validate'):
        good_src = 'manifests.validate (legacy)'
    else:
        good_src = 'fallback processed/gold_standard_validate.csv'
    log(f"   Good manifest path: {good_manifest_path} (source={good_src})")
    if not os.path.exists(good_manifest_path):
        raise FileNotFoundError(f"Good images manifest not found: {good_manifest_path}")

    good_dataset = DRDataset(good_manifest_path, base_config, mode='val')
    good_loader = DataLoader(
        good_dataset,
        batch_size=base_config.get('inference', {}).get('batch_size', 8),
        shuffle=False,
        num_workers=4
    )
    log(f"   Loaded: {len(good_dataset)} 'Good' images")

    # ⬇︎ PATCH: support batch dengan >2 elemen dari DRDataset
    for batch in tqdm(good_loader, desc="Processing 'Good' images"):
        # DRDataset bisa mengembalikan (image, label, meta, ...)
        if isinstance(batch, (list, tuple)):
            images = batch[0]          # elemen pertama = tensor gambar
        elif isinstance(batch, dict):
            # jaga-jaga kalau nanti DataLoader mengembalikan dict
            images = batch.get("image") or batch.get("images")
            if images is None:
                raise ValueError("Batch dict tidak memiliki key 'image'/'images'.")
        else:
            # fallback: kalau DataLoader langsung mengembalikan tensor
            images = batch

        errors = get_reconstruction_error(ae_model, images, device)
        all_errors.extend(errors)
        all_labels.extend([1] * len(errors))  # 1 = Good


    # 2) BAD images
    log("\n2. Loading 'Bad' quality images (ungradable folder)...")
    bad_folder = base_config['paths'].get('ungradable_folder') or \
                 os.path.join(base_config['paths']['data_root'], '5_ungradable_oia')
    log(f"   Bad folder path: {bad_folder} (source=paths.ungradable_folder)")
    if not os.path.exists(bad_folder):
        raise FileNotFoundError(f"Bad images folder not found: {bad_folder}")

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
                'weak_label_class': 5
            })
    bad_manifest_temp = '/tmp/bad_images_temp.csv'
    pd.DataFrame(bad_manifest_data).to_csv(bad_manifest_temp, index=False)

    bad_dataset = DRDataset(bad_manifest_temp, base_config, mode='val')
    bad_loader = DataLoader(
        bad_dataset,
        batch_size=base_config.get('inference', {}).get('batch_size', 8),
        shuffle=False,
        num_workers=4
    )
    log(f"   Loaded: {len(bad_dataset)} 'Bad' images")

    # ⬇︎ PATCH: sama seperti GOOD, ambil batch[0] sebagai gambar
    for batch in tqdm(bad_loader, desc="Processing 'Bad' images"):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        elif isinstance(batch, dict):
            images = batch.get("image") or batch.get("images")
            if images is None:
                raise ValueError("Batch dict tidak memiliki key 'image'/'images'.")
        else:
            images = batch

        errors = get_reconstruction_error(ae_model, images, device)
        all_errors.extend(errors)
        all_labels.extend([0] * len(errors))  # 0 = Bad


    # ROC
    log("\n3. Computing ROC curve...")
    errors_array = np.array(all_errors)
    labels_array = np.array(all_labels)

    min_error = float(errors_array.min())
    max_error = float(errors_array.max())

    scores = -errors_array
    fpr, tpr, thresholds = roc_curve(labels_array, scores)
    youdens_j = tpr - fpr
    optimal_idx = np.argmax(youdens_j)
    optimal_error_threshold = -float(thresholds[optimal_idx])
    roc_auc = auc(fpr, tpr)

    log("\n" + "="*80)
    log("ROC CALIBRATION RESULTS")
    log("="*80)
    log(f"Total 'Good' images: {int((labels_array==1).sum())}")
    log(f"Total 'Bad' images: {int((labels_array==0).sum())}")
    log(f"Optimal Error Threshold: {optimal_error_threshold:.6f}")
    log(f"Error range (calibration): [{min_error:.6f}, {max_error:.6f}]")
    log(f"AUC-ROC: {roc_auc:.4f}")

    return optimal_error_threshold, min_error, max_error


# =============================================================================
# Ensemble A loader
# =============================================================================

def load_ensemble_a(config_paths, base_config, device, logger=None):
    """
    Load Ensemble A (1..N models) + temperatures.
    Menghormati kebijakan offline (bobot lokal via config model).
    """
    log = (logger.info if logger else print)

    log("\n" + "="*80)
    log("LOADING ENSEMBLE A")
    log("="*80)

    # Default configs (ConvNeXt + DINOv2 only; CLIPViT dropped)
    if config_paths is None:
        config_paths = [
            'configs/convnext_config.yaml',
            'configs/dinov2_config.yaml',
        ]
    if isinstance(config_paths, str):
        config_paths = [config_paths]

    ensemble_a, temperatures = [], []

    for i, config_path in enumerate(config_paths, 1):
        log(f"\n{i}. Loading model from: {config_path}")
        with open(config_path, 'r') as f:
            mcfg = yaml.safe_load(f)

        # Inject global base paths
        mcfg['paths'] = base_config['paths']

        # Create model (harus load bobot lokal di factory)
        model = create_model_from_config(mcfg, logger=logger)

        # Checkpoint path (lokal)
        checkpoint_path = os.path.join(
            mcfg['model']['output_dir'],
            mcfg['model']['checkpoint_name']
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device).eval()
        log(f"   ✓ Model loaded: {checkpoint_path}")

        # Temperature path (lokal)
        T_path = mcfg['model']['calibration_path']
        if not os.path.exists(T_path):
            raise FileNotFoundError(f"Temperature file not found: {T_path}")

        T_data = torch.load(T_path, map_location='cpu')
        T_optimal = T_data['temperature'] if isinstance(T_data, dict) else T_data
        log(f"   ✓ Temperature loaded: T = {float(T_optimal):.4f}")

        ensemble_a.append(model)
        temperatures.append(float(T_optimal))

    log(f"\n✓ Ensemble A loaded: {len(ensemble_a)} models")
    return ensemble_a, temperatures


# =============================================================================
# Tiering helpers
# =============================================================================

def _resolve_tiering_outputs(base_config):
    tcfg = base_config.get('tiering', {})
    outdir = tcfg.get('outputs', {}).get('dir') or base_config['paths']['processed_dir']
    tier_a = os.path.join(outdir, tcfg.get('outputs', {}).get('tier_a', 'tier_A.csv'))
    tier_b = os.path.join(outdir, tcfg.get('outputs', {}).get('tier_b', 'tier_B.csv'))
    tier_c = os.path.join(outdir, tcfg.get('outputs', {}).get('tier_c', 'tier_C.csv'))
    os.makedirs(outdir, exist_ok=True)
    return tier_a, tier_b, tier_c

def _assign_tiers(df: pd.DataFrame, base_config: dict) -> pd.DataFrame:
    """
    Assign tier A/B/C berdasarkan thresholds di base_config['tiering'].
    Menggunakan kolom: entropy, margin, (opsional) agreement.
    Jika 'agreement' tidak tersedia, aturan agreement di-skip.
    """
    tcfg = base_config.get('tiering', {})
    if not tcfg or not tcfg.get('use', False):
        df['tier'] = 'A'  # default (tidak membatasi)
        return df

    th = tcfg.get('thresholds', {})
    # thresholds structured as in base_config: entropy.a_max/b_max; margin.a_min/b_min; agreement.a_min/b_min
    def decide_row(row):
        ent = row.get('entropy', np.nan)
        mar = row.get('margin', np.nan)
        agr = row.get('agreement', np.nan)  # bisa NaN

        # Flags
        a_ok = True
        b_ok = True

        # Entropy rule
        if 'entropy' in th:
            a_ok &= (ent <= th['entropy'].get('a_max', np.inf))
            b_ok &= (ent <= th['entropy'].get('b_max', np.inf))

        # Margin rule
        if 'margin' in th:
            a_ok &= (mar >= th['margin'].get('a_min', -np.inf))
            b_ok &= (mar >= th['margin'].get('b_min', -np.inf))

        # Agreement rule (opsional)
        if 'agreement' in th and not np.isnan(agr):
            a_ok &= (agr >= th['agreement'].get('a_min', -np.inf))
            b_ok &= (agr >= th['agreement'].get('b_min', -np.inf))

        if a_ok:
            return 'A'
        if b_ok:
            return 'B'
        return 'C'

    df['tier'] = df.apply(decide_row, axis=1)
    return df

def _export_tier_manifests(results_df: pd.DataFrame, base_config: dict, manifest_df: pd.DataFrame):
    """
    Tulis tier_A/B/C.csv berdasarkan kolom 'tier'.
    Mengambil kolom penting dari manifest asli agar downstream (dataset/sampler) mudah.
    """
    tier_a_path, tier_b_path, tier_c_path = _resolve_tiering_outputs(base_config)

    # 1) Tentukan kolom meta dari manifest (TANPA 'label') untuk menghindari duplikasi
    existing_cols = set(results_df.columns)

    meta_cols = []
    for c in ['source', 'device', 'year', 'grade']:
        if c in manifest_df.columns and c not in existing_cols:
            meta_cols.append(c)

    # 2) Merge: skor (results_df) + meta (manifest_df) via image_path
    #    - label diambil dari results_df (sudah ada di full_inference_results)
    #    - meta_cols hanya yang belum ada di results_df untuk menghindari suffix _x/_y
    cols_for_merge = ['image_path'] + meta_cols if 'image_path' in manifest_df.columns else meta_cols

    merged = results_df.merge(
        manifest_df[cols_for_merge],
        on='image_path',
        how='left'
    )

    # 3) Kolom yang diekspor ke tier_A/B/C.csv:
    #    gunakan yang BENAR-BENAR ada di merged (supaya tidak ada KeyError)
    export_cols = []
    for c in ['image_path', 'label', 'source', 'device', 'year', 'grade']:
        if c in merged.columns:
            export_cols.append(c)

    # 4) Split per tier
    A = merged[merged['tier'] == 'A']
    B = merged[merged['tier'] == 'B']
    C = merged[merged['tier'] == 'C']

    # 5) Simpan hanya kolom yang tersedia (export_cols)
    A[export_cols].to_csv(tier_a_path, index=False)
    B[export_cols].to_csv(tier_b_path, index=False)
    C[export_cols].to_csv(tier_c_path, index=False)

    print("\n" + "="*80)
    print("TIERING SUMMARY")
    print("="*80)
    print(f"Tier A: {len(A):,} → {tier_a_path}")
    print(f"Tier B: {len(B):,} → {tier_b_path}")
    print(f"Tier C: {len(C):,} → {tier_c_path}")

# =============================================================================
# MAIN INFERENCE PIPELINE
# =============================================================================

def run_full_inference(base_config_path='configs/base_config_coweps.yaml',
                      ensemble_a_configs=None,
                      ae_config_path='configs/autoencoder_config.yaml',
                      mode='master',
                      logger=None):

    """
    Run full CoWePS inference pipeline (AI Ganda v2.5)

    Steps:
      1) Load base & AE configs
      2) Load Ensemble A (models + T)
      3) (Optional) Load AE + calibrate Q threshold
      4) Prepare loader (mode-aware)
      5) Inference: probs → C_score, entropy, margin; AE → Q_score
      6) Save results CSV
      7) (Optional) Tiering & export tier_A/B/C
    """
    log = (logger.info if logger else print)

    print("\n" + "="*80)
    print("CoWePS v2.5 - FULL INFERENCE PIPELINE (AI GANDA)")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 1) Load configs
    print("\n" + "="*80)
    print("1. LOADING CONFIGURATIONS")
    print("="*80)
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    print(f"✓ Base config loaded: {base_config_path}")

    ae_enabled = False
    skip_q = base_config.get('inference', {}).get('skip_qscore', False)

    if ae_config_path is not None and not skip_q:
        with open(ae_config_path, 'r') as f:
            ae_config = yaml.safe_load(f)
        ae_config['paths'] = base_config['paths']
        print(f"✓ Autoencoder config loaded: {ae_config_path}")
        ae_enabled = True
    else:
        ae_config = None
        if skip_q:
            print("✓ Q-score disabled via base_config['inference']['skip_qscore']=True (AE tidak dipakai).")
        else:
            print("✓ No AE config provided, Q-score disabled.")

    # 2) Load Ensemble A
    # Normalize/record the Ensemble A config list for provenance.
    ensemble_a_configs_norm = ensemble_a_configs
    if ensemble_a_configs_norm is None:
        ensemble_a_configs_norm = [
            'configs/convnext_config.yaml',
            'configs/dinov2_config.yaml',
        ]
    if isinstance(ensemble_a_configs_norm, str):
        ensemble_a_configs_norm = [ensemble_a_configs_norm]

    ensemble_a, temperatures = load_ensemble_a(ensemble_a_configs_norm, base_config, device, logger)

    # 3) Load AE + calibrate Q threshold
    ae_model, Q_threshold, q_err_min, q_err_max = None, None, None, None
    if ae_enabled:
        print("\n" + "="*80)
        print("LOADING ENSEMBLE B (AUTOENCODER)")
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

        Q_threshold, q_err_min, q_err_max = find_q_threshold_roc(
            ae_model, base_config, device, logger
        )

    # 4) Prepare loader
    print("\n" + "="*80)
    print("5. PREPARING MAIN DATA LOADER")
    print("="*80)
    manifest_path = _resolve_manifest_for_mode(base_config, mode, logger)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found for mode '{mode}': {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    main_dataset = DRDataset(manifest_path, base_config, mode='val')
    bs = base_config.get('inference', {}).get('batch_size', 8)
    main_loader = DataLoader(main_dataset, batch_size=bs, shuffle=False, num_workers=4)

    print(f"✓ Mode: {mode}")
    print(f"✓ Manifest: {manifest_path}")
    print(f"✓ Main dataset loaded: {len(main_dataset)} images")
    print(f"✓ Batch size: {main_loader.batch_size}")

    # 5) Inference loop
    print("\n" + "="*80)
    print("6. RUNNING FULL INFERENCE (AI GANDA)")
    print("="*80)

    results = []
    idx_global = 0

    with torch.no_grad():
        for batch in tqdm(main_loader, desc="CoWePS Inference"):
            # Dukung output (images, labels) atau (images, labels, meta/extra)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    images, labels = batch
                elif len(batch) >= 3:
                    images, labels = batch[0], batch[1]
                else:
                    raise ValueError(f"Unexpected batch length from DataLoader: {len(batch)}")
            elif isinstance(batch, dict):
                images = batch.get("image") or batch.get("images")
                labels = batch.get("label") or batch.get("labels")
                if images is None or labels is None:
                    raise ValueError("Batch dict must contain 'image(s)' and 'label(s)' keys.")
            else:
                raise ValueError(f"Unexpected batch type from DataLoader: {type(batch)}")

            images = images.to(device)
            B = images.shape[0]

            # (a) Ensemble A → probs
            #     - BEFORE calibration: average raw logits
            #     - AFTER  calibration: temperature-scale each model then average
            raw_logits_list = []
            scaled_logits_list = []
            for model, T in zip(ensemble_a, temperatures):
                logits = model(images)
                raw_logits_list.append(logits.unsqueeze(0))

                scaled_logits = _apply_temperature(logits, T)
                scaled_logits_list.append(scaled_logits.unsqueeze(0))

            avg_raw_logits = torch.mean(torch.cat(raw_logits_list, dim=0), dim=0)
            avg_scaled_logits = torch.mean(torch.cat(scaled_logits_list, dim=0), dim=0)

            probs_before = torch.softmax(avg_raw_logits, dim=1)
            probs_after = torch.softmax(avg_scaled_logits, dim=1)

            C_score_before, Pred_Class_before = probs_before.max(dim=1)
            entropy_before, margin_before = _probs_entropy_margin(probs_before)

            C_score, Pred_Class = probs_after.max(dim=1)
            entropy, margin = _probs_entropy_margin(probs_after)

            # (b) AE → Q-score
            if ae_enabled:
                raw_error = get_reconstruction_error(ae_model, images, device)
                denom = (q_err_max - q_err_min) if (q_err_max > q_err_min) else 1.0
                q_cont = (q_err_max - raw_error) / denom
                q_cont = np.clip(q_cont, 0.0, 1.0)
                Q_bin = (raw_error < Q_threshold).astype(np.int32)
            else:
                raw_error = np.full((B,), np.nan, dtype=np.float32)
                q_cont = np.full((B,), np.nan, dtype=np.float32)
                Q_bin = np.full((B,), 1, dtype=np.int32)  # treat as good if AE disabled

            # (c) Stitch results with manifest rows in-order
            # (c) Stitch results with manifest rows in-order + label & probs
            #    - Simpan juga label ground-truth
            #    - Simpan p_k per kelas (p0..p{C-1})
            #    - Hitung label_confidence_R = p(label | x)
            num_classes = probs_after.shape[1]

            for i in range(B):
                # baris manifest sesuai urutan loader (DRDataset standard)
                if idx_global >= len(manifest_df):
                    break

                img_path = manifest_df.iloc[idx_global].get('image_path', None)
                # fallback (kalau kolom bernama lain, mis: filepath)
                if img_path is None:
                    for cand in ['filepath', 'file_path', 'path']:
                        if cand in manifest_df.columns:
                            img_path = manifest_df.iloc[idx_global][cand]
                            break

                # Ambil label ground-truth dari batch
                try:
                    label_val = int(labels[i].item())
                except Exception:
                    # fallback jika labels bukan tensor standar
                    label_val = int(labels[i])

                # Hitung confidence pada label ground-truth (R)
                if 0 <= label_val < num_classes:
                    label_conf_R = float(probs_after[i, label_val].item())
                    label_conf_R_before = float(probs_before[i, label_val].item())
                else:
                    # misal untuk kelas "ungradable" di luar range num_classes
                    label_conf_R = float('nan')
                    label_conf_R_before = float('nan')

                # Bangun row hasil
                row = {
                    'image_path': img_path,
                    'label': label_val,
                    'C_score_before': float(C_score_before[i].item()),
                    'entropy_before': float(entropy_before[i].item()),
                    'margin_before': float(margin_before[i].item()),
                    'Pred_Class_before': int(Pred_Class_before[i].item()),
                    'label_confidence_R_before': label_conf_R_before,
                    'C_score': float(C_score[i].item()),
                    'entropy': float(entropy[i].item()),
                    'margin': float(margin[i].item()),
                    'Q_score': int(Q_bin[i]),
                    'Q_score_continuous': float(q_cont[i]),
                    'Pred_Class': int(Pred_Class[i].item()),
                    'reconstruction_error': float(raw_error[i]),
                    'label_confidence_R': label_conf_R,
                }

                # Tambahkan probabilitas per-kelas BEFORE calibration (p0_before..)
                for c in range(num_classes):
                    row[f'p{c}_before'] = float(probs_before[i, c].item())

                # Tambahkan probabilitas per-kelas AFTER calibration (p0..p{num_classes-1})
                for c in range(num_classes):
                    row[f'p{c}'] = float(probs_after[i, c].item())

                results.append(row)
                idx_global += 1


    results_df = pd.DataFrame(results)

    # Provenance columns (constant per run). Safe for downstream consumers.
    try:
        results_df.insert(0, 'ensemble_a_configs', "|".join(map(str, ensemble_a_configs_norm)))
    except Exception:
        results_df['ensemble_a_configs'] = "|".join(map(str, ensemble_a_configs_norm))

    results_df['ae_config_path'] = str(ae_config_path) if ae_enabled else ""
    results_df['inference_mode'] = str(mode)

    # 6) Save results
    print("\n" + "="*80)
    print("7. SAVING RESULTS")
    print("="*80)
    # resolve output path
    save_csv = base_config.get('inference', {}).get('save_scores_csv', None)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        output_path = save_csv if save_csv.endswith('.csv') else os.path.join(save_csv, f'full_inference_results_{mode}.csv')
    else:
        scores_dir = base_config.get('paths', {}).get('scores_dir', 'data/scores')
        os.makedirs(scores_dir, exist_ok=True)
        output_path = os.path.join(scores_dir, f'full_inference_results_{mode}.csv')

    results_df.to_csv(output_path, index=False)

    print("INFERENCE SUMMARY")
    print("="*80)
    print(f"✓ Results saved: {output_path}")
    print(f"✓ Total images processed: {len(results_df):,}")
    if 'Q_score' in results_df.columns and results_df['Q_score'].notna().any():
        good = int((results_df['Q_score'] == 1).sum())
        bad = int((results_df['Q_score'] == 0).sum())
        print(f"\nQ-score Distribution: Q=1 {good:,} ({good/len(results_df)*100:.1f}%), Q=0 {bad:,} ({bad/len(results_df)*100:.1f}%)")
    print(f"\nC-score Statistics: mean={results_df['C_score'].mean():.4f}, "
          f"median={results_df['C_score'].median():.4f}, "
          f"min={results_df['C_score'].min():.4f}, max={results_df['C_score'].max():.4f}")

    # 7) Tiering A/B/C (optional)
    tcfg = base_config.get('tiering', {})
    if tcfg and tcfg.get('use', False):
        # jika kolom 'agreement' belum ada, isi NaN (aturan agreement otomatis dilewati)
        if 'agreement' not in results_df.columns:
            results_df['agreement'] = np.nan

        # assign tiers
        results_df = _assign_tiers(results_df, base_config)

        # export tier manifests (join kolom penting dari manifest asli)
        _export_tier_manifests(results_df, base_config, manifest_df)

    print("\n" + "="*80)
    print("✓ FASE 3 COMPLETE")
    print("="*80)

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CoWePS full inference (v2.5)")
    parser.add_argument('--base-config', type=str, default='configs/base_config_coweps.yaml')
    parser.add_argument('--ae-config', type=str, default='configs/autoencoder_config.yaml')
    parser.add_argument('--mode', type=str, default='master', choices=['master', 'train', 'val', 'gold_standard'])
    args = parser.parse_args()

    run_full_inference(
        base_config_path=args.base_config,
        ae_config_path=args.ae_config,
        mode=args.mode
    )
