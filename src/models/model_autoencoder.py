#!/usr/bin/env python
"""
CoWePS v2.4 - Autoencoder Model (Ensemble B)

U-Net architecture untuk quality assessment via reconstruction error.

KUNCI v2.4:
- Input: (B, 3, 512, 512) - gambar RGB 512x512 yang sudah dimasker
- Output: (B, 3, 512, 512) - rekonstruksi gambar
- Architecture: U-Net dengan skip connections
- Purpose: Learn to reconstruct high-quality retinal images

Author: CoWePS v2.4 Implementation Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# U-NET BUILDING BLOCKS
# ============================================================================

class DoubleConv(nn.Module):
    """
    Double Convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    
    Standard building block untuk U-Net
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block: MaxPool -> DoubleConv
    
    Encoder path dari U-Net
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: Upsample -> DoubleConv
    
    Decoder path dari U-Net dengan skip connections
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        # Use bilinear upsampling atau transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Output dari decoder level sebelumnya
            x2: Skip connection dari encoder (same resolution)
        """
        x1 = self.up(x1)
        
        # Handle size mismatch (jika ada)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution: 1x1 conv untuk menghasilkan final output
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


# ============================================================================
# U-NET AUTOENCODER
# ============================================================================

class UNet(nn.Module):
    """
    U-Net Autoencoder untuk quality assessment
    
    Architecture:
    - Encoder: 4 downsampling stages (512 -> 256 -> 128 -> 64 -> 32)
    - Bottleneck: Deepest feature representation
    - Decoder: 4 upsampling stages dengan skip connections
    
    Input: (B, 3, 512, 512)
    Output: (B, 3, 512, 512)
    
    Reference: Ronneberger et al., "U-Net: Convolutional Networks for 
               Biomedical Image Segmentation" (2015)
    """
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        """
        Initialize U-Net
        
        Args:
            n_channels: Input channels (3 untuk RGB)
            n_classes: Output channels (3 untuk RGB reconstruction)
            bilinear: Use bilinear upsampling (True) atau transposed conv (False)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, 512, 512)
        
        Returns:
            reconstructed: Reconstructed image (B, 3, 512, 512)
        """
        # Encoder dengan skip connections
        x1 = self.inc(x)      # (B, 64, 512, 512)
        x2 = self.down1(x1)   # (B, 128, 256, 256)
        x3 = self.down2(x2)   # (B, 256, 128, 128)
        x4 = self.down3(x3)   # (B, 512, 64, 64)
        x5 = self.down4(x4)   # (B, 512, 32, 32) - bottleneck
        
        # Decoder dengan skip connections
        x = self.up1(x5, x4)  # (B, 256, 64, 64)
        x = self.up2(x, x3)   # (B, 128, 128, 128)
        x = self.up3(x, x2)   # (B, 64, 256, 256)
        x = self.up4(x, x1)   # (B, 64, 512, 512)
        
        # Output (no activation - linear output untuk MSE loss)
        reconstructed = self.outc(x)  # (B, 3, 512, 512)
        
        return reconstructed
    
    def get_reconstruction_error(self, x):
        """
        Compute per-sample reconstruction error
        
        Args:
            x: Input tensor (B, 3, 512, 512)
        
        Returns:
            errors: Per-sample MSE (B,)
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Compute MSE per sample (average over C, H, W)
            errors = ((x - reconstructed) ** 2).mean(dim=[1, 2, 3])
        return errors


# ============================================================================
# SIMPLER AUTOENCODER (Alternative - Lighter)
# ============================================================================

class SimpleAutoencoder(nn.Module):
    """
    Simpler autoencoder alternative (jika U-Net terlalu berat)
    
    Architecture:
    - Encoder: Conv layers dengan downsampling
    - Bottleneck: Compressed representation
    - Decoder: Transposed conv layers dengan upsampling
    
    NOTE: U-Net lebih direkomendasikan karena skip connections
    """
    def __init__(self, n_channels=3):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, padding=1),  # 512->256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 256->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 128->64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 64->32
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32->64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128->256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256->512
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_autoencoder(config):
    """
    Create autoencoder from config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: Autoencoder model
    """
    architecture = config.get('model', {}).get('architecture', 'UNet')
    
    if architecture == 'UNet':
        model = UNet(n_channels=3, n_classes=3, bilinear=True)
        print(f"Created U-Net Autoencoder")
    elif architecture == 'SimpleAutoencoder':
        model = SimpleAutoencoder(n_channels=3)
        print(f"Created Simple Autoencoder")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


# ============================================================================
# EXPORT
# ============================================================================

# Main model untuk digunakan
AutoencoderModel = UNet

# Alternative
SimpleAutoencoderModel = SimpleAutoencoder


if __name__ == "__main__":
    """Test autoencoder architecture"""
    print("Testing U-Net Autoencoder...")
    
    # Create model
    model = UNet(n_channels=3, n_classes=3)
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)  # Batch of 2 images
    print(f"Input shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    
    # Test reconstruction error
    errors = model.get_reconstruction_error(x)
    print(f"Reconstruction errors: {errors}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✓ Autoencoder architecture test passed!")