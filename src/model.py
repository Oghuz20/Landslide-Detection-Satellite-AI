"""
Attention U-Net for Semantic Segmentation
Architecture with attention gates for better feature selection
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double convolution block: Conv -> BN -> ReLU -> Dropout -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # Regularization
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionBlock(nn.Module):
    """
    Attention Gate
    
    Highlights important features from skip connections
    Suppresses irrelevant regions
    
    Args:
        F_g: Number of feature maps in gating signal (decoder)
        F_l: Number of feature maps in skip connection (encoder)
        F_int: Number of intermediate feature maps
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        # Gating signal pathway
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Skip connection pathway
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder
            x: Skip connection from encoder
            
        Returns:
            Attention-weighted skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)  # Combine signals
        psi = self.psi(psi)        # Compute attention
        return x * psi             # Apply attention to skip connection


class AttentionUNet(nn.Module):
    """
    Attention U-Net for Landslide Segmentation
    
    Architecture:
        - Encoder: 5 levels (64, 128, 256, 512, 1024 channels)
        - Decoder: 4 levels with attention gates
        - Skip connections: Attention-gated
        
    Input: [B, 14, 128, 128] (14-channel satellite image)
    Output: [B, 2, 128, 128] (2-class segmentation)
    """
    
    def __init__(self, n_channels=14, n_classes=2):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder Level 1
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.upconv1 = DoubleConv(1024, 512)
        
        # Decoder Level 2
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.upconv2 = DoubleConv(512, 256)
        
        # Decoder Level 3
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.upconv3 = DoubleConv(256, 128)
        
        # Decoder Level 4
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.upconv4 = DoubleConv(128, 64)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, 14, 128, 128] input image
            
        Returns:
            [B, 2, 128, 128] class logits
        """
        # Encoder path
        x1 = self.inc(x)      # 64 channels, 128x128
        x2 = self.down1(x1)   # 128 channels, 64x64
        x3 = self.down2(x2)   # 256 channels, 32x32
        x4 = self.down3(x3)   # 512 channels, 16x16
        x5 = self.down4(x4)   # 1024 channels, 8x8
        
        # Decoder path with attention
        d1 = self.up1(x5)                    # Upsample to 16x16
        x4 = self.att1(g=d1, x=x4)          # Apply attention to skip
        d1 = torch.cat((x4, d1), dim=1)     # Concatenate
        d1 = self.upconv1(d1)               # Process
        
        d2 = self.up2(d1)                    # Upsample to 32x32
        x3 = self.att2(g=d2, x=x3)          # Apply attention
        d2 = torch.cat((x3, d2), dim=1)
        d2 = self.upconv2(d2)
        
        d3 = self.up3(d2)                    # Upsample to 64x64
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.upconv3(d3)
        
        d4 = self.up4(d3)                    # Upsample to 128x128
        x1 = self.att4(g=d4, x=x1)
        d4 = torch.cat((x1, d4), dim=1)
        d4 = self.upconv4(d4)
        
        # Output
        out = self.outc(d4)  # [B, 2, 128, 128]
        return out


def create_model(model_type='attention_unet', n_channels=14, n_classes=2):
    """
    Factory function to create model
    
    Args:
        model_type: 'attention_unet' or 'unet'
        n_channels: Number of input channels (default: 14)
        n_classes: Number of output classes (default: 2)
        
    Returns:
        PyTorch model
    """
    if model_type == 'attention_unet':
        return AttentionUNet(n_channels, n_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test model
    model = create_model('attention_unet')
    x = torch.randn(2, 14, 128, 128)  # Batch of 2 images
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")