
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = ConvBlock(out_c * 2, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        # Guard against size mismatch from padding
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net.
    base_ch=64 → channels: 64,128,256,512,1024
    """
    def __init__(self, in_channels: int = 1,
                 out_channels: int = 1,
                 base_ch: int = 64):
        super().__init__()
        b = base_ch
        # Encoder
        self.enc1 = EncoderBlock(in_channels, b)
        self.enc2 = EncoderBlock(b,     b*2)
        self.enc3 = EncoderBlock(b*2,   b*4)
        self.enc4 = EncoderBlock(b*4,   b*8)
        # Bottleneck
        self.bottleneck = ConvBlock(b*8, b*16)
        # Decoder
        self.dec4 = DecoderBlock(b*16, b*8)
        self.dec3 = DecoderBlock(b*8,  b*4)
        self.dec2 = DecoderBlock(b*4,  b*2)
        self.dec1 = DecoderBlock(b*2,  b)
        # Output — raw logits (no sigmoid)
        self.out  = nn.Conv2d(b, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b      = self.bottleneck(p4)
        d4     = self.dec4(b,  s4)
        d3     = self.dec3(d4, s3)
        d2     = self.dec2(d3, s2)
        d1     = self.dec1(d2, s1)
        return self.out(d1)   # raw logits


if __name__ == "__main__":
    device = torch.device( "cpu")
    model  = UNet(in_channels=1, out_channels=1, base_ch=64).to(device)
    x      = torch.randn(2, 1, 256, 256).to(device)
    y      = model(x)
    total  = sum(p.numel() for p in model.parameters())
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {total:,}")
    assert y.shape == x.shape, "Output shape mismatch!"
    print("Self-test passed ✅")