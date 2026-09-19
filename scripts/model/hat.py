import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.avg_pool(x)
        attn = self.conv1(attn)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class HATBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        # Hybrid part: Channel Attention applied in parallel
        self.cab = ChannelAttention(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Channel Attention (spatial domain)
        cab_out = self.cab(x)
        
        # Window Attention (patch domain)
        x_flat = x.flatten(2).transpose(1, 2) # B, H*W, C
        
        # Simple windowing approximation for lightweight HAT
        res = x_flat
        x_flat = self.norm1(x_flat)
        attn_out = self.attn(x_flat)
        x_flat = res + attn_out
        
        res = x_flat
        x_flat = self.norm2(x_flat)
        mlp_out = self.mlp(x_flat)
        x_flat = res + mlp_out
        
        x_spatial = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        # Fuse spatial and channel streams
        out = x_spatial + cab_out
        return out

class HAT(nn.Module):
    """Lightweight Hybrid Attention Transformer"""
    def __init__(self, inp_channels=3, out_channels=3, dim=32, num_blocks=[2, 2, 2, 2], num_heads=[1, 2, 4, 8]):
        super(HAT, self).__init__()
        
        self.embed = nn.Conv2d(inp_channels, dim, 3, 1, 1)
        
        self.enc1 = nn.Sequential(*[HATBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, dim*2, 4, 2, 1)
        
        self.enc2 = nn.Sequential(*[HATBlock(dim*2, num_heads[1]) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim*2, dim*4, 4, 2, 1)
        
        self.enc3 = nn.Sequential(*[HATBlock(dim*4, num_heads[2]) for _ in range(num_blocks[2])])
        self.down3 = nn.Conv2d(dim*4, dim*8, 4, 2, 1)
        
        self.bottleneck = nn.Sequential(*[HATBlock(dim*8, num_heads[3]) for _ in range(num_blocks[3])])
        
        self.up3 = nn.ConvTranspose2d(dim*8, dim*4, 2, 2)
        self.reduce3 = nn.Conv2d(dim*8, dim*4, 1)
        self.dec3 = nn.Sequential(*[HATBlock(dim*4, num_heads[2]) for _ in range(num_blocks[2])])
        
        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, 2, 2)
        self.reduce2 = nn.Conv2d(dim*4, dim*2, 1)
        self.dec2 = nn.Sequential(*[HATBlock(dim*2, num_heads[1]) for _ in range(num_blocks[1])])
        
        self.up1 = nn.ConvTranspose2d(dim*2, dim, 2, 2)
        self.reduce1 = nn.Conv2d(dim*2, dim, 1)
        self.dec1 = nn.Sequential(*[HATBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x):
        inp = x
        x1 = self.embed(x)
        
        x1_out = self.enc1(x1)
        x2 = self.down1(x1_out)
        
        x2_out = self.enc2(x2)
        x3 = self.down2(x2_out)
        
        x3_out = self.enc3(x3)
        x4 = self.down3(x3_out)
        
        latent = self.bottleneck(x4)
        
        d3 = self.up3(latent)
        d3 = torch.cat([d3, x3_out], dim=1)
        d3 = self.reduce3(d3)
        d3_out = self.dec3(d3)
        
        d2 = self.up2(d3_out)
        d2 = torch.cat([d2, x2_out], dim=1)
        d2 = self.reduce2(d2)
        d2_out = self.dec2(d2)
        
        d1 = self.up1(d2_out)
        d1 = torch.cat([d1, x1_out], dim=1)
        d1 = self.reduce1(d1)
        d1_out = self.dec1(d1)
        
        out = self.output(d1_out) + inp
        return torch.clamp(out, 0, 1)
