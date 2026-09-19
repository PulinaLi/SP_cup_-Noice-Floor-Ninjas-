import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.conv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Norm requires channels last
        x_norm = x.reshape(b, c, h*w).transpose(1, 2)
        x_norm = self.norm1(x_norm).transpose(1, 2).reshape(b, c, h, w)
        x = x + self.attn(x_norm)
        
        x_norm = x.reshape(b, c, h*w).transpose(1, 2)
        x_norm = self.norm2(x_norm).transpose(1, 2).reshape(b, c, h, w)
        x = x + self.ffn(x_norm)
        
        return x

class Restormer(nn.Module):
    """Lightweight Restormer to fit in 6GB VRAM"""
    def __init__(self, inp_channels=3, out_channels=3, dim=24, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], expansion_factor=2.66):
        super(Restormer, self).__init__()

        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1, bias=False)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0], expansion_factor) for _ in range(num_blocks[0])])
        self.down1_2 = nn.Conv2d(dim, dim*2, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim*2, num_heads[1], expansion_factor) for _ in range(num_blocks[1])])
        self.down2_3 = nn.Conv2d(dim*2, dim*4, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim*4, num_heads[2], expansion_factor) for _ in range(num_blocks[2])])
        self.down3_4 = nn.Conv2d(dim*4, dim*8, kernel_size=4, stride=2, padding=1, bias=False)

        self.latent = nn.Sequential(*[TransformerBlock(dim*8, num_heads[3], expansion_factor) for _ in range(num_blocks[3])])

        self.up4_3 = nn.ConvTranspose2d(dim*8, dim*4, kernel_size=2, stride=2, bias=False)
        self.reduce_chan_level3 = nn.Conv2d(dim*8, dim*4, kernel_size=1, bias=False)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim*4, num_heads[2], expansion_factor) for _ in range(num_blocks[2])])

        self.up3_2 = nn.ConvTranspose2d(dim*4, dim*2, kernel_size=2, stride=2, bias=False)
        self.reduce_chan_level2 = nn.Conv2d(dim*4, dim*2, kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim*2, num_heads[1], expansion_factor) for _ in range(num_blocks[1])])

        self.up2_1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2, bias=False)
        self.reduce_chan_level1 = nn.Conv2d(dim*2, dim, kernel_size=1, bias=False)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0], expansion_factor) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        
        latent = self.latent(inp_enc_level4)
        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out = self.output(out_dec_level1) + inp_img
        
        return torch.clamp(out, 0, 1)
