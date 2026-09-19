import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # resize g1 if sizes don't match
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, filters=64):
        super(AttentionUNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = ConvBlock(img_ch, filters)
        self.Conv2 = ConvBlock(filters, filters*2)
        self.Conv3 = ConvBlock(filters*2, filters*4)
        self.Conv4 = ConvBlock(filters*4, filters*8)
        
        self.Conv5 = ConvBlock(filters*8, filters*16)
        
        self.Up5 = nn.ConvTranspose2d(filters*16, filters*8, kernel_size=2, stride=2)
        self.Att5 = AttentionGate(F_g=filters*8, F_l=filters*8, F_int=filters*4)
        self.Up_conv5 = ConvBlock(filters*16, filters*8)
        
        self.Up4 = nn.ConvTranspose2d(filters*8, filters*4, kernel_size=2, stride=2)
        self.Att4 = AttentionGate(F_g=filters*4, F_l=filters*4, F_int=filters*2)
        self.Up_conv4 = ConvBlock(filters*8, filters*4)
        
        self.Up3 = nn.ConvTranspose2d(filters*4, filters*2, kernel_size=2, stride=2)
        self.Att3 = AttentionGate(F_g=filters*2, F_l=filters*2, F_int=filters)
        self.Up_conv3 = ConvBlock(filters*4, filters*2)
        
        self.Up2 = nn.ConvTranspose2d(filters*2, filters, kernel_size=2, stride=2)
        self.Att2 = AttentionGate(F_g=filters, F_l=filters, F_int=filters//2)
        self.Up_conv2 = ConvBlock(filters*2, filters)
        
        self.Conv_1x1 = nn.Conv2d(filters, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)
        
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)
        
        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        # match dimensions if maxpool caused rounding
        if d5.shape[2:] != x4.shape[2:]:
            d5 = F.interpolate(d5, size=x4.shape[2:], mode='bilinear', align_corners=False)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        if d4.shape[2:] != x3.shape[2:]:
            d4 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        if d3.shape[2:] != x2.shape[2:]:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        if d2.shape[2:] != x1.shape[2:]:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        out = self.Conv_1x1(d2)
        return torch.sigmoid(out) # Return [0, 1] output
