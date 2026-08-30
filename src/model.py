"""U-Net surrogate: geometry + operating conditions  ->  (ux, uy, p, T) fields."""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=8, out_ch=4, base=48):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.inc = DoubleConv(in_ch, c[0])
        self.d1 = DoubleConv(c[0], c[1])
        self.d2 = DoubleConv(c[1], c[2])
        self.d3 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.u3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.u2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.u1 = DoubleConv(c[1], c[0])
        self.outc = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x):
        x0 = self.inc(x)              # (b,c0,H,W)
        x1 = self.d1(self.pool(x0))   # /2
        x2 = self.d2(self.pool(x1))   # /4
        x3 = self.d3(self.pool(x2))   # /8
        y = self.up3(x3)
        y = self.u3(torch.cat([y, x2], 1))
        y = self.up2(y)
        y = self.u2(torch.cat([y, x1], 1))
        y = self.up1(y)
        y = self.u1(torch.cat([y, x0], 1))
        return self.outc(y)
