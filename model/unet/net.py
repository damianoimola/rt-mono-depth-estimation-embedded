import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, mode='sum'):
        super(UNet, self).__init__()
        self.mode = mode
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        if mode == 'concat':
            self.dec4 = self.upconv_block(512, 256)
            self.dec3 = self.upconv_block(512, 128)
            self.dec2 = self.upconv_block(256, 64)
            self.dec1 = self.upconv_block(128, out_channels)
        else:
            self.dec4 = self.upconv_block(512, 256)
            self.dec3 = self.upconv_block(256, 128)
            self.dec2 = self.upconv_block(128, 64)
            self.dec1 = self.upconv_block(64, out_channels)

        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU())

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.LeakyReLU())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        if self.mode == 'concat':
            d4 = self.dec4(e4)
            d4 = torch.cat((d4, e3), dim=1)
            d3 = self.dec3(d4)
            d3 = torch.cat((d3, e2), dim=1)
            d2 = self.dec2(d3)
            d2 = torch.cat((d2, e1), dim=1)
            d1 = self.dec1(d2)
        else:
            d4 = self.dec4(e4)
            d4 = torch.add(d4, e3)
            d3 = self.dec3(d4)
            d3 = torch.add(d3, e2)
            d2 = self.dec2(d3)
            d2 = torch.add(d2, e1)
            d1 = self.dec1(d2)

        return self.sigmoid(d1)