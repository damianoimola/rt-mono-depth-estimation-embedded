import torch
import torch.nn as nn
import torch.nn.functional as F


class MonoDeRT(nn.Module):
    def __init__(self, in_channels, out_channels, multi_output=False, mode='sum'):
        super(MonoDeRT, self).__init__()
        self.mode = mode
        self.mo = multi_output

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        if mode == 'concat':
            self.dec3 = self.upconv_block(512, 128)
            self.pred3 = self.prediction_decoder(256, out_channels)
            self.dec2 = self.upconv_block(256, 64)
            self.pred2 = self.prediction_decoder(128, out_channels)
            self.dec1 = self.upconv_block(128, 32)
            self.pred1 = self.prediction_decoder(32, out_channels)
        else:
            self.dec3 = self.conv_block(256, 128)
            self.pred3 = self.prediction_decoder(128, out_channels)
            self.dec2 = self.conv_block(128, 64)
            self.pred2 = self.prediction_decoder(64, out_channels)
            self.dec1 = self.conv_block(64, 32)
            self.pred1 = self.prediction_decoder(32, out_channels)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU())

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.LeakyReLU())

    def prediction_decoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e1 = self.bicubic(e1, 0.5)
        e2 = self.enc2(e1)
        e2 = self.bicubic(e2, 0.5)
        e3 = self.enc3(e2)
        e3 = self.bicubic(e3, 0.5)


        d3 = self.dec3(e3)
        d3 = self.bicubic(d3, 2)
        d3 = torch.add(d3, e2)
        d2 = self.dec2(d3)
        d2 = self.bicubic(d2, 2)
        d2 = torch.add(d2, e1)
        d1 = self.dec1(d2)
        d1 = self.bicubic(d1, 2)

        # if multi outputs, latents outputs
        if self.mo:
            p1 = self.pred1(d1)
            p2 = self.pred2(d2)
            p3 = self.pred3(d3)

            return [p1, p2, p3]
        else:
            return self.pred1(d1)

    def bicubic(self, batch_tensor, scale_factor):
        return F.interpolate(batch_tensor, scale_factor=scale_factor, mode='bicubic', align_corners=False, antialias=False)