import torch
import torch.nn as nn

class DownSampling(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(DownSampling, self).__init__()
        print(dim_in, dim_out)
        self.activation = nn.LeakyReLU()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class PyramidEncoder(nn.Module):

    def __init__(self, image_channels, input_dimension, network_depth, residual_mask):
        super(PyramidEncoder, self).__init__()
        print('### PYRAMID ENDCODER')
        self.network_depth = network_depth
        self.residual_mask = residual_mask
        dim_in = image_channels
        self.downs = nn.ModuleList([])
        for i in range(network_depth):
            if i == 0:
                dim_out = input_dimension
            else:
                dim_out = dim_in * 2
            self.downs.append(DownSampling(dim_in, dim_out))
            dim_in = dim_out

    def forward(self, x):
        h = []
        for i in range(self.network_depth):
            l = self.downs[i]
            x = l(x)
            if self.residual_mask[i]:
                h.append(x)
            else:
                h.append(torch.zeros(x.size()))
        return (x, h)