# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: D:\Python\univ_proj\computer_vision\computer_vision_project\model\monodepth_rt\monodepthrt.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2024-05-15 20:12:21 UTC (1715803941)

import numpy as np
import torch.nn as nn
from computer_vision_project.model.monodepth_rt.depth_decoder import DepthDecoder
from computer_vision_project.model.monodepth_rt.pyramidal_encoder import PyramidEncoder

class MonoDepthRT(nn.Module):

    def __init__(self, input_size, base_channels, network_depth, training):
        super(MonoDepthRT, self).__init__()
        c, h, w = input_size
        self.activation = nn.LeakyReLU()
        self.residual_mask = np.append(np.ones(network_depth - 1, dtype=np.uint8), [0])
        self.pyr_enc = PyramidEncoder(c, base_channels, network_depth, self.residual_mask)
        self.dep_dec = DepthDecoder(base_channels * 2 ** (network_depth - 1), network_depth, training)

    def forward(self, x):
        x, res = self.pyr_enc(x)
        residuals = list(reversed(res))[1:]
        x = self.dep_dec(x, residuals)
        return x