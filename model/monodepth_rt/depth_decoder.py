import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, dim_in):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in//2, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(dim_in//2, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Upsampling, self).__init__()
        print(dim_in, dim_out)

        self.activation = nn.LeakyReLU()
        self.upconv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.upconv(x)
        x = self.activation(x)
        return x

class DepthDecoder(nn.Module):
    def __init__(self, input_dimension, network_depth, training):
        super(DepthDecoder, self).__init__()
        print("### DEPTH DECODER")

        self.training = training
        self.network_depth = network_depth

        dim_in = input_dimension
        dim_mod_in = input_dimension
        self.ups = nn.ModuleList([])
        self.decs = nn.ModuleList([])
        for i in range(network_depth):
            dim_out = dim_in//2
            self.ups.append(Upsampling(dim_mod_in, dim_out))
            self.decs.append(Decoder(dim_in//2 if i == network_depth-1 else dim_in))
            dim_in//=2
            if i > 0:
                dim_mod_in//=2


    def forward(self, x, residuals):
        decoder_outputs = []

        for i in range(self.network_depth):
            l = self.ups[i]
            d = self.decs[i]

            # last layer behavior
            if i == self.network_depth-1:
                x = l(x)
            else:
                r = residuals[i]
                # upscale
                x = l(x)
                # stack along channels dim
                x = torch.cat((x, r), dim=1)

            # if inference or last layer
            if self.training or i == self.network_depth-1:
                decoder_outputs.append(d(x))
        return decoder_outputs

