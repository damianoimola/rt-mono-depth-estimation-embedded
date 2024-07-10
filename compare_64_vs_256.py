import torch

from model.monodert.net import MonoDeRT
from utilities.comparison import Compare

if __name__ == "__main__":
    model_64 = MonoDeRT(3, 1, False)
    model_256 = MonoDeRT(3, 1, False)

    c = Compare(model_64, model_256)
    input_64 = torch.rand((1, 3, 64, 64))
    input_256 = torch.rand((1, 3, 256, 256))

    print(c.compare(input_64, input_256))