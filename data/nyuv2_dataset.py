import cv2
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms import transforms

def transformation():
    return transforms.Compose([transforms.ConvertImageDtype(torch.float32)])

class NYUV2Dataset(Dataset):

    def __init__(self, dataframe, options=None):
        self.dataframe = dataframe
        self.dim = (256, 256)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        image = row[0]
        depth = row[1]
        return self.load(image, depth)

    def __len__(self):
        return len(self.dataframe)

    def load(self, image, depth_map):
        image = rearrange(image, 'C W H -> H W C')
        image = cv2.resize(image, self.dim)
        image = image / 255.0
        image = transformation()(torch.tensor(image))
        depth_map = rearrange(depth_map, 'W H -> H W')
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, -1)
        depth_map = transformation()(torch.tensor(depth_map))
        depth_map = (depth_map - torch.min(depth_map)) / (torch.max(depth_map) - torch.min(depth_map))
        return (image, depth_map)