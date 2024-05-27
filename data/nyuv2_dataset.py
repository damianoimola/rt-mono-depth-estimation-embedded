import random

import cv2
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from computer_vision_project.data.augmentation import augment_data


# def to_float32():
#     return transforms.Compose([transforms.ConvertImageDtype(torch.float32)])

class NYUV2Dataset(Dataset):

    def __init__(self, dataframe, options):
        self.dataframe = dataframe
        self.dim = (options.height, options.width)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        image = row[0]
        depth = row[1]
        return self.load(image, depth)

    def __len__(self):
        return len(self.dataframe)

    def load(self, image, depth_map):
        flip = random.random() > 0.5

        # image processing
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute((0, 2, 1))
        image = transforms.Resize(size=self.dim)(image)
        image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(image)
        if flip: image = TF.hflip(image)

        # depth processing
        depth_map = torch.tensor(depth_map, dtype=torch.float32)
        depth_map = depth_map.unsqueeze(0)
        depth_map = depth_map.permute((0, 2, 1))
        depth_map = transforms.Resize(size=self.dim)(depth_map)
        if flip: depth_map = TF.hflip(depth_map)

        # normalize depth
        depth_map = (depth_map - torch.min(depth_map)) / (torch.max(depth_map) - torch.min(depth_map))

        return (image, depth_map)