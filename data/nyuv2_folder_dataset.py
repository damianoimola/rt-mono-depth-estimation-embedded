import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class NYUV2DatasetFolder(Dataset):
    def __init__(self, dataframe, options, base_path):
        self.dataframe = dataframe
        self.max_depth = options.max_depth
        self.min_depth = options.min_depth
        self.dim = (options.height, options.width)
        self.base_path = base_path

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        image_path = row[0]
        depth_path = row[1]
        return self.load(image_path, depth_path)

    def __len__(self):
        return len(self.dataframe)

    def load(self, image_path, depth_map_path, eps=1e-06):
        flip = random.random() > 0.5

        image = None
        try:
            image = cv2.imread(self.base_path + f"\\{image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.dim)
            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute((2, 0, 1))
        except Exception as e:
            print(f'Error loading input image: {e}')

        depth_map = None
        try:
            depth_map = cv2.imread(self.base_path + f"\\{depth_map_path}", 0)
            depth_map = cv2.resize(depth_map, self.dim)
            depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(-1)
            depth_map = depth_map.permute((2, 0, 1))
            depth_map = depth_map.clamp(min=eps)
            depth_map = (depth_map - torch.min(depth_map)) / (torch.max(depth_map) - torch.min(depth_map))
        except Exception as e:
            print(f'Error loading depth map: {e}')

        return image, depth_map
