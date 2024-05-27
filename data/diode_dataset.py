import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class DiodeDataset(Dataset):
    def __init__(self, dataframe, options):
        self.dataframe = dataframe
        self.max_depth = options.max_depth
        self.min_depth = options.min_depth
        self.dim = (options.height, options.width)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        image_path = row[0]
        depth_path = row[1]
        mask_path = row[2]
        return self.load(image_path, depth_path, mask_path)

    def __len__(self):
        return len(self.dataframe)

    def load(self, image_path, depth_map, mask, eps=1e-06):
        flip = random.random() > 0.5

        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.dim)
            image = self.to_float32()(torch.tensor(image))
        except Exception as e:
            print(f'Error loading input image: {e}')


        try:
            depth_map = np.load(depth_map).squeeze()
        except Exception as e:
            print(f'Error loading depth map: {e}')


        try:
            mask = np.load(mask)
            mask = mask > 0
        except Exception as e:
            print(f'Error loading mask: {e}')


        try:
            max_depth = min(self.max_depth, np.percentile(depth_map, 99))
            depth_map = np.clip(depth_map, self.min_depth, max_depth)
            depth_map = np.log(depth_map, where=mask)
            depth_map = np.ma.masked_where(~mask, depth_map)
            depth_map = np.clip(depth_map, self.min_depth, np.log(max_depth))
            depth_map = cv2.resize(depth_map, self.dim)
            depth_map = np.expand_dims(depth_map, axis=2)
            depth_map = self.to_float32()(torch.tensor(depth_map))
            depth_map = depth_map.clamp(min=eps)
            depth_map = (depth_map - torch.min(depth_map)) / (torch.max(depth_map) - torch.min(depth_map))
        except Exception as e:
            print(f'Error preprocessing depth map: {e}')

        return image, depth_map

    def to_float32(self):
        return transforms.Compose([transforms.ConvertImageDtype(torch.float32)])