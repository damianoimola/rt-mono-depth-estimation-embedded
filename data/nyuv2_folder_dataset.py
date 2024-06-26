import random

import cv2
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class NYUV2DatasetFolder(Dataset):
    def __init__(self, dataframe, options, base_path):
        self.dataframe = dataframe
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
            # read
            image = cv2.imread(self.base_path + f"\\{image_path}")
            # cv2 -> human colors
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # [0, 1]
            image = image / 255.0
            # to tensor
            image = torch.tensor(image, dtype=torch.float32)
            # to C H W
            image = image.permute((2, 0, 1))
            # manageable sizes
            image = transforms.Resize(size=self.dim)(image)
            # augmentation
            image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(image)
            if flip: image = TF.hflip(image)
        except Exception as e:
            print(f'Error loading input image: {e}')

        depth_map = None
        try:
            depth_map = cv2.imread(self.base_path + f"\\{depth_map_path}", 0)
            depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(-1)
            depth_map = depth_map.permute((2, 0, 1))
            depth_map = transforms.Resize(size=self.dim)(depth_map)
            if flip: depth_map = TF.hflip(depth_map)
            depth_map = depth_map.clamp(min=eps)
            depth_map = (depth_map - torch.min(depth_map)) / (torch.max(depth_map) - torch.min(depth_map))
        except Exception as e:
            print(f'Error loading depth map: {e}')

        return image, depth_map
