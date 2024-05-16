import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CustomDataset(Dataset):
    def __init__(self, dataframe, options):
        self.dataframe = dataframe
        self.max_depth = options.max_depth
        self.min_depth = options.min_depth
        self.dim = (options.height, options.width)
        self.debug = True


    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        image_path = row[0]
        depth_path = row[1]
        mask_path = row[2]

        return self.load(image_path, depth_path, mask_path)

    def __len__(self):
        return len(self.dataframe)

    def load(self, image_path, depth_map, mask):
        try:
            # load image from path
            image_ = cv2.imread(image_path)
            image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
            image_ = cv2.resize(image_, self.dim)
            image_ = self.transformation()(torch.tensor(image_))
        except Exception as e:
            print(f"Error loading input image: {e}")
            return None, None

        try:
            # load depth map
            depth_map = np.load(depth_map).squeeze()
        except Exception as e:
            print(f"Error loading depth map: {e}")
            return None, None

        try:
            # load mask (of the depth map)
            mask = np.load(mask)
            mask = mask > 0
        except Exception as e:
            print(f"Error loading mask: {e}")
            return None, None


        try:
            ### MASKED VALUES PREPROCESSING
            # 99° percentile = the value in which the 99 percent of observation falls below
            max_depth = min(self.max_depth, np.percentile(depth_map, 99))
            depth_map = np.clip(depth_map, self.min_depth, max_depth)

            # apply log to every value, except to the one unmasked
            depth_map = np.log(depth_map, where=mask)


            ### UNMASKED VALUES PREPROCESSING
            depth_map = np.ma.masked_where(~mask, depth_map)
            depth_map = np.clip(depth_map, self.min_depth, np.log(max_depth))

            depth_map = cv2.resize(depth_map, self.dim)
            depth_map = np.expand_dims(depth_map, axis=2)
            depth_map = self.transformation()(torch.tensor(depth_map))

            # normalization
            # epsilon = 1e-4
            # depth_map = (depth_map - torch.min(depth_map))/(torch.max(depth_map) - torch.min(depth_map) + epsilon)
            # depth_map = np.linalg.norm(depth_map)
            depth_map = depth_map/torch.max(depth_map)
        except Exception as e:
            print(f"Error preprocessing depth map: {e}")
            return None, None


        return image_, depth_map

    def transformation(self):
        return transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # transforms.Lambda(lambda x: (x+1)/2),
            transforms.ConvertImageDtype(torch.float32),
        ])