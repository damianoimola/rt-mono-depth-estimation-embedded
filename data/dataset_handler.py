import os
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from data.diode_dataset import DiodeDataset
from data.nyuv2_dataset import NYUV2Dataset
from data.nyuv2_folder_dataset import NYUV2DatasetFolder


class DatasetHandler:

    def __init__(self, path, options):
        self.path = path
        self.opt = options

    def load_nyu_v2(self):
        f = h5py.File(self.path)

        data = {
            'image': [x for x in f['images']],
            'depth': [x for x in f['depths']]
        }

        dataframe = pd.DataFrame(data)
        train_df, valid_df = train_test_split(dataframe, test_size=0.1, shuffle=True)
        train_df, test_df = train_test_split(train_df, test_size=0.1, shuffle=True)

        print(f'train dataset samples {len(train_df)}')
        print(f'valid dataset samples {len(valid_df)}')
        print(f'test dataset samples {len(test_df)}')

        train_data = NYUV2Dataset(dataframe=train_df, options=self.opt)
        valid_data = NYUV2Dataset(dataframe=valid_df, options=self.opt)
        test_data = NYUV2Dataset(dataframe=test_df, options=self.opt)

        return (train_data, valid_data, test_data)

    def load_nyu_v2_folders(self):
        # loading paths
        train_files = pd.read_csv(self.path+"\\data\\nyu2_train.csv")
        test_files = pd.read_csv(self.path+"\\data\\nyu2_test.csv")

        train_path_df = pd.DataFrame(train_files)
        train_path, valid_path = train_test_split(train_path_df, test_size=0.1, shuffle=True)

        test_path = pd.DataFrame(test_files)

        print(f'train dataset samples {len(train_path)}')
        print(f'valid dataset samples {len(valid_path)}')
        print(f'test dataset samples {len(test_path)}')

        print(test_path.head())

        train_data = NYUV2DatasetFolder(dataframe=train_path, options=self.opt, base_path=self.path)
        valid_data = NYUV2DatasetFolder(dataframe=valid_path, options=self.opt, base_path=self.path)
        test_data = NYUV2DatasetFolder(dataframe=test_path, options=self.opt, base_path=self.path)

        return (train_data, valid_data, test_data)

    def load_diode(self):
        filelist = []

        for root, dirs, files in os.walk(self.path):
            for file in files:
                filelist.append(os.path.join(root, file))

        filelist.sort()

        data = {
            'image': [x for x in filelist if x.endswith('.png')],
            'depth': [x for x in filelist if x.endswith('_depth.npy')],
            'mask': [x for x in filelist if x.endswith('_depth_mask.npy')]
        }

        path_df = pd.DataFrame(data)
        train_path, valid_path = train_test_split(path_df, test_size=0.1, shuffle=True)
        train_path, test_path = train_test_split(train_path, test_size=0.1, shuffle=True)

        print(f'train dataset samples {len(train_path)}')
        print(f'valid dataset samples {len(valid_path)}')
        print(f'test dataset samples {len(test_path)}')

        train_data = DiodeDataset(dataframe=train_path, options=self.opt)
        valid_data = DiodeDataset(dataframe=valid_path, options=self.opt)
        test_data = DiodeDataset(dataframe=test_path, options=self.opt)

        return (train_data, valid_data, test_data)