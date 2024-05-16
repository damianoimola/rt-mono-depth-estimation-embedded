import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from computer_vision_project.data.custom_dataset import CustomDataset
from model.monodepth_rt.monodepthrt import MonoDepthRT
from model.monodepth_rt.lightning_model import LitMonoDERT

from utilities.logger import get_logger
from utilities.callbacks import get_callbacks

import lightning as L


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.device = self.opt.device

        # data
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = self.get_data()

        # lit trainer
        logs_name = f"d={self.opt.dataset}-lr={self.opt.learning_rate}-e={self.opt.num_epochs}"
        self.logger = get_logger(logs_name)
        self.cb_list = get_callbacks()
        self.trainer = L.Trainer(
            max_epochs=self.opt.epochs,
            log_every_n_steps=1,
            logger=self.logger,
            callbacks=self.cb_list,
            accelerator="auto")

        # models
        self.size = (self.opt.channels, self.opt.height, self.opt.width)
        self.plain_model = MonoDepthRT(self.size, base_channels=64, network_depth=4, training=True)
        self.lit_model = LitMonoDERT(self.plain_model.to(self.device), self.opt.height)




    def train(self):
        self.trainer.fit(
            model=self.lit_model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader
        )

    def eval(self):
        self.trainer.test(self.lit_model, dataloaders=self.test_dataloader)

    def save(self):
        self.trainer.save_checkpoint(f"checkpoints/monodepth-rt-d={self.opt.dataset}-lr={self.opt.learning_rate}-e={self.opt.num_epochs}.ckpt")

    def load(self, checkpoint_name):
        self.lit_model = LitMonoDERT.load_from_checkpoint(checkpoint_path=f"checkpoints/{checkpoint_name}.ckpt",
                                                          model=self.plain_model,
                                                          size=self.opt.width)

    def set_eval(self):
        self.lit_model.eval()

    def set_train(self):
        self.lit_model.train()

    def predict(self, input):
        return self.lit_model(input)

    def get_data(self):
        path = self.opt.data_path

        filelist = []
        for root, dirs, files in os.walk(path):
            for file in files:
                filelist.append(os.path.join(root, file))

        filelist.sort()
        data = {
            "image": [x for x in filelist if x.endswith(".png")],
            "depth": [x for x in filelist if x.endswith("_depth.npy")],
            "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
        }
        path_df = pd.DataFrame(data)
        train_path, valid_path = train_test_split(path_df, test_size=0.2, shuffle=True)
        train_path, test_path = train_test_split(train_path, test_size=0.2, shuffle=True)

        print(f"train dataset samples {len(train_path)}")
        print(f"valid dataset samples {len(valid_path)}")
        print(f"test dataset samples {len(test_path)}")

        train_data = CustomDataset(dataframe=train_path)
        valid_data = CustomDataset(dataframe=valid_path)
        test_data = CustomDataset(dataframe=test_path)

        train_dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)
        valid_dataloader = DataLoader(valid_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)

        return train_dataloader, valid_dataloader, test_dataloader