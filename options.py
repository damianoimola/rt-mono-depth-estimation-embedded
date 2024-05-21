# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: D:\Python\univ_proj\computer_vision\computer_vision_project\utilities\options.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2024-05-15 20:17:04 UTC (1715804224)

import os
import argparse
file_dir = os.path.dirname(__file__)

class Options:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='program options')


        self.parser.add_argument(
            '--device',
            type=str,
            help='device to use',
            default='cuda',
            choices=['cuda', 'cpu'])

        self.parser.add_argument(
            '--data_path',
            type=str,
            help='path to the training data',
            default="D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\diode\\val",
            choices=[
                "D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\nyu\\nyu.mat",
                "D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\kitti",
                "D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\diode\\val",
            ]
        )
        self.parser.add_argument(
            '--model_name',
            type=str,
            help='name of the model to be used',
            default='unet',
            choices=['unet', 'monodepthrt', 'monodert'])
        self.parser.add_argument(
            '--dataset',
            type=str,
            help='dataset to train on',
            default='diode_val',
            choices=['kitti_depth', 'nyu_v2', 'diode_val'])
        self.parser.add_argument(
            '--height',
            type=int,
            help='input image height',
            default=256)
        self.parser.add_argument(
            '--width',
            type=int,
            help='input image width',
            default=256)
        self.parser.add_argument(
            '--channels',
            type=int,
            help='input image channels',
            default=3)
        self.parser.add_argument(
            '--min_depth',
            type=float,
            help='minimum depth',
            default=0.1)
        self.parser.add_argument(
            '--max_depth',
            type=float,
            help='maximum depth',
            default=200.0)
        self.parser.add_argument(
            '--batch_size',
            type=int,
            help='batch size',
            default=12)
        self.parser.add_argument(
            '--learning_rate',
            type=float,
            help='learning rate',
            default=0.0001)
        self.parser.add_argument(
            '--num_epochs',
            type=int,
            help='number of epochs',
            default=1)
        self.parser.add_argument(
            '--num_workers',
            type=int,
            help='number of dataloader workers',
            default=os.cpu_count()//2)
        self.parser.add_argument(
            '--logger',
            type=str,
            help='number of batches between each log',
            default="csv", choices=['csv', 'tensorboard', 'wandb'])

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options