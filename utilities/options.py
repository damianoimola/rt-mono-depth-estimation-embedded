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
        self.parser.add_argument('--data_path', type=str, help='path to the training data', default=os.path.join(file_dir, 'kitti_data'))
        self.parser.add_argument('--log_dir', type=str, help='log directory', default=os.path.join(os.path.expanduser('~'), 'tmp'))
        self.parser.add_argument('--dataset', type=str, help='dataset to train on', default='diode_val', choices=['kitti_depth', 'nyu', 'diode_val'])
        self.parser.add_argument('--height', type=int, help='input image height', default=256)
        self.parser.add_argument('--width', type=int, help='input image width', default=256)
        self.parser.add_argument('--min_depth', type=float, help='minimum depth', default=0.1)
        self.parser.add_argument('--max_depth', type=float, help='maximum depth', default=100.0)
        self.parser.add_argument('--batch_size', type=int, help='batch size', default=12)
        self.parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.0001)
        self.parser.add_argument('--num_epochs', type=int, help='number of epochs', default=20)
        self.parser.add_argument('--no_cuda', help='if set disables CUDA', action='store_true')
        self.parser.add_argument('--num_workers', type=int, help='number of dataloader workers', default=12)
        self.parser.add_argument('--logger', type=str, help='number of batches between each log', default=1, choices=['csv', 'tensorboard', 'wandb'])
        self.parser.add_argument('--log_frequency', type=int, help='number of batches between each log', default=1)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options