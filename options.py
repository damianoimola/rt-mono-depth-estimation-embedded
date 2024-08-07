import os
import argparse
file_dir = os.path.dirname(__file__)

class Options:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='program options')

        # MISC
        self.parser.add_argument(
            '--device',
            type=str,
            help='device to use',
            default='cuda',
            choices=['cuda', 'cpu'])

        self.parser.add_argument(
            '--num_workers',
            type=int,
            help='number of dataloader workers',
            default=os.cpu_count()//2)

        self.parser.add_argument(
            '--logger',
            type=str,
            help='number of batches between each log',
            default="csv",
            choices=['csv', 'tensorboard', 'wandb'])

        self.parser.add_argument(
            '--checkpoint_dir',
            type=str,
            help='directory in which save the model checkpoint',
            default="checkpoints")


        # DATA
        self.parser.add_argument(
            '--data_path',
            type=str,
            help='path to the training data',
            default="D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\nyu\\nyu_data",
            choices=[
                # nyu_v2
                "D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\nyu\\nyu.mat",
                "D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\nyu\\nyu_data",
                # diode
                "D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\kitti",
                # diode
                "D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\diode\\val",
                "D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\diode\\indoors",
            ])

        self.parser.add_argument(
            '--dataset',
            type=str,
            help='dataset to train on',
            default='nyu_v2_folder',
            choices=['kitti_depth', 'nyu_v2', 'diode_val', 'nyu_v2_folder'])

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
            default=150.0)


        # MODEL
        self.parser.add_argument(
            '--model_name',
            type=str,
            help='name of the model to be used',
            default='monodert',
            choices=['unet', 'monodepthrt', 'monodert'])

        self.parser.add_argument(
            '--batch_size',
            type=int,
            help='batch size',
            default=12)

        self.parser.add_argument(
            '--learning_rate',
            type=float,
            help='learning rate',
            default=0.0003)

        self.parser.add_argument(
            '--num_epochs',
            type=int,
            help='number of epochs',
            default=200)


        # CAM INFERENCE
        self.parser.add_argument(
            '--ckpt',
            type=str,
            help='which checkpoint to load',
            default="mde60e_kaggle")

        self.parser.add_argument(
            '--fps_verbose',
            type=bool,
            help='only for online mode; if the current number of FPS must be shown in command line',
            default=False)

        self.parser.add_argument(
            '--online',
            type=bool,
            help='improve performances, sampling at 60 FPS',
            default=True)

        self.parser.add_argument(
            '--in_root',
            type=bool,
            help='whether the checkpoint is in the root or in the it dedicated folder',
            default=False)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options