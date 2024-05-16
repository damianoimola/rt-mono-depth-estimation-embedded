import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="program options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 # default=os.path.join(file_dir, "kitti_data")
                                 default="D:\\Python\\univ_proj\\computer_vision\\computer_vision_notebook\\diode\\val\\indoors"
                                 )
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="diode_val",
                                 choices=["kitti_depth", "nyu", "diode_val"])
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=256)
        self.parser.add_argument("--channels",
                                 type=int,
                                 help="input image channels",
                                 default=3)
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=6)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)


        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=0)


        # LOGGING options
        self.parser.add_argument("--logger",
                                 type=str,
                                 help="number of batches between each log",
                                 default=1,
                                 choices=["csv", "tensorboard", "wandb"])
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each log",
                                 default=1)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options