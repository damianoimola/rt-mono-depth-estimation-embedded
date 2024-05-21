import lightning as L
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from computer_vision_project.data.dataset_handler import DatasetHandler
from computer_vision_project.model.unet.lightning_model import LitUNet
from computer_vision_project.model.unet.net import UNet
from computer_vision_project.utilities.plots import plot_predictions
from model.monodepth_rt.lightning_model import LitMonoDERT
from model.monodepth_rt.monodepthrt import MonoDepthRT
from utilities.callbacks import get_callbacks
from utilities.logger import get_logger

class Trainer:

    def __init__(self, options):
        self.opt = options
        self.device = self.opt.device
        self.size = (self.opt.channels, self.opt.height, self.opt.width)
        self.experiment_name = f'{self.opt.model_name}-d={self.opt.dataset}-lr={self.opt.learning_rate}-e={self.opt.num_epochs}'
        self.loaded = False
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = (None, None, None)
        self.logger = get_logger(self.experiment_name)
        self.version = self.logger.version
        self.cb_list = get_callbacks()
        self.trainer = L.Trainer(max_epochs=self.opt.num_epochs, log_every_n_steps=1, logger=self.logger, callbacks=self.cb_list, accelerator='auto')
        self.plain_model = None
        self.lit_model = None
        self.select_model()

    def select_model(self):
        if self.opt.model_name == 'unet':
            self.plain_model = UNet(3, 1)
            self.lit_model = LitUNet(self.plain_model.to(self.device), self.opt.height, self.opt.learning_rate)

        elif self.opt.model_name == 'monodepthrt':
            self.plain_model = MonoDepthRT(self.size, base_channels=64, network_depth=4, training=True)
            self.lit_model = LitMonoDERT(self.plain_model.to(self.device), self.opt.height, self.opt.learning_rate)

        elif self.opt.model_name == 'monodert':
            return

    def train(self):
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = self.get_data()
        self.trainer.fit(model=self.lit_model, train_dataloaders=self.train_dataloader, val_dataloaders=self.valid_dataloader)
        self.loaded = True

    def eval(self):
        self.trainer.test(self.lit_model, dataloaders=self.test_dataloader)

    def save(self):
        self.trainer.save_checkpoint(f'{self.opt.checkpoint_dir}/{self.experiment_name}.ckpt')

    def load(self, checkpoint_name):
        self.lit_model = LitMonoDERT.load_from_checkpoint(
            checkpoint_path=f'{self.opt.checkpoint_dir}/{checkpoint_name}.ckpt',
            plain_model=self.plain_model,
            size=self.opt.width,
            lr=self.opt.learning_rate
        )
        self.loaded = True

    def set_eval(self):
        self.lit_model.eval()

    def set_train(self):
        self.lit_model.train()

    def predict(self, input):
        return self.lit_model(input.to(self.lit_model.device))

    def plot_metrics(self, path_to_metrics=None):
        if path_to_metrics is None:
            if not self.loaded:
                print('LOAD SOME CHECKPOINT')
                return
            path_to_metrics = self.logger.log_dir

        results = pd.read_csv(path_to_metrics)

        import matplotlib
        matplotlib.use('TkAgg')
        plt.style.use('default')
        plt.rcParams['text.usetex'] = False

        fig, axes = plt.subplots(2, 2, figsize=(15, 7))

        ax = axes[0][0]
        ax.set_title('Losses per epoch')
        ax.plot(results['train_total_loss_epoch'].dropna(ignore_index=True), label='train total loss', color='dodgerblue')
        ax.plot(results['train_berhu_loss_epoch'].dropna(ignore_index=True), label='train BerHu loss', color='goldenrod')
        ax.plot(results['train_eas_loss_epoch'].dropna(ignore_index=True), label='train EAS loss', color='darkgreen')
        ax.plot(results['train_silog_loss_epoch'].dropna(ignore_index=True), label='train SiLog loss', color='palegreen')
        ax.plot(results['train_sasinv_loss_epoch'].dropna(ignore_index=True), label='train SaSInv loss', color='coral')

        ax.plot(results['valid_total_loss_epoch'].dropna(ignore_index=True), label='validation total loss', color='dodgerblue', alpha=0.7)
        ax.plot(results['valid_berhu_loss_epoch'].dropna(ignore_index=True), label='validation BerHu loss', color='goldenrod', alpha=0.7)
        ax.plot(results['valid_eas_loss_epoch'].dropna(ignore_index=True), label='validation EAS loss', color='darkgreen', alpha=0.7)
        ax.plot(results['valid_silog_loss_epoch'].dropna(ignore_index=True), label='validation SiLog loss', color='palegreen', alpha=0.7)
        ax.plot(results['valid_sasinv_loss_epoch'].dropna(ignore_index=True), label='validation SaSInv loss', color='coral', alpha=0.7)
        ax.legend()
        ax.set_ylim(0, 2)
        ax.grid()



        ax = axes[0][1]
        ax.set_title('Deltas per epoch')
        plt.rcParams['text.usetex'] = True
        ax.plot(results['train_delta1_epoch'].dropna(ignore_index=True), color='royalblue', label='$\\delta_1$ train')
        ax.plot(results['train_delta2_epoch'].dropna(ignore_index=True), color='cornflowerblue', label='$\\delta_2$ train')
        ax.plot(results['train_delta3_epoch'].dropna(ignore_index=True), color='lightsteelblue', label='$\\delta_3$ train')

        ax.plot(results['valid_delta1_epoch'].dropna(ignore_index=True), color='orangered', label='$\\delta_1$ validation')
        ax.plot(results['valid_delta2_epoch'].dropna(ignore_index=True), color='coral', label='$\\delta_2$ validation')
        ax.plot(results['valid_delta3_epoch'].dropna(ignore_index=True), color='darksalmon', label='$\\delta_3$ validation')
        ax.legend()
        ax.grid()
        plt.rcParams['text.usetex'] = False



        ax = axes[1][0]
        ax.set_title('Errors per epoch')
        ax.plot(results['train_mae_epoch'].dropna(ignore_index=True), color='royalblue', label='(MAE) mean absolute error train')
        ax.plot(results['train_log_mae_epoch'].dropna(ignore_index=True), color='cornflowerblue', label='(MAElog) mean absolute log error train')
        ax.plot(results['train_rmse_epoch'].dropna(ignore_index=True), color='darkgreen', label='(RMSE) root mean squared error')
        ax.plot(results['train_log_rmse_epoch'].dropna(ignore_index=True), color='forestgreen', label='(RMSElog) root mean squared log error')

        ax.plot(results['valid_mae_epoch'].dropna(ignore_index=True), color='royalblue', label='(MAE) mean absolute error validation', alpha=0.7)
        ax.plot(results['valid_log_mae_epoch'].dropna(ignore_index=True), color='cornflowerblue', label='(MAElog) mean absolute log error validation', alpha=0.7)
        ax.plot(results['valid_rmse_epoch'].dropna(ignore_index=True), color='darkgreen', label='(RMSE) root mean squared error validation', alpha=0.7)
        ax.plot(results['valid_log_rmse_epoch'].dropna(ignore_index=True), color='forestgreen', label='(RMSElog) root mean squared log error validation', alpha=0.7)
        ax.set_ylim(0, 2)
        ax.legend()
        ax.grid()



        ax = axes[1][1]
        ax.set_title('Absolute Relative Error per epoch')
        ax.plot(results['train_abs_rel_epoch'].dropna(ignore_index=True), color='crimson', label='(AbsRel) absolute relative error train')
        ax.plot(results['valid_abs_rel_epoch'].dropna(ignore_index=True), color='crimson', label='(AbsRel) absolute relative error validation', alpha=0.7)
        ax.legend()
        ax.grid()
        plt.show()

    def plot_batch_predictions(self, save=False, title=None):
        if not self.loaded:
            print('LOAD SOME CHECKPOINT')
            return
        self.set_train()
        if title is None:
            title = f'V{self.version}-{self.experiment_name}'
        if self.test_dataloader is None:
            _, _, self.test_dataloader = self.get_data()
        inputs, target = next(iter(self.test_dataloader))
        all_preds = self.predict(inputs)
        plot_predictions(inputs, target, all_preds, save=save, title=title)

    def get_data(self):
        print('### INITIALIZING DATALOADERS')
        h = DatasetHandler(self.opt.data_path, self.opt)
        if self.opt.dataset == 'nyu_v2':
            train_data, valid_data, test_data = h.load_nyu_v2()
        elif self.opt.dataset == 'diode_val':
            train_data, valid_data, test_data = h.load_diode()
        else:
            train_data, valid_data, test_data = (None, None, None)

        train_dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)
        valid_dataloader = DataLoader(valid_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)

        print('### DATALOADERS INITIALIZED')
        return (train_dataloader, valid_dataloader, test_dataloader)