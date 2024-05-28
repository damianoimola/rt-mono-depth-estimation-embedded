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

        # misc
        self.device = self.opt.device
        self.size = (self.opt.channels, self.opt.height, self.opt.width)
        self.experiment_name = f'{self.opt.model_name}-d={self.opt.dataset}-lr={self.opt.learning_rate}-e={self.opt.num_epochs}'
        self.loaded = False

        # data
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = (None, None, None)

        # trainer settings
        self.checkpoint_path = None
        self.logger = get_logger(self.experiment_name)
        self.version = self.logger.version
        self.cb_list = get_callbacks()
        self.trainer = self.trainer = L.Trainer(max_epochs=self.opt.num_epochs, log_every_n_steps=1, logger=self.logger,
                                                callbacks=self.cb_list, accelerator='auto')

        # model loading
        self.plain_model = None
        self.lit_model = None
        self.select_model()

    def select_model(self):
        if self.opt.model_name == 'unet':
            self.plain_model = UNet(3, 1)
            self.lit_model = LitUNet(self.plain_model.to(self.device), self.opt.height, self.opt.learning_rate)
            return UNet, LitUNet

        elif self.opt.model_name == 'monodepthrt':
            self.plain_model = MonoDepthRT(self.size, base_channels=64, network_depth=4, training=True)
            self.lit_model = LitMonoDERT(self.plain_model.to(self.device), self.opt.height, self.opt.learning_rate)
            return MonoDepthRT, LitMonoDERT

        elif self.opt.model_name == 'monodert':
            return None, None

    def train(self):
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = self.get_data()

        if self.checkpoint_path:
            self.trainer.fit(model=self.lit_model, ckpt_path=self.checkpoint_path,
                             train_dataloaders=self.train_dataloader, val_dataloaders=self.valid_dataloader)
        else:
            self.trainer.fit(model=self.lit_model, train_dataloaders=self.train_dataloader, val_dataloaders=self.valid_dataloader)

        self.loaded = True

    def eval(self):
        self.trainer.test(self.lit_model, dataloaders=self.test_dataloader)

    def save(self):
        self.trainer.save_checkpoint(f'{self.opt.checkpoint_dir}/{self.experiment_name}.ckpt')

    def load(self, checkpoint_name):
        self.checkpoint_path = f'{self.opt.checkpoint_dir}/{checkpoint_name}.ckpt'

        _, lit_class = self.select_model()

        self.lit_model = lit_class.load_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
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

        fig, axes = plt.subplots(3, 2, figsize=(15, 7))

        gs = axes[0, 0].get_gridspec()
        for ax in axes[:2, 0]: ax.remove()
        ax = fig.add_subplot(gs[:2, 0])
        # ax = axes[0, :]
        ax.set_title('Losses per epoch')
        ax.plot(results['train_total_loss_epoch'].dropna(ignore_index=True), label='train total loss', color='dodgerblue')
        ax.plot(results['train_mse_loss_epoch'].dropna(ignore_index=True), label='train MSE loss', color='orangered')
        ax.plot(results['train_berhu_loss_epoch'].dropna(ignore_index=True), label='train BerHu loss', color='goldenrod')
        ax.plot(results['train_eas_loss_epoch'].dropna(ignore_index=True), label='train EAS loss', color='darkgreen')
        ax.plot(results['train_silog_loss_epoch'].dropna(ignore_index=True), label='train SiLog loss', color='palegreen')
        ax.plot(results['train_sasinv_loss_epoch'].dropna(ignore_index=True), label='train SaSInv loss', color='coral')

        ax.plot(results['valid_total_loss'].dropna(ignore_index=True), label='validation total loss', color='dodgerblue', alpha=0.5)
        ax.plot(results['valid_mse_loss'].dropna(ignore_index=True), label='validation MSE loss', color='orangered', alpha=0.5)
        ax.plot(results['valid_berhu_loss'].dropna(ignore_index=True), label='validation BerHu loss', color='goldenrod', alpha=0.5)
        ax.plot(results['valid_eas_loss'].dropna(ignore_index=True), label='validation EAS loss', color='darkgreen', alpha=0.5)
        ax.plot(results['valid_silog_loss'].dropna(ignore_index=True), label='validation SiLog loss', color='palegreen', alpha=0.5)
        ax.plot(results['valid_sasinv_loss'].dropna(ignore_index=True), label='validation SaSInv loss', color='coral', alpha=0.5)
        ax.legend(loc='upper right', ncol=2)
        # ax.set_ylim(0, 2)
        ax.grid()



        ax = axes[0][1]
        ax.set_title('Deltas per epoch')
        # plt.rcParams['text.usetex'] = True
        ax.plot(results['train_delta1_epoch'].dropna(ignore_index=True), color='royalblue', label='δ_1 train')
        ax.plot(results['train_delta2_epoch'].dropna(ignore_index=True), color='cornflowerblue', label='δ_2 train')
        ax.plot(results['train_delta3_epoch'].dropna(ignore_index=True), color='lightsteelblue', label='δ_3 train')

        ax.plot(results['valid_delta1'].dropna(ignore_index=True), color='orangered', label='δ_1 validation')
        ax.plot(results['valid_delta2'].dropna(ignore_index=True), color='coral', label='δ_2 validation')
        ax.plot(results['valid_delta3'].dropna(ignore_index=True), color='darksalmon', label='δ_3 validation')
        ax.legend()
        ax.grid()
        # plt.rcParams['text.usetex'] = False



        ax = axes[1][1]
        ax.set_title('Absolute Relative Error per epoch')
        ax.plot(results['train_abs_rel_epoch'].dropna(ignore_index=True), color='crimson', label='(AbsRel) absolute relative error train')
        ax.plot(results['valid_abs_rel'].dropna(ignore_index=True), color='crimson', label='(AbsRel) absolute relative error validation', alpha=0.7)
        ax.legend()
        ax.grid()



        gs = axes[2, 0].get_gridspec()
        for ax in axes[2, :2]: ax.remove()
        ax = fig.add_subplot(gs[2, :2])
        ax.set_title('Errors per epoch')
        ax.plot(results['train_mae_epoch'].dropna(ignore_index=True), color='royalblue', label='(MAE) mean absolute error train')
        ax.plot(results['train_log_mae_epoch'].dropna(ignore_index=True), color='cornflowerblue', label='(MAElog) mean absolute log error train')
        ax.plot(results['train_rmse_epoch'].dropna(ignore_index=True), color='darkgreen', label='(RMSE) root mean squared error')
        ax.plot(results['train_log_rmse_epoch'].dropna(ignore_index=True), color='forestgreen', label='(RMSElog) root mean squared log error')

        ax.plot(results['valid_mae'].dropna(ignore_index=True), color='royalblue', label='(MAE) mean absolute error validation', alpha=0.7)
        ax.plot(results['valid_log_mae'].dropna(ignore_index=True), color='cornflowerblue', label='(MAElog) mean absolute log error validation', alpha=0.7)
        ax.plot(results['valid_rmse'].dropna(ignore_index=True), color='darkgreen', label='(RMSE) root mean squared error validation', alpha=0.7)
        ax.plot(results['valid_log_rmse'].dropna(ignore_index=True), color='forestgreen', label='(RMSElog) root mean squared log error validation', alpha=0.7)
        # ax.set_ylim(0, 2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)
        ax.grid()

        fig.tight_layout(pad=2.0)

        plt.show()

    def plot_batch_predictions(self, save=False, title=None):
        if not self.loaded:
            print('LOAD SOME CHECKPOINT')
            return
        self.set_train()

        if title is None: title = f'V{self.version}-{self.experiment_name}'

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