import lightning as L
import torch.nn as nn
import torch
from einops import rearrange
from torch import optim
from torchvision.transforms import transforms
from utilities.losses import combined_loss
from utilities.metrics import Metrics

class LitUNet(L.LightningModule):

    def __init__(self, plain_model, size, lr):
        super(LitUNet, self).__init__()
        self.model = plain_model
        self.transform = transforms.Resize(size=(size, size))
        self.lr = lr
        self.metrics = Metrics()
        self.with_scheduler = True

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        log_dict = {}

        gt = self.transform(y)

        eas_loss, berhu_loss, silog_loss, sasinv_loss, ssim_loss, gd_loss, tv_loss, mse_loss = combined_loss(pred, gt)
        loss = sum([eas_loss, berhu_loss, ssim_loss, gd_loss, tv_loss, silog_loss])

        log_dict['train_total_loss'] = loss.item()
        log_dict['train_ssim_loss'] = ssim_loss.item()
        log_dict['train_mse_loss'] = mse_loss.item()
        log_dict['train_eas_loss'] = eas_loss.item()
        log_dict['train_gd_loss'] = gd_loss.item()
        log_dict['train_tv_loss'] = tv_loss.item()
        log_dict['train_berhu_loss'] = berhu_loss.item()
        log_dict['train_silog_loss'] = silog_loss.item()
        log_dict['train_sasinv_loss'] = sasinv_loss.item()

        for i in range(pred.size()[0]):
            self.metrics.update(gt[i].detach().cpu(), pred[i].detach().cpu())
        log_dict.update(self.metrics.retrieve_avg('train'))
        self.metrics.clean()

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        log_dict = {}

        gt = self.transform(y)

        eas_loss, berhu_loss, silog_loss, sasinv_loss, ssim_loss, gd_loss, tv_loss, mse_loss = combined_loss(pred, gt)
        loss = sum([eas_loss, berhu_loss, ssim_loss, gd_loss, tv_loss, silog_loss])

        log_dict['valid_total_loss'] = loss.item()
        log_dict['valid_ssim_loss'] = ssim_loss.item()
        log_dict['valid_mse_loss'] = mse_loss.item()
        log_dict['valid_eas_loss'] = eas_loss.item()
        log_dict['valid_gd_loss'] = gd_loss.item()
        log_dict['valid_tv_loss'] = tv_loss.item()
        log_dict['valid_berhu_loss'] = berhu_loss.item()
        log_dict['valid_silog_loss'] = silog_loss.item()
        log_dict['valid_sasinv_loss'] = sasinv_loss.item()

        for i in range(pred.size()[0]):
            self.metrics.update(gt[i].detach().cpu(), pred[i].detach().cpu())
        log_dict.update(self.metrics.retrieve_avg('valid'))
        self.metrics.clean()

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        log_dict = {}

        gt = self.transform(y)

        eas_loss, berhu_loss, silog_loss, sasinv_loss, ssim_loss, gd_loss, tv_loss, mse_loss = combined_loss(pred, gt)
        loss = sum([eas_loss, berhu_loss, ssim_loss, gd_loss, tv_loss, silog_loss])

        log_dict['test_total_loss'] = loss.item()
        log_dict['test_ssim_loss'] = ssim_loss.item()
        log_dict['test_mse_loss'] = mse_loss.item()
        log_dict['test_eas_loss'] = eas_loss.item()
        log_dict['test_gd_loss'] = gd_loss.item()
        log_dict['test_tv_loss'] = tv_loss.item()
        log_dict['test_berhu_loss'] = berhu_loss.item()
        log_dict['test_silog_loss'] = silog_loss.item()
        log_dict['test_sasinv_loss'] = sasinv_loss.item()

        for i in range(pred.size()[0]):
            self.metrics.update(gt[i].detach().cpu(), pred[i].detach().cpu())
        log_dict.update(self.metrics.retrieve_avg('test'))
        self.metrics.clean()

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        if self.with_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.95)
            return ([optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}])
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, metric):
        if self.with_scheduler:
            scheduler.step(metric)