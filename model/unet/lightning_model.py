# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: D:\Python\univ_proj\computer_vision\computer_vision_project\model\unet\lightning_model.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2024-05-21 20:44:36 UTC (1716324276)

import lightning as L
import torch.nn as nn
import torch
from einops import rearrange
from torch import optim
from torchvision.transforms import transforms
from computer_vision_project.utilities.losses import combined_loss
from computer_vision_project.utilities.metrics import Metrics

class LitUNet(L.LightningModule):

    def __init__(self, plain_model, size, lr):
        super(LitUNet, self).__init__()
        self.model = plain_model
        self.transform = transforms.Resize(size=(size, size))
        self.lr = lr
        self.metrics = Metrics()

    def forward(self, inputs):
        preds = self.model(rearrange(inputs, 'B H W C -> B C H W'))
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x.permute(0, 3, 1, 2))
        log_dict = {}

        gt = y.permute((0, 3, 1, 2))
        gt = self.transform(gt)

        eas_loss, berhu_loss, silog_loss, sasinv_loss = combined_loss(pred, gt)
        mse_loss = nn.MSELoss()(pred, gt)
        loss = sum([eas_loss, mse_loss])

        log_dict['train_total_loss'] = loss.item()
        log_dict['train_mse_loss'] = mse_loss.item()
        log_dict['train_eas_loss'] = eas_loss.item()
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
        pred = self.model(x.permute(0, 3, 1, 2))
        log_dict = {}

        gt = y.permute((0, 3, 1, 2))
        gt = self.transform(gt)

        eas_loss, berhu_loss, silog_loss, sasinv_loss = combined_loss(pred, gt)
        mse_loss = nn.MSELoss()(pred, gt)
        loss = sum([eas_loss, mse_loss])

        log_dict['valid_total_loss'] = loss.item()
        log_dict['valid_mse_loss'] = mse_loss.item()
        log_dict['valid_eas_loss'] = eas_loss.item()
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
        pred = self.model(x.permute(0, 3, 1, 2))

        log_dict = {}
        gt = y.permute((0, 3, 1, 2))
        gt = self.transform(gt)

        eas_loss, berhu_loss, silog_loss, sasinv_loss = combined_loss(pred, gt)
        mse_loss = nn.MSELoss()(pred, gt)
        loss = sum([eas_loss, mse_loss])

        log_dict['test_total_loss'] = loss.item()
        log_dict['test_mse_loss'] = mse_loss.item()
        log_dict['test_eas_loss'] = eas_loss.item()
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        return ([optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}])

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)