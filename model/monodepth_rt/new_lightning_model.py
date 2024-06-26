import lightning as L
import torch.nn as nn
import torch
from einops import rearrange
from torch import optim
from torchvision.transforms import transforms
from utilities.losses import combined_loss
from utilities.metrics import Metrics
import torch.nn.functional as F


class NEWLitMonoDepthRT(L.LightningModule):
    def __init__(self, plain_model, size, lr):
        super(NEWLitMonoDepthRT, self).__init__()
        self.model = plain_model
        self.resize = transforms.Resize(size=(size, size), antialias=False)
        self.lr = lr
        self.metrics = Metrics()
        self.with_scheduler = True

    def forward(self, inputs):
        return self.model(inputs)

    def compute_loss(self, ground, pred, split):
        eas_loss, berhu_loss, silog_loss, sasinv_loss, ssim_loss, gd_loss, tv_loss, mse_loss = combined_loss(pred, ground)
        loss = sum([eas_loss, berhu_loss, ssim_loss, gd_loss, tv_loss, silog_loss])

        log_dict = None
        if split is not None:
            log_dict = {
                f'{split}_total_loss': loss.item(),
                f'{split}_ssim_loss': ssim_loss.item(),
                f'{split}_mse_loss': mse_loss.item(),
                f'{split}_eas_loss': eas_loss.item(),
                f'{split}_gd_loss': gd_loss.item(),
                f'{split}_tv_loss': tv_loss.item(),
                f'{split}_berhu_loss': berhu_loss.item(),
                f'{split}_silog_loss': silog_loss.item(),
                f'{split}_sasinv_loss': sasinv_loss.item()
            }

        return loss, log_dict

    def compute_loss_and_metrics(self, ground, pred, split):

        loss, log_dict = self.compute_loss(ground, pred, split)

        for i in range(pred.size()[0]):
            self.metrics.update(ground[i].detach().cpu(), pred[i].detach().cpu())
        log_dict.update(self.metrics.retrieve_avg(split))
        self.metrics.clean()

        return loss, log_dict


    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        p1, p2, p3, p4 = preds[0], preds[1], preds[2], preds[3]

        gt = self.resize(y)

        loss, log_dict = self.compute_loss_and_metrics(p1, gt, 'train')
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for i, pred in enumerate([p2, p3, p4]):
            ground = F.interpolate(gt, scale_factor=1 / (2 ** (i + 1)), mode='bicubic', align_corners=False, antialias=False)
            loss = loss + self.compute_loss(ground, pred, None)[0]

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        p1, p2, p3, p4 = preds[0], preds[1], preds[2], preds[3]
        log_dict = {}

        gt = self.resize(y)

        loss, log_dict = self.compute_loss_and_metrics(p1, gt, 'valid')
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for i, pred in enumerate([p2, p3, p4]):
            ground = F.interpolate(gt, scale_factor=1 / (2 ** (i + 1)), mode='bicubic', align_corners=False, antialias=False)
            loss = loss + self.compute_loss(ground, pred, None)[0]

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        p1, p2, p3, p4 = preds[0], preds[1], preds[2], preds[3]
        log_dict = {}

        gt = self.resize(y)

        loss, log_dict = self.compute_loss_and_metrics(p1, gt, 'test')
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for i, pred in enumerate([p2, p3, p4]):
            ground = F.interpolate(gt, scale_factor=1 / (2 ** (i + 1)), mode='bicubic', align_corners=False, antialias=False)
            loss = loss + self.compute_loss(ground, pred, None)[0]

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