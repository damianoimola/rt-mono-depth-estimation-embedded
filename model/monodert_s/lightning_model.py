import lightning as L
import torch.nn as nn
import torch
from einops import rearrange
from torch import optim
from torchvision.transforms import transforms
from utilities.losses import combined_loss, LossManager
from utilities.metrics import Metrics
import torch.nn.functional as F

class LitMonoDeRTs(L.LightningModule):
    def __init__(self, plain_model, size, lr):
        super(LitMonoDeRTs, self).__init__()
        self.model = plain_model
        self.resize = transforms.Resize(size=(size, size), antialias=False)
        self.lr = lr
        self.metrics = Metrics()
        self.with_scheduler = True
        self.loss_manager = LossManager(['berhu', 'ssim', 'silog'])

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        p1, p2, p3 = preds[0], preds[1], preds[2]

        gt = self.resize(y)

        # computing loss and its dict to display
        loss, log_dict = self.loss_manager.compute_loss(p1, gt, 'train')

        # computing metrics to take into account
        for i in range(p1.size()[0]):
            self.metrics.update(gt[i].detach().cpu(), p1[i].detach().cpu())
        log_dict.update(self.metrics.retrieve_avg('train'))
        self.metrics.clean()

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # compute loss per sub-image
        for i, pred in enumerate([p2, p3]):
            ground = F.interpolate(gt, scale_factor=1 / (2 ** (i + 1)), mode='bicubic', align_corners=False,
                                   antialias=False)
            sub_loss, _ = self.loss_manager.compute_loss(pred, ground, 'train')
            loss = loss + sub_loss

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        p1, p2, p3 = preds[0], preds[1], preds[2]

        gt = self.resize(y)

        # computing loss and its dict to display
        loss, log_dict = self.loss_manager.compute_loss(p1, gt, 'valid')

        # computing metrics to take into account
        for i in range(p1.size()[0]):
            self.metrics.update(gt[i].detach().cpu(), p1[i].detach().cpu())
        log_dict.update(self.metrics.retrieve_avg('train'))
        self.metrics.clean()

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # compute loss per sub-image
        for i, pred in enumerate([p2, p3]):
            ground = F.interpolate(gt, scale_factor=1 / (2 ** (i + 1)), mode='bicubic', align_corners=False,
                                   antialias=False)
            sub_loss, _ = self.loss_manager.compute_loss(pred, ground, 'valid')
            loss = loss + sub_loss

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        p1, p2, p3 = preds[0], preds[1], preds[2]

        gt = self.resize(y)

        # computing loss and its dict to display
        loss, log_dict = self.loss_manager.compute_loss(p1, gt, 'test')

        # computing metrics to take into account
        for i in range(p1.size()[0]):
            self.metrics.update(gt[i].detach().cpu(), p1[i].detach().cpu())
        log_dict.update(self.metrics.retrieve_avg('train'))
        self.metrics.clean()

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # compute loss per sub-image
        for i, pred in enumerate([p2, p3]):
            ground = F.interpolate(gt, scale_factor=1 / (2 ** (i + 1)), mode='bicubic', align_corners=False,
                                   antialias=False)
            sub_loss, _ = self.loss_manager.compute_loss(pred, ground, 'test')
            loss = loss + sub_loss

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