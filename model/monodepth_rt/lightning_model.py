import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
import lightning as L
from einops import rearrange
from computer_vision_project.utilities.metrics import Metrics

class LitMonoDERT(L.LightningModule):

    def __init__(self, model, size, lr):
        super().__init__()
        self.model = model
        self.size = size
        self.lr = lr
        self.metrics = Metrics()

    def forward(self, inputs):
        preds = self.model(rearrange(inputs, 'B H W C -> B C H W'))
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(rearrange(x, 'B H W C -> B C H W'))
        log_dict = {}
        pred_losses = []
        for i, p in enumerate(reversed(preds)):
            s = self.size // 2 ** i
            transform = transforms.Resize(size=(s, s))
            gt = transform(rearrange(y, 'B H W C -> B C H W'))
            loss = torch.sqrt(nn.MSELoss()(p, gt))
            pred_losses.append(loss)
            log_dict['train_' + str(s) + 'x' + str(s) + '_loss'] = loss
            if i == 0:
                for i in range(p.size()[0]):
                    self.metrics.update(gt[i].detach().cpu(), p[i].detach().cpu())
                log_dict.update(self.metrics.retrieve_avg())
        total_loss = sum(pred_losses)
        log_dict['train_total_loss'] = total_loss
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(rearrange(x, 'B H W C -> B C H W'))
        log_dict = {}
        pred_losses = []
        for i, p in enumerate(reversed(preds)):
            s = self.size // 2 ** i
            transform = transforms.Resize(size=(s, s))
            gt = transform(rearrange(y, 'B H W C -> B C H W'))
            loss = torch.sqrt(nn.MSELoss()(p, gt))
            pred_losses.append(loss)
            log_dict['valid_' + str(s) + 'x' + str(s) + '_loss'] = loss
            if i == 0:
                for i in range(p.size()[0]):
                    avg_metrics = self.metrics.update(gt[i].detach().cpu(), p[i].detach().cpu())
                log_dict.update(self.metrics.retrieve_avg())
        total_loss = sum(pred_losses)
        log_dict['valid_total_loss'] = total_loss
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(rearrange(x, 'B H W C -> B C H W'))
        log_dict = {}
        pred_losses = []
        for i, p in enumerate(reversed(preds)):
            s = self.size // 2 ** i
            transform = transforms.Resize(size=(s, s))
            gt = transform(rearrange(y, 'B H W C -> B C H W'))
            loss = torch.sqrt(nn.MSELoss()(p, gt))
            pred_losses.append(loss)
            log_dict['test_' + str(s) + 'x' + str(s) + '_loss'] = loss.item()
            if i == 0:
                for i in range(p.size()[0]):
                    avg_metrics = self.metrics.update(gt[i].detach().cpu(), p[i].detach().cpu())
                log_dict.update(self.metrics.retrieve_avg())
        total_loss = sum(pred_losses)
        log_dict['test_total_loss'] = total_loss
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer