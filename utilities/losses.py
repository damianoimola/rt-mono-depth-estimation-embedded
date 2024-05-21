# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: D:\Python\univ_proj\computer_vision\computer_vision_project\utilities\losses.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2024-05-20 20:13:10 UTC (1716235990)

import torch
import torch.nn as nn
import torch.nn.functional as F

class BerHuLoss(nn.Module):

    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, predicted_depth, ground_truth_depth):
        diff = torch.abs(predicted_depth - ground_truth_depth)
        # computing c (0.2 = 1/5)
        c = 0.2 * torch.max(diff).item()
        # getting l1 and l2 masks
        l2_mask = (diff > c).float()
        l1_mask = 1.0 - l2_mask
        # computing l1 and l2 to relative pixels
        l1_loss = l1_mask * diff
        l2_loss = l2_mask * (diff ** 2 + c ** 2) / (2 * c)
        return torch.mean(l1_loss + l2_loss)

class EdgeAwareSmoothnessLossWithSobel(nn.Module):

    def __init__(self):
        super(EdgeAwareSmoothnessLossWithSobel, self).__init__()

    def forward(self, pred, image):
        img_grad_x, img_grad_y = self.compute_image_gradients(image)
        pred_grad_x, pred_grad_y = self.compute_image_gradients(pred)
        weights_x = torch.exp(-torch.abs(img_grad_x))
        weights_y = torch.exp(-torch.abs(img_grad_y))
        smoothness_x = weights_x * torch.abs(pred_grad_x)
        smoothness_y = weights_y * torch.abs(pred_grad_y)
        return smoothness_x.mean() + smoothness_y.mean()

    def compute_image_gradients(self, image):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        return (grad_x, grad_y)

class EdgeAwareSmoothnessLoss(nn.Module):

    def __init__(self):
        super(EdgeAwareSmoothnessLoss, self).__init__()

    def forward(self, predicted_depth, ground_truth_depth):
        dy_pred, dx_pred = (self.gradient_y(predicted_depth), self.gradient_x(predicted_depth))
        dy_image, dx_image = (self.gradient_y(ground_truth_depth), self.gradient_x(ground_truth_depth))
        weight_x = torch.exp(-torch.mean(dx_image, 1, keepdim=True))
        weight_y = torch.exp(-torch.mean(dy_image, 1, keepdim=True))
        smoothness_x = dx_pred * weight_x
        smoothness_y = dy_pred * weight_y
        return smoothness_x.mean() + smoothness_y.mean()

    def gradient_x(self, img):
        gx = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        return gx

    def gradient_y(self, img):
        gy = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        return gy

class SiLogLoss(nn.Module):
    def __init__(self):
        super(SiLogLoss, self).__init__()

    def forward(self, predicted_depth, ground_truth_depth, variance_focus=0.5, eps=1e-06):
        predicted_depth = predicted_depth.clamp(min=eps)
        ground_truth_depth = ground_truth_depth.clamp(min=eps)
        diff = torch.log(predicted_depth) - torch.log(ground_truth_depth)
        loss = (diff ** 2).mean() - variance_focus * diff.mean() ** 2
        return torch.sqrt(loss + eps)

class SASInvLoss(nn.Module):

    def __init__(self):
        super(SASInvLoss, self).__init__()

    def forward(self, predicted_depth, ground_truth_depth, variance_focus=0.5):
        alpha = torch.mean(predicted_depth * ground_truth_depth) / (torch.mean(predicted_depth ** 2) + 1e-06)
        beta = torch.mean(ground_truth_depth) - alpha * torch.mean(predicted_depth)
        adjusted_pred = alpha * predicted_depth + beta
        return torch.mean((adjusted_pred - ground_truth_depth) ** 2)

def combined_loss(predicted_depth, ground_truth_depth):
    eas_loss = EdgeAwareSmoothnessLoss()(predicted_depth, ground_truth_depth)
    berhu_loss = BerHuLoss()(predicted_depth, ground_truth_depth)
    silog_loss = SiLogLoss()(predicted_depth, ground_truth_depth)
    sasinv_loss = SASInvLoss()(predicted_depth, ground_truth_depth)
    return (eas_loss, berhu_loss, silog_loss, sasinv_loss)