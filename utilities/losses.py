import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, predicted_depth, ground_truth_depth, window_size=11, size_average=True):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(predicted_depth, window_size, 1, 0)
        mu_target = F.avg_pool2d(ground_truth_depth, window_size, 1, 0)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.avg_pool2d(predicted_depth * predicted_depth, window_size, 1, 0) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(ground_truth_depth * ground_truth_depth, window_size, 1, 0) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(predicted_depth * ground_truth_depth, window_size, 1, 0) - mu_pred_target

        # [-1, 1]
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / (
                    (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        # [0, 2]
        complementary_ssim_map = 1 - ssim_map

        # [0, 1]
        if size_average:
            return torch.clamp(complementary_ssim_map/2, 0, 1).mean()
        else:
            return torch.clamp(complementary_ssim_map/2, 0, 1).mean([1, 2, 3])


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
    ssim_loss = SSIMLoss()(predicted_depth, ground_truth_depth)
    mse_loss = nn.MSELoss()(predicted_depth, ground_truth_depth)
    return (eas_loss, berhu_loss, silog_loss, sasinv_loss, ssim_loss, mse_loss)