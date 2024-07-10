import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib
from einops import rearrange

matplotlib.use('TkAgg')


def plot_predictions(input, target, all_preds, save=False, title='output', batch_size=4):

    print("PRE", all_preds.size())

    if len(all_preds.size()) < 3:
        raise("There is a problem")
    if len(all_preds.size()) == 3:
        all_preds = all_preds.unsqueeze(0)

    print("POST", all_preds.size())

    # number of predictions for each sample in batch
    num_preds = 1

    fig, axes = plt.subplots(batch_size, num_preds + 2, figsize=(13, 13))
    for i in range(batch_size):
        for j in range(num_preds + 2):
            ax = axes[i, j]
            if j == num_preds:
                t = target[i].permute((1, 2, 0)).numpy()
                pred = cv2.resize(t, (256, 256))
                if i == 0: ax.set_title('ground truth')

            elif j >= num_preds + 1:
                inp = input[i].permute((1, 2, 0)).numpy()
                pred = cv2.resize(inp, (256, 256))
                if i == 0: ax.set_title('reference')

            else:
                # pred = all_preds[j]
                # pred = pred.detach()[i].cpu()
                pred = all_preds[i]
                pred = pred.detach().cpu()
                pred = pred.permute((1, 2, 0)).numpy()
                if i == 0:
                    print("NP shape", pred.shape)
                    ax.set_title(f'shape {pred.shape}')

            ax.axis('off')
            print("PREPRINT", pred.shape)
            ax.imshow(pred, cmap='plasma')

    if save: plt.savefig(f'images/{title}.png')
    plt.tight_layout()
    plt.show()


def plot_metrics(path_to_metrics):
    results = pd.read_csv(path_to_metrics)

    import matplotlib
    matplotlib.use('TkAgg')
    plt.style.use('default')
    plt.rcParams['text.usetex'] = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 7))

    ax = axes[0][0]
    ax.set_title('Losses per epoch')
    ax.plot(results['train_total_loss_epoch'].dropna(ignore_index=True), label='train total loss', color='dodgerblue')
    ax.plot(results['train_mse_loss_epoch'].dropna(ignore_index=True), label='train MSE loss', color='orangered')
    ax.plot(results['train_ssim_loss_epoch'].dropna(ignore_index=True), label='train MSE loss', color='mediumseagreen')
    ax.plot(results['train_berhu_loss_epoch'].dropna(ignore_index=True), label='train BerHu loss', color='goldenrod')
    ax.plot(results['train_eas_loss_epoch'].dropna(ignore_index=True), label='train EAS loss', color='darkgreen')
    ax.plot(results['train_silog_loss_epoch'].dropna(ignore_index=True), label='train SiLog loss', color='palegreen')
    ax.plot(results['train_sasinv_loss_epoch'].dropna(ignore_index=True), label='train SaSInv loss', color='coral')

    ax.plot(results['valid_total_loss'].dropna(ignore_index=True), label='validation total loss', color='dodgerblue', alpha=0.5)
    # ax.plot(results['valid_mse_loss'].dropna(ignore_index=True), label='validation MSE loss', color='orangered', alpha=0.5)
    ax.plot(results['valid_ssim_loss'].dropna(ignore_index=True), label='validation MSE loss', color='mediumseagreen', alpha=0.5)
    ax.plot(results['valid_berhu_loss'].dropna(ignore_index=True), label='validation BerHu loss', color='goldenrod', alpha=0.5)
    ax.plot(results['valid_eas_loss'].dropna(ignore_index=True), label='validation EAS loss', color='darkgreen', alpha=0.5)
    ax.plot(results['valid_silog_loss'].dropna(ignore_index=True), label='validation SiLog loss', color='palegreen', alpha=0.5)
    # ax.plot(results['valid_sasinv_loss'].dropna(ignore_index=True), label='validation SaSInv loss', color='coral', alpha=0.5)
    ax.legend(loc='upper center', prop = { "size": 7.5 }, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    # ax.set_ylim(0, 2)
    ax.grid()



    ax = axes[1][1]
    ax.set_title('Deltas per epoch')
    ax.plot(results['train_delta1_epoch'].dropna(ignore_index=True), color='royalblue', label='δ_1 train')
    ax.plot(results['train_delta2_epoch'].dropna(ignore_index=True), color='cornflowerblue', label='δ_2 train')
    ax.plot(results['train_delta3_epoch'].dropna(ignore_index=True), color='lightsteelblue', label='δ_3 train')

    ax.plot(results['valid_delta1'].dropna(ignore_index=True), color='orangered', label='δ_1 validation')
    ax.plot(results['valid_delta2'].dropna(ignore_index=True), color='coral', label='δ_2 validation')
    ax.plot(results['valid_delta3'].dropna(ignore_index=True), color='darksalmon', label='δ_3 validation')
    ax.legend(loc='upper center', prop = { "size": 7.5 }, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    ax.grid()



    ax = axes[1][0]
    ax.set_title('Absolute Relative Error per epoch')
    ax.plot(results['train_abs_rel_epoch'].dropna(ignore_index=True), color='crimson', label='(AbsRel) absolute relative error train')
    ax.plot(results['valid_abs_rel'].dropna(ignore_index=True), color='crimson', label='(AbsRel) absolute relative error validation', alpha=0.7)
    ax.legend(loc='upper center', prop = { "size": 7.5 }, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1)
    ax.grid()


    ax = axes[0][1]
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
    ax.legend(loc='upper center', prop = { "size": 7.5 }, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    ax.grid()

    fig.tight_layout(pad=1.0)

    plt.show()


def point_cloud_viz(input, prediction):
    depth_vis = rearrange(np.flipud(prediction), "B H W -> H W B").squeeze()
    img_vis = rearrange(np.flipud(input), "B H W -> H W B")

    print(depth_vis.shape, img_vis.shape)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection="3d")

    STEP = 5
    for x in range(0, img_vis.shape[0], STEP):
        for y in range(0, img_vis.shape[1], STEP):
            ax.scatter(
                [depth_vis[x, y]] * 3,
                [y] * 3,
                [x] * 3,
                c=tuple(img_vis[x, y, :3] / 255),
                # c=depth_vis[x, y],
                cmap="plasma",
                s=3,
            )
        ax.view_init(45, 135)
    plt.show()