# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: D:\Python\univ_proj\computer_vision\computer_vision_project\utilities\plots.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2024-05-20 13:40:36 UTC (1716212436)

import matplotlib.pyplot as plt
import torch
from einops import rearrange

def plot_predictions(input, target, all_preds, save=False, title='output', batch_size=6):
    import matplotlib
    matplotlib.use('TkAgg')
    num_preds = len(all_preds)
    if len(all_preds.size()) < 4:
        all_preds = all_preds.unsqueeze(0)
    fig, axes = plt.subplots(batch_size, num_preds + 2, figsize=(13, 13))
    for i in range(batch_size):
        for j in range(num_preds + 2):
            ax = axes[i, j]
            if j == num_preds:
                pred = target[i]
                if i == 0:
                    ax.set_title('ground truth')
            elif j >= num_preds + 1:
                pred = input[i]
                if i == 0:
                    ax.set_title('reference')
            else:
                pred = all_preds[j]
                pred = pred.detach()[i].cpu()
                pred = pred.permute((1, 2, 0))
                if i == 0:
                    ax.set_title(f'shape {pred.numpy().shape}')
            ax.axis('off')
            ax.imshow(pred, cmap='plasma')
    if save:
        plt.savefig(f'images/{title}.png')
    plt.tight_layout()
    plt.show()