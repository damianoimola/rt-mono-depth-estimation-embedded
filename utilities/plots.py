import cv2
import matplotlib.pyplot as plt
import torch
from einops import rearrange


def plot_predictions(input, target, all_preds, save=False, title='output', batch_size=4):
    import matplotlib
    matplotlib.use('TkAgg')

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