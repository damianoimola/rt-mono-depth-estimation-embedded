import matplotlib.pyplot as plt

def plot_predictions(input, target, save=False, title="test", num_preds=4):
    fig, axes = plt.subplots(config.batch_size, num_preds+2, figsize=(13, 13))
    for i in range(config.batch_size):
        for j in range(num_preds+2):
            ax = axes[i, j]
            if(j == 4):
                pred = target[i]
                if(i == 0): ax.set_title("ground truth")
            elif(j >= 5):
                pred = input[i]
                if(i == 0): ax.set_title("reference")
            else:
                pred = all_preds[j]
                pred = pred.detach()[i]
                pred = rearrange(pred, "C H W -> H W C")
                if(i == 0):
                    ax.set_title(f"shape {pred.numpy().shape}")
            ax.axis("off")
            ax.imshow(pred, cmap="plasma")
    if save: plt.savefig(f'{title}.png')