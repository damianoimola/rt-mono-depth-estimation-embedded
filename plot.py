import torch

from trainer import Trainer
from options import Options
from utilities.plotting import point_cloud_viz



options = Options()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)

    # trainer.plot_metrics("training_logs/unet-d=nyu_v2-lr=0.001-e=100/version_0/metrics.csv")

    trainer.load("mde45e_kaggle")
    # trainer.display_batch_predictions()

    tdl, _, _ = trainer.get_data()
    batch_inputs, batch_target = next(iter(tdl))
    batch_preds = trainer.predict(batch_inputs)

    point_cloud_viz(batch_inputs[0].detach().numpy(), batch_preds[0].detach().numpy())
    # point_cloud_viz(batch_inputs[0].detach().numpy(), batch_target[0].detach().numpy())


