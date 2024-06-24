from trainer import Trainer
from options import Options



options = Options()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)

    # trainer.plot_metrics("training_logs/unet-d=nyu_v2-lr=0.001-e=100/version_0/metrics.csv")

    trainer.load("28e_kaggle")
    trainer.plot_batch_predictions()