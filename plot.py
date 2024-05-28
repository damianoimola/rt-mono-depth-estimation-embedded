from trainer import Trainer
from computer_vision_project.options import Options



options = Options()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)

    trainer.plot_metrics("training_logs/unet-d=nyu_v2-lr=0.001-e=200/version_0/metrics.csv")

    trainer.load("unet-d=nyu_v2-lr=0.001-e=200")
    trainer.plot_batch_predictions()