from trainer import Trainer
from computer_vision_project.options import Options



options = Options()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
    trainer.save()