from trainer import Trainer
from options import Options



options = Options()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.load(opts.ckpt)
    trainer.save_as_onnx(opts.height, opts.width)