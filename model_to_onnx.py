from trainer import Trainer
from options import Options



options = Options()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)

    print("##### Loading", opts.ckpt)
    trainer.load(opts.ckpt)
    print("##### Checkpoint loaded")

    print("##### Saving model in ONNX format: 'model.onnx'")
    trainer.save_as_onnx(opts.height, opts.width)
    print("##### Model saved")