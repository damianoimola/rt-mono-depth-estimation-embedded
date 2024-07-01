import time
import torch
from options import Options
from trainer import Trainer

options = Options()
opts = options.parse()
model = "mde25e_kaggle"


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.load(model)

    trainer.save_as_onnx()
    trainer.load_from_onnx()

    frame = torch.randn(1, 3, 256, 256, requires_grad=True)

    start = time.time()
    torch_out = trainer.predict(frame)
    end = time.time()
    print(f"Inference of Pytorch model used {end - start} seconds")

    frame = frame.detach().cpu().numpy()
    start = time.time()
    ort_outs = trainer.onnx_predict(frame)
    end = time.time()
    print(f"Inference of ONNX model used {end - start} seconds")