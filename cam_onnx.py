import time
import matplotlib
from matplotlib import pyplot as plt
from trainer import Trainer
from options import Options
import cv2
import numpy as np
import torch

options = Options()
opts = options.parse()

matplotlib.use('TkAgg')

# model = "28e_kaggle"
# model = "mdrt20e_kaggle"
model = "mde30e_kaggle"

def show(img):
    plt.imshow(img, cmap="plasma")
    plt.show()

def predict_depth(trainer, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0).astype(np.float32)
    frame = frame.transpose((0, 3, 1, 2))/255.0

    depth_map = trainer.onnx_predict(frame)[0].squeeze()

    depth_map = (depth_map * 255.0)
    depth_map = depth_map.astype(np.uint8)

    return depth_map


def start_capture(trainer, height, width):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(f'video/onnx_{model}_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, (width, height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        start_time = time.time()  # start time of the loop
        frame = cv2.resize(frame, (width, height))

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Predict depth map from frame
        depth_map = predict_depth(trainer, frame)

        # Normalize depth map for visualization
        depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_PLASMA)

        # Write the frame into the output video file
        out.write(depth_map_colored)

        # Display the resulting frame
        cv2.imshow('Depth Map', depth_map_colored)

        print("FPS: ", 1.0 / (time.time() - start_time + 1e-6))

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.load(model)

    trainer.save_as_onnx()
    trainer.load_from_onnx()

    start_capture(trainer, opts.height, opts.width)





