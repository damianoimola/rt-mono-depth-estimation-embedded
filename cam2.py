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
model = "mdrt12e_kaggle"

def show(img):
    plt.imshow(img, cmap="plasma")
    plt.show()

def predict_depth(trainer, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
    frame = frame.permute((0, 3, 1, 2))/255.0

    depth_map = trainer.predict(frame)

    depth_map = depth_map.detach()[0].cpu()
    depth_map = depth_map.permute(1, 2, 0).numpy()
    # show(depth_map)
    # depth_map = depth_map.astype(np.uint8)
    depth_map = (depth_map * 255.0)
    # show(depth_map)
    depth_map = depth_map.astype(np.uint8)
    # show(depth_map)
    # depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_L1).astype(np.uint8)

    return depth_map


def start_capture(trainer, height, width):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    frames = []
    start_time = time.time()  # start time of the loop
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (width, height))

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Predict depth map from frame
        depth_map = predict_depth(trainer, frame)

        # Normalize depth map for visualization
        depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_PLASMA)

        # Write the frame into the output video file
        frames.append(depth_map_colored)

        # Display the resulting frame
        cv2.imshow('Depth Map', depth_map_colored)

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    end_time = time.time()

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

    total_time = end_time - start_time
    fps = len(frames) / total_time

    print("MEAN FPS: ", fps)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(f'video/mean_{model}_output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # Write the frames to the video file
    for frame in frames:
        out.write(frame)

    out.release()

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.load(model)

    start_capture(trainer, opts.height, opts.width)




