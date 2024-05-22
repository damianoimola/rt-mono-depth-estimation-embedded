from trainer import Trainer
from computer_vision_project.options import Options
import cv2
import numpy as np
import torch

options = Options()
opts = options.parse()

def predict_depth(trainer, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
    pred = trainer.predict(frame)
    pred = pred.detach()[0].cpu()
    pred = pred.permute(1, 2, 0).numpy()

    return pred


def start_capture(trainer, height, width):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          30, (width, height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (width, height))

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Predict depth map from frame
        # depth_map = predict_depth(trainer, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
        depth_map = trainer.predict(frame)
        depth_map = depth_map.detach()[0].cpu()
        depth_map = depth_map.permute(1, 2, 0).numpy()
        depth_map = (depth_map*255.0).astype(np.uint8)

        # Normalize depth map for visualization
        # depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_PLASMA)

        # Write the frame into the output video file
        out.write(depth_map_colored)

        # Display the resulting frame
        cv2.imshow('Depth Map', depth_map_colored)

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    trainer = Trainer(opts)
    model = "to_load"
    trainer.load(model)

    start_capture(trainer, opts.height, opts.width)





