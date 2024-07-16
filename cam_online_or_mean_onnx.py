import time
from options import Options
import cv2
import numpy as np
import onnxruntime as ort

options = Options()
opts = options.parse()


def predict_depth(ort_session, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0).astype(np.float32)
    frame = frame.transpose((0, 3, 1, 2))/255.0

    depth_map = ort_session.run(None, {"input": frame})[0].squeeze()

    depth_map = (depth_map * 255.0)
    depth_map = depth_map.astype(np.uint8)

    return depth_map


def start_capture_mean_fps(ort_session, height, width):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    frames = []
    start_time = time.time()  # start time of the loop
    old_time = start_time
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (width, height))

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Predict depth map from frame
        depth_map = predict_depth(ort_session, frame)

        # actual FPS
        new_time = time.time()
        fps = 1.0 / (new_time - old_time + 1e-6)
        old_time =  new_time

        # Normalize depth map for visualization
        depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_PLASMA)
        # depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_RAINBOW)

        # Write the frame into the output video file
        frames.append(depth_map_colored)

        # Write number of FPS
        cv2.putText(depth_map_colored, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
    out = cv2.VideoWriter(f'output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # Write the frames to the video file
    for frame in frames:
        out.write(frame)

    out.release()


def start_capture_online_fps(ort_session, height, width, fps_verbose):
    fps = 0

    # initialize webcam
    cap = cv2.VideoCapture(0)

    # define the codec and VideoWriter
    out = cv2.VideoWriter(f'output.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, (width, height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        start_time = time.time()  # start time of the loop
        frame = cv2.resize(frame, (width, height))

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # predict depth map from frame
        depth_map = predict_depth(ort_session, frame)

        # colorize depth map for visualization
        depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_PLASMA)
        # depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_RAINBOW)

        # write number of FPS
        cv2.putText(depth_map_colored, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # write the frame into the output video file
        out.write(depth_map_colored)

        # display the resulting frame
        cv2.imshow('Depth Map', depth_map_colored)

        fps = np.round(1.0 / (time.time() - start_time + 1e-6), 2)
        if fps_verbose: print("FPS: ", fps)

        # press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # release everything once job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # setup runtime options
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # setup execution providers (ordered)
    exec_providers = [
        ('CUDAExecutionProvider', {"cudnn_conv_use_max_workspace": '1'}),
        'CPUExecutionProvider'
    ]

    # define onnx runtime inference session
    ort_session = ort.InferenceSession("model.onnx", so, providers=exec_providers)

    print("##### ONNX loaded")

    if opts.online:
        start_capture_online_fps(ort_session, opts.height, opts.width, opts.fps_verbose)
    else:
        start_capture_mean_fps(ort_session, opts.height, opts.width)