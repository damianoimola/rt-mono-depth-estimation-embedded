import time
from options import Options
import cv2
import numpy as np
import onnx
import onnxruntime as ort

options = Options()
opts = options.parse()


def predict_depth(ort_session, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0).astype(np.float32)
    frame = frame.transpose((0, 3, 1, 2))/255.0

    # depth_map = trainer.onnx_predict(frame)[0].squeeze()
    depth_map = ort_session.run(None, {"input": frame})[0].squeeze()

    depth_map = (depth_map * 255.0)
    depth_map = depth_map.astype(np.uint8)

    return depth_map


def start_capture_mean_fps(ort_session, height, width):
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
        depth_map = predict_depth(ort_session, frame)

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
    out = cv2.VideoWriter(f'video/rpi_output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # Write the frames to the video file
    for frame in frames:
        out.write(frame)

    out.release()



def start_capture_online_fps(ort_session, height, width):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(f'video/rpi_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, (width, height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        start_time = time.time()  # start time of the loop
        frame = cv2.resize(frame, (width, height))

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Predict depth map from frame
        depth_map = predict_depth(ort_session, frame)

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
    onnx_model = onnx.load("model.onnx")

    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")

    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    exproviders = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    ort_session = ort.InferenceSession("model.onnx", so, providers=exproviders)

    options = ort_session.get_provider_options()
    cuda_options = options['CUDAExecutionProvider']
    cuda_options['cudnn_conv_use_max_workspace'] = '1'
    ort_session.set_providers(['CUDAExecutionProvider'], [cuda_options])
    print("ONNX loaded")

    start_capture_mean_fps(ort_session, opts.height, opts.width)




