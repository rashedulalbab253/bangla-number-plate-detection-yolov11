import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import time
import cv2
from ultralytics import YOLO

model = YOLO("./license_plate_detection/yolov11_training/weights/best.pt")
onnx_model = YOLO("./license_plate_detection/yolov11_training/weights/best.onnx")
ov_model = YOLO("./license_plate_detection/yolov11_training/weights/best_openvino_model")
torchscript_model = YOLO("./license_plate_detection/yolov11_training/weights/best.torchscript")
tflite_model = YOLO("./license_plate_detection/yolov11_training/weights/best_saved_model/best_float32.tflite")

def benchmark_inference(folder_path):
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    print(f"Found {len(image_files)} images")

    start = time.time()
    for image_path in image_files:
        result = model(image_path)
    end = time.time()

    total_time = end - start
    avg_time = total_time / len(image_files) if image_files else 0

    start = time.time()
    for image_path in image_files:
        onnx_result = onnx_model(image_path)
    end = time.time()

    total_time_onnx = end - start
    avg_time_onnx = total_time_onnx / len(image_files) if image_files else 0

    start = time.time()
    for image_path in image_files:
        ov_result = ov_model(image_path)
    end = time.time()

    total_time_ov = end - start
    avg_time_ov = total_time_ov / len(image_files) if image_files else 0

    start = time.time()
    for image_path in image_files:
        result = torchscript_model(image_path)
    end = time.time()

    total_time_torchscript = end - start
    avg_time_torchscript = total_time_torchscript / len(image_files) if image_files else 0

    start = time.time()
    for image_path in image_files:
        ov_result = tflite_model(image_path)
    end = time.time()

    total_time_tflite = end - start
    avg_time_tflite = total_time_tflite / len(image_files) if image_files else 0

    print(f"\nRuntime Inference Time:")
    print(f"Total: {total_time:.2f} seconds")
    print(f"Average per image: {avg_time:.4f} seconds")

    print(f"\nONNX Runtime Inference Time:")
    print(f"Total: {total_time_onnx:.2f} seconds")
    print(f"Average per image: {avg_time_onnx:.4f} seconds")

    print(f"\nOPENVINO Runtime Inference Time:")
    print(f"Total: {total_time_ov:.2f} seconds")
    print(f"Average per image: {avg_time_ov:.4f} seconds")

    print(f"\nTORCHSCRIPT Runtime Inference Time:")
    print(f"Total: {total_time_torchscript:.2f} seconds")
    print(f"Average per image: {avg_time_torchscript:.4f} seconds")

    print(f"\nTFLITE Runtime Inference Time:")
    print(f"Total: {total_time_tflite:.2f} seconds")
    print(f"Average per image: {avg_time_tflite:.4f} seconds")

# Run benchmark
benchmark_inference("./val_resize")
