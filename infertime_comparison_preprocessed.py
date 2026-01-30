import os
import time
import numpy as np
import cv2
from PIL import Image
import argparse

import torch
import onnxruntime as ort
from ultralytics import YOLO
from openvino.runtime import Core
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        tflite = None


# Paths
input_folder = "./val_resize"
yolo_path = "./license_plate_detection/yolov11_training/weights/best.pt"
torchscript_path = "./license_plate_detection/yolov11_training/weights/best.torchscript"
onnx_path = "./license_plate_detection/yolov11_training/weights/best.onnx"
openvino_path = "./license_plate_detection/yolov11_training/weights/best_openvino_model/best.xml"
tflite_f32_path = "./license_plate_detection/yolov11_training/weights/best_saved_model/best_float32.tflite"
tflite_f16_path = "./license_plate_detection/yolov11_training/weights/best_saved_model/best_float16.tflite"

results = {}

def get_image_files(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

# ---------- YOLO ----------
def benchmark_yolo(folder_path):
    model = YOLO(yolo_path)
    image_files = get_image_files(folder_path)
    print(f"[YOLO] Found {len(image_files)} images")
    start = time.time()
    for img_path in image_files:
        image = Image.open(img_path)
        _ = model(image)
    end = time.time()
    total = end - start
    avg = total / len(image_files)
    results['YOLO'] = (total, avg)

# ---------- TorchScript ----------
def benchmark_torchscript(folder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(torchscript_path).to(device)
    model.eval()
    input_shape = [1, 3, 640, 640]
    def preprocess(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (input_shape[3], input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0).to(device)
    image_files = get_image_files(folder_path)
    print(f"[TorchScript] Found {len(image_files)} images")
    warmup_img = preprocess(image_files[0])
    _ = model(warmup_img)
    start = time.time()
    with torch.no_grad():
        for img_path in image_files:
            img = preprocess(img_path)
            _ = model(img)
    end = time.time()
    total = end - start
    avg = total / len(image_files)
    results['TorchScript'] = (total, avg)

# ---------- ONNX ----------
def benchmark_onnx(folder_path):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    def preprocess(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (input_shape[3], input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    image_files = get_image_files(folder_path)
    print(f"[ONNX] Found {len(image_files)} images")
    start = time.time()
    for img_path in image_files:
        img = preprocess(img_path)
        _ = session.run(None, {input_name: img})
    end = time.time()
    total = end - start
    avg = total / len(image_files)
    results['ONNX'] = (total, avg)

# ---------- OpenVINO ----------
def benchmark_openvino(folder_path):
    core = Core()
    model = core.read_model(openvino_path)
    compiled_model = core.compile_model(model, "CPU")
    input_layer = compiled_model.input(0)
    input_shape = input_layer.shape
    def preprocess(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (input_shape[3], input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    image_files = get_image_files(folder_path)
    print(f"[OpenVINO] Found {len(image_files)} images")
    start = time.time()
    for img_path in image_files:
        img = preprocess(img_path)
        _ = compiled_model([img])
    end = time.time()
    total = end - start
    avg = total / len(image_files)
    results['OpenVINO'] = (total, avg)

# ---------- TFLite ----------
def benchmark_tflite(folder_path, model_path, label):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    def preprocess(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    image_files = get_image_files(folder_path)
    print(f"[{label}] Found {len(image_files)} images")
    warmup_img = preprocess(image_files[0])
    interpreter.set_tensor(input_details[0]['index'], warmup_img)
    interpreter.invoke()
    start = time.time()
    for img_path in image_files:
        img = preprocess(img_path)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    total = end - start
    avg = total / len(image_files)
    results[label] = (total, avg)

# ---------- MAIN ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model inference times.")
    parser.add_argument("--model", type=str, default="ALL",
                        help="Which model to run: YOLO, TorchScript, ONNX, OpenVINO, TFLite Float32, TFLite Float16, or ALL")
    args = parser.parse_args()

    selected = args.model.strip().upper()

    if selected == "YOLO" or selected == "ALL":
        benchmark_yolo(input_folder)
    if selected == "TORCHSCRIPT" or selected == "ALL":
        benchmark_torchscript(input_folder)
    if selected == "ONNX" or selected == "ALL":
        benchmark_onnx(input_folder)
    if selected == "OPENVINO" or selected == "ALL":
        benchmark_openvino(input_folder)
    if selected == "TFLITE FLOAT32" or selected == "ALL":
        benchmark_tflite(input_folder, tflite_f32_path, "TFLite Float32")
    if selected == "TFLITE FLOAT16" or selected == "ALL":
        benchmark_tflite(input_folder, tflite_f16_path, "TFLite Float16")

    print("\n========= Inference Summary =========")
    for key, (total, avg) in results.items():
        print(f"{key}: Total = {total:.2f}s, Avg per image = {avg:.4f}s")
