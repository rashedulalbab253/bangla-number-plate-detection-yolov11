from ultralytics import YOLO

model_path = "./license_plate_detection/yolov11_training/weights/best.pt"
model = YOLO(model_path)

model.export(format="openvino")  # creates 'yolov8n_openvino_model/'
ov_model = YOLO("./license_plate_detection/yolov11_training/weights/best_openvino_model")

model.export(format="onnx")
onnx_model = YOLO("./license_plate_detection/yolov11_training/weights/best.onnx")

model.export(format="tflite")
tflite_model = YOLO("./license_plate_detection/yolov11_training/weights/best_saved_model/best_float32.tflite")

model.export(format="torchscript")
torchscript_model = YOLO("./license_plate_detection/yolov11_training/weights/best.torchscript")