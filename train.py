from ultralytics import YOLO

model = YOLO("yolo11n.pt")

device = 'cuda'

results = model.train(
    data='/mnt/storage2/Enam/YOLOv11/NumberPlateDetection/truck_ocr_dataset_new/Bangla_License_Plate.yaml',
    epochs=50,            # Adjust based on your needs
    batch=16,             # Adjust based on available GPU memory
    device=device,
    patience=10,
    project='license_plate_detection',
    name='yolov11_training'
)