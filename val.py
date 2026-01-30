from ultralytics import YOLO

# Specify the path to your trained model 
model_path = "./license_plate_detection/yolov11_training/weights/best.pt"  # Update this path to where your best.pt is saved
model = YOLO(model_path)

val_results = model.val(data='./truck_ocr_dataset_new/Bangla_License_Plate.yaml')