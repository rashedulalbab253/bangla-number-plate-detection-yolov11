import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid
from PIL import Image
import time

# Load the YOLO model
MODEL_PATH = "./license_plate_detection/yolov11_training/weights/best.pt"
model = YOLO(MODEL_PATH)

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

def detect_license_plates(input_image):

    # Generate a unique ID for this detection
    image_id = str(uuid.uuid4())[:8]
    
    # Convert Gradio image to numpy array if it's not already
    if isinstance(input_image, np.ndarray):
        img_array = input_image
    else:
        img_array = np.array(input_image)
    
    # Save temp input image
    input_path = f"uploads/{image_id}_input.jpg"
    cv2.imwrite(input_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    # Start timing
    start_time = time.time()
    
    # Run inference
    results = model(input_path)
    result = results[0]
    
    # Get the annotated image with bounding boxes
    annotated_img = result.plot()
    
    # Convert back to RGB for display in Gradio
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Get detection details for display
    detection_info = ""
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = float(box.conf[0].item())
        coordinates = box.xyxy[0].tolist()  # x1, y1, x2, y2 format
        
        detection_info += f"Detection {i+1}:\n"
        detection_info += f"  - Class: {class_name}\n"
        detection_info += f"  - Confidence: {confidence:.2f}\n"
        detection_info += f"  - Coordinates: [{coordinates[0]:.1f}, {coordinates[1]:.1f}, {coordinates[2]:.1f}, {coordinates[3]:.1f}]\n\n"

    processing_time = time.time() - start_time

    if os.path.exists(input_path):
        os.remove(input_path)
    
    if not detection_info:
        detection_info = "No license plates detected."
    
    detection_info += f"\nProcessing time: {processing_time:.3f} seconds"
    
    return annotated_img_rgb, detection_info

# Create Gradio interface
with gr.Blocks(title="License Plate Detection") as demo:
    gr.Markdown("# License Plate Detection")
    gr.Markdown("Upload an image to detect license plates")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            detect_button = gr.Button("Detect License Plates")
        
        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            detection_text = gr.Textbox(label="Detection Details", lines=10)
    
    detect_button.click(
        fn=detect_license_plates,
        inputs=input_image,
        outputs=[output_image, detection_text]
    )
    
    gr.Examples(
        examples=[
            "./examples/car1.jpg",
            "./examples/car2.jpg",
        ],
        inputs=input_image,
    )
    
    gr.Markdown("""
    ## How to use
    1. Upload an image containing a license plate
    2. Click the "Detect License Plates" button
    3. View the detected license plates with bounding boxes
    4. Check the detection details for confidence scores and coordinates
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)  # Set share=False if you don't want a public link