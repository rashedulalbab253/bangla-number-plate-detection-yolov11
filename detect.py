#!/usr/bin/env python3
# YOLOv11 inference script that accepts image path from command line

import argparse
import os
from ultralytics import YOLO

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run YOLOv11 inference on an image')
    
    # Add arguments
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to the input image file')
    parser.add_argument('--model', '-m', type=str, 
                        default="./license_plate_detection/yolov11_training/weights/best.pt",
                        help='Path to the trained model file (default: ./license_plate_detection/yolov11_training/weights/best.pt)')
    parser.add_argument('--output', '-o', type=str, 
                        default="./prediction_output.jpg",
                        help='Path to save the output image (default: ./prediction_output.jpg)')
    parser.add_argument('--conf', '-c', type=float, 
                        default=0.25,
                        help='Confidence threshold (default: 0.25)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return
    
    print(f"Loading model from: {args.model}")
    model = YOLO(args.model)
    
    print(f"Running inference on: {args.image}")
    results = model(args.image, conf=args.conf)
    
    # Get the first result
    result = results[0]
    
    # Display the image with predictions (this will work in GUI environments)
    try:
        result.show()
    except Exception as e:
        print(f"Note: Could not display image: {e}")
    
    # Save the prediction to a file
    result.save(filename=args.output)
    print(f"Prediction saved to {args.output}")
    
    # Print the detected objects and confidence scores
    print("\nDetected objects:")
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = float(box.conf[0].item())
        coordinates = box.xyxy[0].tolist()  # x1, y1, x2, y2 format
        
        print(f"Class: {class_name}, Confidence: {confidence:.2f}, Coordinates: {coordinates}")

if __name__ == "__main__":
    main()