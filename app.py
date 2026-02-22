import io
import os
import uuid
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Bangla License Plate Scanner")

# Load model
MODEL_PATH = "./license_plate_detection/yolov11_training/weights/best.pt"
model = YOLO(MODEL_PATH)

# Setup directories
os.makedirs("static", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/logo.png")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# In-memory history (Session based)
detection_history = []

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25):
    start_time = time.time()
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Run Inference with custom confidence
    results = model(img, conf=conf)
    result = results[0]
    
    # Plot results
    annotated_img = result.plot()
    
    # Save optimized result
    res_id = str(uuid.uuid4())[:8]
    output_filename = f"static/results/{res_id}.jpg"
    cv2.imwrite(output_filename, annotated_img)
    
    # Parse detections
    detections = []
    for box in result.boxes:
        c = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append({
            "class": model.names[cls],
            "confidence": f"{c:.2%}"
        })
    
    process_time = f"{time.time() - start_time:.3f}s"
    
    # Update History
    history_item = {
        "id": res_id,
        "time": time.strftime("%H:%M:%S"),
        "count": len(detections),
        "img": f"/{output_filename}"
    }
    detection_history.insert(0, history_item)
    
    return {
        "image_url": f"/{output_filename}",
        "detections": detections,
        "process_time": process_time,
        "history": detection_history[:5]  # Return last 5 items
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
