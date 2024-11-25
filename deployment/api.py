from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
from ultralytics import YOLO
import subprocess
import shutil

app = FastAPI()

project_dir = os.path.join(os.path.dirname(os.getcwd()), 'Real-time_Number_Plate_Recognition')
depl_dir = os.path.join(project_dir, 'deployment')
weights_dir = os.path.join(depl_dir, 'weights')
yolov5_dir = os.path.join(project_dir, 'yolov5')
temp_dir = os.path.join(depl_dir, 'temp')

# Predefined YOLO models with paths
MODEL_WEIGHTS = {
    "YOLO_v5": os.path.join(weights_dir, 'yolov5.pt'),
    "YOLO_v11_10": os.path.join(weights_dir, 'yolov11_10.pt'),
    "YOLO_v11_35": os.path.join(weights_dir, 'yolov11_35.pt'),
}

loaded_models = {}

def load_yolo_model(model_name: str):
    """Load YOLOv11 model dynamically."""
    if model_name not in loaded_models:
        weights_path = MODEL_WEIGHTS.get(model_name)
        if not weights_path:
            raise ValueError(f"Model {model_name} not found!")
        loaded_models[model_name] = YOLO(weights_path)
    return loaded_models[model_name]

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en')

# Directories
output_dir = os.path.join(depl_dir, "static")
Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(temp_dir).mkdir(parents=True, exist_ok=True)

class DetectionResult(BaseModel):
    text: str
    image_path: str

@app.post("/detect/", response_model=DetectionResult)
async def detect_plate(
    file: UploadFile = File(...),
    model_name: str = Form(...),  # Select model dynamically
):
    if model_name not in MODEL_WEIGHTS:
        return JSONResponse({"error": f"Model {model_name} not available."}, status_code=400)

    weights_path = MODEL_WEIGHTS[model_name]

    # Save uploaded file
    img_data = await file.read()
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    temp_image_path = os.path.join(temp_dir, "input.jpg")
    cv2.imwrite(temp_image_path, img)

    if model_name.startswith("YOLO_v5"):
        # Handle YOLOv5 using detect.py
        detection_output_dir = os.path.join(temp_dir, "yolov5_results")
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item_path}")
        labels_dir = os.path.join(detection_output_dir, "labels")

        detect_command = [
            "python", os.path.join(yolov5_dir, "detect.py"),
            "--weights", weights_path,
            "--img", "416",
            "--conf", "0.4",
            "--source", temp_image_path,
            "--save-txt",
            "--save-conf",
            "--project", temp_dir,
            "--name", "yolov5_results",
        ]
        subprocess.run(detect_command, check=True)

        label_file = os.path.join(labels_dir, "input.txt")
        if not os.path.exists(label_file):
            return JSONResponse({"error": "No license plate detected."}, status_code=400)

        # Read bounding boxes and crop image
        with open(label_file, 'r') as f:
            max_conf = -1.0
            x1, y1, x2, y2 = 0, 0, 0, 0
            for line in f:
                class_id, x_center, y_center, width, height, conf = map(float, line.split())
                if conf > max_conf:
                    max_conf = conf
                    img_height, img_width = img.shape[:2]
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height

                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

        cropped_plate = img[y1:y2, x1:x2]
    else:
        # Handle YOLOv11
        model = load_yolo_model(model_name)
        results = model(img)
        max_conf = -1
        x1, y1, x2, y2 = 0, 0, 0, 0

        for det in results[0].boxes.data:
            conf = float(det[4])
            if conf > max_conf:
                max_conf = conf
                x1, y1, x2, y2 = map(int, det[:4])

        if max_conf == -1:
            return JSONResponse({"error": "No license plate detected."}, status_code=400)

        cropped_plate = img[y1:y2, x1:x2]

    # Save cropped plate
    cropped_path = os.path.join(output_dir, "cropped_plate.jpg")
    cv2.imwrite(cropped_path, cropped_plate)

    # Run OCR
    result = ocr.ocr(cropped_path, cls=True)
    recognized_text = "".join([line[-1][0] for line in result[0]])

    # Save result image with bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    result_img_path = os.path.join(output_dir, "result.jpg")
    cv2.imwrite(result_img_path, img)

    return DetectionResult(text=recognized_text, image_path=result_img_path)

# Serve static files
app.mount("/static", StaticFiles(directory=output_dir), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
