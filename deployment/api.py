from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR

app = FastAPI()

project_dir = os.path.dirname(os.getcwd())
depl_dir = os.path.join(project_dir, 'Real-time_Number_Plate_Recognition', 'deployment')

# Initialize YOLO and PaddleOCR
model_weights_location = os.path.join(depl_dir, 'best.pt')
model = YOLO(model_weights_location)
ocr = PaddleOCR(lang='en')

# Directories
output_dir = os.path.join(depl_dir, 'static')
Path(output_dir).mkdir(parents=True, exist_ok=True)

class DetectionResult(BaseModel):
    text: str
    image_path: str

@app.post("/detect/", response_model=DetectionResult)
async def detect_plate(file: UploadFile = File(...)):
    # Save uploaded file
    img_data = await file.read()
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO detection
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

    # Crop plate and save
    cropped_plate = img[y1:y2, x1:x2]
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
