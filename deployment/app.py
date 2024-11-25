import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("License Plate Detection and Recognition")

# List of YOLO models
yolo_models = ["YOLO_v5", "YOLO_v11_10", "YOLO_v11_35"]
selected_model = st.selectbox("Select YOLO Model", yolo_models)

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Process"):
        # Send to FastAPI
        files = {"file": uploaded_file.getvalue()}
        data = {"model_name": selected_model}
        response = requests.post("http://localhost:8000/detect/", files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            detected_img_url = f"http://localhost:8000/static/result.jpg"
            st.image(detected_img_url, caption="Processed Image", use_container_width=True)
            st.success(f"Detected Plate Text: {result['text']}")
        else:
            st.error("Detection failed. Please try again.")
