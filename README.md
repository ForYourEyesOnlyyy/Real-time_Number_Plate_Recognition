# Real-Time Car License Plate Recognition System

## Overview

The **Real-Time Car License Plate Recognition System** is designed to automatically detect and recognize vehicle license plates in real-time. This project leverages **YOLOv5** for fast and efficient license plate detection and **EasyOCR** for accurate character recognition. Our system aims to perform robustly under various conditions, including different lighting, angles, and plate designs, making it suitable for use in traffic management, toll collection, law enforcement, and more.

## Features

- **Real-time License Plate Detection:** Utilizing YOLOv5, the system can detect license plates from live video streams or images with low latency.
- **Accurate Character Recognition:** EasyOCR enables high-accuracy recognition of text characters, supporting multiple languages and plate formats.
- **Robust to Variations:** Works well under various challenging conditions like occlusions, different lighting environments, and distorted plate positions.
- **Lexicon-Based Filtering:** Validates recognized text against known license plate formats to enhance recognition accuracy.

## System Architecture

The system is composed of two main components:
1. **License Plate Detection** using YOLOv5.
2. **Character Recognition** using EasyOCR.

After detecting the plate, post-processing steps such as lexicon-based filtering and confidence thresholding ensure accurate and reliable output.


## Datasets

We use publicly available datasets for training and evaluation:

- **[Kaggle’s Car License Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection):** A dataset containing 433 images with annotated license plates.
- **[Roboflow’s Vehicle Registration Plates Dataset](https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk):** Over 8,800 images with diverse license plate annotations for robust training.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- YOLOv5 by Ultralytics
- EasyOCR by JaidedAI
- Kaggle and Roboflow for providing datasets