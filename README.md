# Person Detection using YOLOv8n (ONNX)

This repository contains a lightweight, efficient script for detecting persons in images using the YOLOv8 Nano model in ONNX format. It is designed to be simple to understand and easy to run/deploy without needing the full Ultralytics YOLO package.

## 📂 Project Structure

```
person_detection_using_yolov8n_onnx/
├── models/
│   └── yolov8n.onnx      # Pre-trained YOLOv8 Nano model
├── detect.py             # Main detection script
├── test.jpg              # Sample input image
├── README.md             # This documentation
└── ...
```

## 🛠️ Prerequisites

To run this code, you need Python installed along with the following libraries:

- **OpenCV** (`cv2`): For image processing and visualization.
- **NumPy**: For numerical operations and array manipulation.
- **ONNX Runtime**: For loading and running the ONNX model efficiently.

## 🚀 Installation

You can install the required dependencies using pip:

```bash
pip install opencv-python numpy onnxruntime
```
*(Note: If you have a GPU, you might want to install `onnxruntime-gpu` instead for faster inference).*

## 🏃 Usage

1.  **Place your image**: Ensure you have an image named `test.jpg` in the project root directory (or update the `IMAGE_PATH` variable in `detect.py` to point to your image).
2.  **Run the script**:

    ```bash
    python detect.py
    ```

3.  **View Results**: The script will open a window named "Result" showing the detected persons with bounding boxes and confidence scores. It will also print the total count of detected persons in the console.

## 🧠 Code Explanation

This script (`detect.py`) is capable of:
1.  **Loading the Model**: It initializes an ONNX Inference Session with `models/yolov8n.onnx`.
2.  **Preprocessing**:
    - The input image is resized to **640x640** (the standard input size for YOLOv8).
    - It is converted from BGR to RGB.
    - Pixel values are normalized to `[0, 1]`.
    - Approximately reshapes the dimensions to `(Batch, Channels, Height, Width)`.
3.  **Inference**: The processed image is passed through the model to get raw predictions.
4.  **Post-Processing**:
    - **Filtering**: We iterate through the predictions and only keep those that match the **Person** class (Class ID `0`) and have a confidence score higher than `0.5`.
    - **Bounding Box Scaling**: The coordinates are scaled back from the 640x640 model size to the original image dimensions.
    - **NMS (Non-Maximum Suppression)**: We use `cv2.dnn.NMSBoxes` to remove overlapping boxes and keep only the best detection for each person.
5.  **Visualization**: Finally, it draws green bounding boxes and labels on the image and displays the result.

## ⚙️ Configuration

You can easily tweak these variables at the top of `detect.py`:

- `CONF_THRESHOLD`: Minimum confidence score to detect a person (default: `0.5`).
- `NMS_THRESHOLD`: Threshold for removing overlapping boxes (default: `0.45`).
- `IMAGE_PATH`: Path to the input image.

---
*Created for review and educational purposes.*
