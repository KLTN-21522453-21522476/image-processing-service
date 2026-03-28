# Image Processing Service

A Flask-based RESTful API service for processing images using various machine learning models (e.g., YOLO, PyTorch, PyTesseract, VietOCR). It handles multi-model inference, dynamic model downloading on startup, and parallel image processing under a unified API.

## Features
- **RESTful API**: Exposes an endpoint to upload and process images.
- **Concurrent Processing**: Leverages thread pool executors to process multiple images in parallel.
- **Dynamic Model Loading**: Automatically downloads required models on startup based on `ModelsConfig`.
- **Multi-Model Support**: Supports inference using single models like `yolo8v6` and complex multi-model pipelines via `MultiModelImageProcessingService`.

## Requirements
- Python 3.9+
- Flask & Werkzeug
- OpenCV
- PyTorch & Torchvision
- Ultralytics (YOLO)
- PyTesseract & VietOCR
- Appwrite (for file storage support)

See `src/requirements.txt` for the full list of dependencies.

## Usage

### Running the Application

1. Install dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```
2. Start the server:
   ```bash
   python src/main.py
   ```
The service will start on `http://0.0.0.0:5002`.

### API Endpoint

**POST `/image-process`**
- **Query Parameters**:
  - `model`: Model name to use (e.g., `yolo8v6`).
- **Body**: 
  - `files`: One or more image files (`multipart/form-data`).

**Success Response (200 OK)**
Returns a JSON array containing the detailed processing results for each uploaded image.

## Configuration
Model configurations and download URLs are managed in `src/models_config.py`. Upload and results paths are managed in `src/config.py`.