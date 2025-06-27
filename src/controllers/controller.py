import os
import concurrent.futures
from models.error_response import ErrorResponse
from werkzeug.utils import secure_filename
from config import Config
from flask import request, json, jsonify, Blueprint
from services.image_processing import ImageProcessingService
from services.multi_model_image_processing import MultiModelImageProcessingService

image_processing_bp = Blueprint("image_processing", __name__)

@image_processing_bp.route("/image-process", methods=["POST"])
def image_process():
    
    model_name = request.args.get('model')
    if model_name == None:
        return jsonify({"error": "No selected model"}), 400
    
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    image_files = request.files.getlist('files') 
    
    if not image_files or all(file.filename == '' for file in image_files):
        return jsonify({"error": "No selected image files"}), 400
    
    # Save all files first
    file_data = []
    for image_file in image_files:
        if image_file.filename:  # Check if filename is not empty
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            image_file.save(filepath)
            file_data.append((filepath, filename))
    
    # Process files in parallel
    response = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for filepath, filename in file_data:
            if model_name == 'yolo8v6':
                image_processor = ImageProcessingService()
            else:
                image_processor = MultiModelImageProcessingService(model_name)
                
            future = executor.submit(process_single_image, image_processor, filepath, filename, model_name)
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            response.append(future.result())
    
    return json.dumps(response, ensure_ascii=False, indent=4), 200

def process_single_image(processor, filepath, filename, model_name):
    try:
        result = processor.process_image(filepath, filename, model_name)
        return result                       
    except Exception as e:
        error_result = ErrorResponse(filename, str(e))
        return error_result.to_dict()