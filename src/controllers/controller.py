import os
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
    if model_name == 'yolo8v6':
        image_processor = ImageProcessingService()
    else:
        image_processor = MultiModelImageProcessingService(model_name)
    
    
    if 'files' in request.files:
        image_files = request.files.getlist('files') 
        
        if not image_files or all(file.filename == '' for file in image_files):
            return jsonify({"error": "No selected image files"}), 400
        
        response = []
        for image_file in image_files:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            image_file.save(filepath)
            
            try:
                results = image_processor.process_image(filepath, filename, model_name)
                response.append(results)                        
            except Exception as e:
                error_result = ErrorResponse(filename, str(e))
                results.append(error_result.to_dict())

        return json.dumps(response, ensure_ascii=False, indent=4), 200
    else:
        return jsonify({'error': 'No files uploaded'}), 400
 