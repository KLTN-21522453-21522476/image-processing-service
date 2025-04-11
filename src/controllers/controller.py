import os
from models.error_response import ErrorResponse
from werkzeug.utils import secure_filename
from config import Config
from flask import request, json, jsonify, Blueprint
from services.image_processing import ImageProcessingService

image_processing_bp = Blueprint("image_processing", __name__)

@image_processing_bp.route("/image-process", methods=["POST"])
def image_process():
    image_processor = ImageProcessingService()

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
                results = image_processor.process_image(filepath, filename)
                response.append(results)                        
            except Exception as e:
                error_result = ErrorResponse(filename, str(e))
                results.append(error_result.to_dict())

        return json.dumps(response, ensure_ascii=False, indent=4), 200
    else:
        return jsonify({'error': 'No files uploaded'}), 400
 