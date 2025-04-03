import os
from werkzeug.utils import secure_filename
from app.configurations.Config import Config
from flask import request, json, jsonify
from app.services.ImageProcessingService import ImageProcessingService



class Controller:
    def __init__(self):
        self.image_processor = ImageProcessingService()

    
    def ProcessImages(self):
        if request.method == 'POST':
            if 'files' in request.files:
                image_files = request.files.getlist('files') 
                
                if not image_files or all(file.filename == '' for file in image_files):
                    return jsonify({"error": "No selected image files"}), 400
                
                results = []
                for image_file in image_files:
                    filename = secure_filename(image_file.filename)
                    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
                    image_file.save(filepath)
                    
                    try:
                        result_json = self.image_processor.process_image(filepath, filename)
                        results.append(json.loads(result_json))                        
                    except Exception as e:
                        return jsonify({"error": f"Error processing image {filename}: {str(e)}"}), 500

                return jsonify(results)  
            else:
                return jsonify({'error': 'No files uploaded'}), 400

        
