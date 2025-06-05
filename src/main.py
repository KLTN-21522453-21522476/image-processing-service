import os
from flask import Flask
from config import Config
from controllers.controller import image_processing_bp
from services.model_downloader import ModelDownloader
from models_config import ModelsConfig  # Import config models

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.RESULT_FOLDER, exist_ok=True)
    
    # Initialize downloader
    downloader = ModelDownloader()
    
    # Use centralized model list - NO MORE HARDCODED LIST HERE!
    downloader.download_models(ModelsConfig.DOWNLOAD_MODEL_LIST)
    
    app.register_blueprint(image_processing_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5002)
