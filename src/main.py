import os
from flask import Flask
from config import Config
from controllers.controller import image_processing_bp
from services.model_downloader import ModelDownloader

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.RESULT_FOLDER, exist_ok=True)
    downloader = ModelDownloader()
    
    model_list = ["yolo5.pt", "yolo6.pt", "yolo7.pt", "yolo10.pt", "yolo11.pt", "yolo8v6.pt", "yolo8.pt"]
    downloader.download_models(model_list)
    
    app.register_blueprint(image_processing_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5002)
