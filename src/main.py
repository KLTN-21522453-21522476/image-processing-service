import os
from flask import Flask
from config import Config
from controllers.controller import image_processing_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.RESULT_FOLDER, exist_ok=True)
    
    app.register_blueprint(image_processing_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5002)
