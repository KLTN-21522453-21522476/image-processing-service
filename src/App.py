import os
from flask import Flask
from app.routes import Routes
from app.configurations.Config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.RESULT_FOLDER, exist_ok=True)
    
    Routes.Routes(app)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run()
