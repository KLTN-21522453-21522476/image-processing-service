import os

class Config:
    UPLOAD_FOLDER = '/tmp/uploads'
    RESULT_FOLDER = '/tmp/results'
    MODELS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
    
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)