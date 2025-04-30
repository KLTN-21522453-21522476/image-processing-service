import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    REPO_MODEL_ID = os.environ.get('REPO_MODEL_ID', '')
    REPO_MODEL_API = os.environ.get('REPO_MODEL_API', '')
    
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
    MODELS_FOLDER = os.path.abspath(os.path.join(BASE_DIR, 'ml_models'))
    
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)