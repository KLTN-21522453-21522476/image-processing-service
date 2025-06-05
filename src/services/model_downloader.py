import os
from huggingface_hub import snapshot_download, hf_hub_download
import logging
from config import Config

class ModelDownloader:
    """
    A class to handle downloading models from Hugging Face Hub.
    """
    
    def __init__(self, models_folder=None, log_level=logging.INFO):
        """
        Initialize the ModelDownloader.
        
        Args:
            models_folder (str, optional): Path to store downloaded models. 
                                          Defaults to Config.MODELS_FOLDER.
            log_level (int, optional): Logging level. Defaults to logging.INFO.
        """
        # Set up logging
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Set models folder
        self.models_folder = models_folder if models_folder else Config.MODELS_FOLDER
        os.makedirs(self.models_folder, exist_ok=True)
        
    def download_models(self, models_list):
        """
        Download YOLO models from Hugging Face if they don't exist locally.
        
        Args:
            models_list (list): List of model filenames to download
            
        Returns:
            dict: Dictionary mapping model names to their local paths
        """
        downloaded_models = {}
        
        for model_name in models_list:
            model_path = os.path.join(self.models_folder, model_name, model_name)
            folder_path = os.path.join(self.models_folder, model_name)
            
            # Check if the model already exists
            if os.path.exists(model_path):
                self.logger.info(f"Model {model_name} already exists in {self.models_folder}")
            else:
                self.logger.info(f"Downloading model {model_name} to {model_path}")
                try:
                    # Download the complete model repository
                    hf_hub_download(
                        repo_id=Config.REPO_MODEL_ID,
                        local_dir=folder_path,
                        filename=model_name,
                        token=Config.REPO_MODEL_API,
                    )
                    self.logger.info(f"Successfully downloaded {model_name}")
                except Exception as e:
                    self.logger.error(f"Error downloading {model_name}: {str(e)}")
            
            downloaded_models[model_name] = model_path
            
        return downloaded_models
