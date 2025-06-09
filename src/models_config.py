# models_config.py
"""
Central configuration file for all YOLO models.
Programmers only need to update this file to add/remove models.
"""

class ModelsConfig:
    """Configuration for YOLO models used in the application."""
    
    # List of all models to download
    DOWNLOAD_MODEL_LIST = [
        "yolo5.pt",
        "yolo6.pt", 
        "yolo7.pt",
        "yolo8.pt",
        "yolo8v2.pt",
        "yolo8v6.pt",
        "yolo10.pt",
        "yolo10v2.pt",
        "yolo11.pt",
        "yolo11v2.pt",
        "yolo10_full_text.pt"
    ]
    
    # Mapping of model aliases to actual model files
    # This allows users to use friendly names instead of exact filenames
    AVAILABLE_MODELS = {
        'yolo5': 'yolo5.pt',
        'yolo6': 'yolo6.pt',
        'yolo7': 'yolo7.pt',
        'yolo8': 'yolo8.pt',
        'yolo8v2': 'yolo8v2.pt',
        'yolo8v6': 'yolo8v6.pt',
        'yolo10': 'yolo10.pt',
        'yolo10_full_text': 'yolo10_full_text.pt',
        'yolo10v2': 'yolo10v2.pt',
        'yolo11': 'yolo11.pt',
        'yolo11v2': 'yolo11v2.pt',
        'default': 'yolo11.pt'  # Default model
    }
    
    @classmethod
    def get_model_file(cls, model_name: str) -> str:
        """
        Get the actual model file name from alias.
        
        Args:
            model_name: Model alias or filename
            
        Returns:
            Actual model filename
        """
        return cls.AVAILABLE_MODELS.get(model_name, cls.AVAILABLE_MODELS['default'])
    
    @classmethod
    def get_available_model_names(cls) -> list:
        """Get list of available model aliases."""
        return list(cls.AVAILABLE_MODELS.keys())
    
    @classmethod
    def is_valid_model(cls, model_name: str) -> bool:
        """Check if a model name is valid."""
        return model_name in cls.AVAILABLE_MODELS
