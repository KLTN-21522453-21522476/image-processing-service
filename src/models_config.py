# models_config.py
"""
Central configuration file for all YOLO models.
Programmers only need to update this file to add/remove models.
"""

class ModelsConfig:
    """Configuration for YOLO models used in the application."""
    
    # List of all models to download
    DOWNLOAD_MODEL_LIST = [
        "YOLO10v2.pt",
        "YOLO11v2.pt",
        "yolo8.pt",
        "YOLO8v3.pt",
        "yolo8v6.pt",
    ]

    
    # Mapping of model aliases to actual model files
    # This allows users to use friendly names instead of exact filenames
    AVAILABLE_MODELS = {
        'yolo8': 'YOLO8v3.pt',
        'yolo8v6': 'yolo8v6.pt',
        'yolo10': 'YOLO10v2.pt',
        'yolo11': 'yolo11v2.pt',
        'default': 'yolo11v2.pt'
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
