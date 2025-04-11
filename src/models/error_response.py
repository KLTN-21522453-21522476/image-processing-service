class ErrorResponse:
    def __init__(self, img_name: str, error: str):
        self.img_name = img_name
        self.error = error
        
    def to_dict(self):
        return {
            "img_name": self.img_name,
            "error": self.error
        }