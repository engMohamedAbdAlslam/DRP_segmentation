from helpers.config import get_settings
import os

class BaseController:

    def __init__(self):
        self.app_settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.files_dir = os.path.join(
            self.base_dir,
            "assets/files"
            )
        self.models_dir = os.path.join(self.base_dir,"assets/models")
        
    
    def get_model_path(self,model_name:str):
        database_path = os.path.join(self.models_dir,model_name)
        if not os.path.exists(database_path):
            os.makedirs(database_path)
        return(database_path)
    