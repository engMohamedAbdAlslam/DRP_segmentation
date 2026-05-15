import os
from typing import Optional
class TemplateParser():
    def __init__(self, languge : Optional[str]  ,default_languge = "en" ) :
        self.current_path =os.path.dirname(os.path.abspath(__file__))
        self.default_languge = default_languge

        self.languge = languge
        self.set_languge(languge)

    def set_languge(self, languge : Optional[str]):
        if not languge:
            self.languge = self.default_languge
            return None
        self.languge_path = os.path.join(self.current_path,"locales",languge)
        if os.path.exists(self.languge_path):
            self.languge = languge
        
        else:
            self.languge = self.default_languge


    def get(self, group:str , key : str , vars : dict={}):
        if not group and not key:
            return None
        group_path = os.path.join(self.current_path,"locales",self.languge,f"{group}.py") # type: ignore
        targeted_languge = self.languge

        if not os.path.exists(group_path):
            group_path = os.path.join(self.current_path,"locales",self.default_languge,f"{group}.py") # type: ignore
        targeted_languge = self.default_languge
        if not os.path.exists(group_path):
            return None
        
        module = __import__(f"llm.tamplates.locales.{targeted_languge}.{group}",fromlist=[group])
        if not module:
            return None
        
        key_attribute = getattr(module,key)
        return key_attribute.substitute(vars)