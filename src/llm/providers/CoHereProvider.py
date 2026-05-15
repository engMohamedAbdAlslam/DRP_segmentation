from ..LLLMInterface import LLMInterface 
import cohere # type: ignore
import logging
from ..LLMEnum import CoHereEnum 

class CoHereProvider(LLMInterface):
    def __init__(self , api_key : str , 
                        default_input_max_chars :int = 2000,
                        default_output_max_tokens :int = 2000,
                        default_temperature :float =0.1):
        
        self.api_key = api_key
        self.enums = CoHereEnum
        self.default_input_max_chars = default_input_max_chars
        self.default_output_max_tokens = default_output_max_tokens
        self.default_temperature = default_temperature

        self.generation_model_id = None


        self.client = cohere.Client(api_key=self.api_key)
        self.async_client = cohere.AsyncClient(api_key=self.api_key)
        
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self,model_id : str):
        self.generation_model_id = model_id
        
    def generate_text(self,prompt : str, chat_history : list = [] , max_output_tokens:int= None , temperature:float = None): # type: ignore
       
        if not self.client:
            self.logger.error("Co here client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("generation model for co here was not found")
            return None
       
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_output_max_tokens
        temperature = temperature if temperature else self.default_temperature

        
        response = self.client.chat(
            model = self.generation_model_id,
            chat_history = chat_history,
            message = self.process_text(text= prompt), # type: ignore
            max_tokens= max_output_tokens,
            temperature = temperature
        )
        if not response or not response.text:
            self.logger.error("error while generation text with CoHere")
            return None

        return response.text
  
    def process_text(self, text:str):
        return text[:self.default_input_max_chars].strip()
    

    def construct_prompt(self, prompt : str , role :str):
        return {
            "role":role,
            "text":prompt
        }
