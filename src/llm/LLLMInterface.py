from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any,Union


class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model_id: str) -> None:
        pass


    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str) -> Dict[str, str]:
        pass

