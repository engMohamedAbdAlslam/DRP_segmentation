import logging

from llm.LLMEnum import CoHereEnum
from .BaseController import BaseController

from llm.tamplates.template_parser import TemplateParser

class NLPController(BaseController):
    def __init__(self,  generation_client, template_parser: TemplateParser):
        super().__init__()
        self.generation_client = generation_client
        self.template_parser = template_parser
        self.logger = logging.getLogger(__name__)

    
    async def generation_report(self, model_result: list): 

        system_prompt = self.template_parser.get(group="report", key="system_prompt")


        formatted_results = []
        for idx, model in enumerate(model_result):
            metrics_str = ", ".join([f"{k}: {v}" for k, v in model['metrics'].items()])
            
            res_prompt = self.template_parser.get(
                group="report",
                key="model_result_prompt", 
                vars={"model_num": idx + 1, "recall": metrics_str} )
            formatted_results.append(res_prompt)

        model_result_prompt = "\n".join(formatted_results)
        
        footer_prompt = self.template_parser.get(group="report", key="footer_prompt")
        full_prompt = "\n\n".join([model_result_prompt, footer_prompt or ""])
        chat_history=[
                    {"role": CoHereEnum.SYSTEM.value, "message": system_prompt}
]
        report = self.generation_client.generate_text(prompt=full_prompt, chat_history=chat_history)

        return report, full_prompt
