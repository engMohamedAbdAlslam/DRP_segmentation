from fastapi import APIRouter,Depends,status,Request
from fastapi.responses import JSONResponse
import logging
import json
import re
from models.enums.ResponesEnums import ResponseSignal
from controllers import NLPController

logger = logging.getLogger("uvicorn.error")

nlp_router = APIRouter(
    prefix="/apiv1/nlp",
    tags= [ "api_v1","nlp"]
)
def clean_json_string(text: str) -> str:
    text = re.sub(r'^```json\s*|```$', '', text.strip(), flags=re.MULTILINE)
    return text.strip()

@nlp_router.post("/result/answer")
async def answer_rag(request : Request):
    
    nlp_controller = NLPController(
                    generation_client = request.app.generation_client,
                    template_parser = request.app.template_parser) 
    
    model_results_list = [
    {"name": "SVM", "metrics": {"Recall": "99%", "Precision": "95%"}},
    {"name": "YOLOv8", "metrics": {"Recall": "92%", "mAP": "0.89"}},
    {"name": "U-Net", "metrics": {"Dice Coefficient": "0.88"}}
]
    report, full_prompt= await nlp_controller.generation_report(model_result =model_results_list)
    if not report:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal":ResponseSignal.REPORT_RESPONSE_ERROR.value })
    cleaned_report = clean_json_string(report)
    try:
        formatted_report = json.loads(cleaned_report)
        return JSONResponse(content={"signal":ResponseSignal.REPORT_RESPONSE_SUCCESS.value,
                                     "report":formatted_report,
                                     "full_prompt": full_prompt,
                                     })
    except json.JSONDecodeError as e:
        return JSONResponse(content={"error": "Failed to decode JSON", "raw": report})

     