import json
import re
from fastapi import APIRouter, Request, UploadFile, File, status
from fastapi.responses import JSONResponse
from controllers.VisionController import VisionController
from models.enums.ResponesEnums import ResponseSignal

vision_router = APIRouter(
    prefix="/apiv1/vision",
    tags=["api_v1", "vision"]
)

def clean_json_string(text: str) -> str:
    text = re.sub(r'^```json\s*|```$', '', text.strip(), flags=re.MULTILINE)
    return text.strip()

@vision_router.post("/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "INVALID_FILE_TYPE", "message": "Uploaded file must be an image"}
        )

    image_bytes = await file.read()
    
    vision_controller = VisionController(
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser,
        segmentation_engine=request.app.segmentation_engine
    )

    try:
        result = await vision_controller.analyze_retina_image(image_bytes)
        
        # تنظيف وتحويل التقرير
        cleaned_report = clean_json_string(result["report"]) if result["report"] else "{}"
        try:
            formatted_report = json.loads(cleaned_report)
        except json.JSONDecodeError:
            formatted_report = {"raw": result["report"]}

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "signal": ResponseSignal.REPORT_RESPONSE_SUCCESS.value,
                "original_image": result["original_image"],
                "overlay_image": result["overlay_image"],
                "report": formatted_report,
                "full_prompt": result["full_prompt"]
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.REPORT_RESPONSE_ERROR.value, "error": str(e)}
        )