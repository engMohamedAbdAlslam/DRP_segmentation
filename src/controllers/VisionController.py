from controllers.BaseController import BaseController
from controllers.NLPController import NLPController
from engine.segmentation_engine import SegmentationEngine

class VisionController(BaseController):
    def __init__(self, generation_client, template_parser, segmentation_engine: SegmentationEngine):
        super().__init__()
        self.nlp_controller = NLPController(
            generation_client=generation_client,
            template_parser=template_parser
        )
        self.segmentation_engine = segmentation_engine

    async def analyze_retina_image(self, image_bytes: bytes):
        vision_res = self.segmentation_engine.predict_and_overlay(image_bytes)

        model_results = [
            {
                "name": "U-Net Vessels", 
                "metrics": {"Vessel Density": f"{vision_res['metrics_summary']['vessel_density_percentage']}%"}
            },
            {
                "name": "U-Net Lesions", 
                "metrics": {
                    "Total Lesion Pixels": str(vision_res['metrics_summary']['total_lesion_pixels']),
                    **{k: str(v) for k, v in vision_res['metrics_summary']['lesions_breakdown'].items() if v > 0}
                }
            }
        ]

        report, full_prompt = await self.nlp_controller.generation_report(model_result=model_results)

        return {
            "original_image": vision_res["original_image_base64"],
            "overlay_image": vision_res["overlay_image_base64"],
            "report": report,
            "full_prompt": full_prompt
        }