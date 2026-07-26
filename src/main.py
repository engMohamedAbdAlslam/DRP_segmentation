from pathlib import Path

from fastapi import FastAPI
from routes import nlp,vision
from llm.LLMProviderFactory import LLMProviderFactory
from llm.tamplates.template_parser import TemplateParser
import os
from engine.segmentation_engine import SegmentationEngine
from helpers.config import get_settings

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
VESSEL_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unet_vessels.pth")
LESION_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unet_lesions_30_epoch.pth")

@app.on_event("startup")
async def startup_span():
    settings = get_settings()

    # ===== Factories =====
    llm_provider_factory = LLMProviderFactory(settings)

    # ===== Generation Client =====
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)  # type: ignore

    if app.generation_client is None: # type: ignore
        raise RuntimeError("Generation client was not created")

    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)  # type: ignore

    
    # ====== Template Parser =====
    app.template_parser = TemplateParser(languge=settings.ORGINAL_LANGUGE,default_languge=settings.DEFAULT_LANGUGE) # type: ignore

    # ======== load models ========
    
    app.segmentation_engine = SegmentationEngine(
        vessel_model_path=VESSEL_MODEL_PATH,
        lesion_model_path=LESION_MODEL_PATH

    )

# ===== Routers =====
app.include_router(nlp.nlp_router)
app.include_router(vision.vision_router)
