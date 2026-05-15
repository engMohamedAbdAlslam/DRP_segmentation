from fastapi import FastAPI
from routes import nlp
from llm.LLMProviderFactory import LLMProviderFactory
from llm.tamplates.template_parser import TemplateParser

from helpers.config import get_settings

app = FastAPI()

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
    



# ===== Routers =====
app.include_router(nlp.nlp_router)
