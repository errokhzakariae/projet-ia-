from fastapi import FastAPI
from backend.app.api.extract import router as extract_router
from backend.app.api.classify import router as classify_router
from backend.app.api.summarize import router as summarize_router


app = FastAPI(
    title="LegalAI-FR Backend",
    description="Backend intelligent pour l'analyse de documents juridiques français.",
    version="0.1.0",
)

app.include_router(classify_router)
app.include_router(extract_router)
app.include_router(summarize_router)

@app.get("/")
def root() -> dict:
    return {
        "project": "LegalAI-FR",
        "status": "running",
        "docs": "/docs",
    }
