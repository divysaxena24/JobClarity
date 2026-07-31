from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="JobClarity API",
    version="1.0.0",
    description="AI-powered Fake Job Detection API"
)

app.include_router(router)