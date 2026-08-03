from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://jobclarity-frontend.vercel.app",
]

app = FastAPI(
    title="JobClarity API",
    version="1.0.0",
    description="AI-powered Fake Job Detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)