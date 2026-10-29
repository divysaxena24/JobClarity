import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://jobclarity.vercel.app",
]

env_origins = os.getenv("ALLOWED_ORIGINS")
if env_origins:
    configured_origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    ALLOWED_ORIGINS = list(dict.fromkeys(DEFAULT_ALLOWED_ORIGINS + configured_origins))
else:
    ALLOWED_ORIGINS = list(DEFAULT_ALLOWED_ORIGINS)

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
