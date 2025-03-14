import os
import sys
import signal
from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from src.api.endpoints import router

# Load environment variables
env_file = ".env.production" if os.getenv("ENVIRONMENT") == "production" else ".env"
load_dotenv(dotenv_path=env_file)

ENVIRONMENT = os.getenv("ENVIRONMENT")
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))
LOG_LEVEL = os.getenv("LOG_LEVEL")

app = FastAPI(title="Sentiment Analysis API", debug=(ENVIRONMENT == "development"))

app.include_router(router)

# Configure logging
logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level=LOG_LEVEL.upper())

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Monitoring (Only for production)
if ENVIRONMENT == "production":
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

def shutdown_handler(*args):
    logger.info("Shutting down API...")
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

logger.info(f"API running in {ENVIRONMENT.upper()} mode on {HOST}:{PORT}")
