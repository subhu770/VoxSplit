"""
VoxSplit - Modular Audio Transcription & Speaker Diarization Microservice.
FastAPI Application Entrypoint with Lifespan Management, Middleware, and API Routing.
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import router as api_router
from app.api.stream_ws import router as ws_router
from app.services.asr import ASRService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("voxsplit.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle context manager for model warmup and resource cleanup.
    """
    settings = get_settings()
    logger.info("Initializing VoxSplit v%s on Python environment...", settings.APP_VERSION)
    logger.info(
        "VAD Mode: Aggressiveness=%d, Hangover=%d ms, Min Speech=%d ms",
        settings.VAD_AGGRESSIVENESS,
        settings.VAD_PADDING_DURATION_MS,
        settings.VAD_MIN_SPEECH_DURATION_MS
    )
    logger.info(
        "Diarization: Threshold=%.2f, Max Speakers=%d",
        settings.SPEAKER_SIMILARITY_THRESHOLD,
        settings.MAX_SPEAKERS
    )
    logger.info(
        "ASR Engine: Model=%s, Device=%s, Quantization=%s",
        settings.WHISPER_MODEL_SIZE,
        settings.WHISPER_DEVICE,
        settings.WHISPER_COMPUTE_TYPE
    )

    # Optional eager warmup
    if not settings.DEBUG:
        try:
            asr = ASRService(settings)
            asr.load_model()
        except Exception as e:
            logger.warning("Eager model loading deferred to first request: %s", e)

    yield

    logger.info("Shutting down VoxSplit service gracefully.")


def create_application() -> FastAPI:
    """
    Factory creating and configuring the primary FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="VoxSplit",
        description="Production-grade, modular audio transcription and speaker diarization engine.",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Cross-Origin Resource Sharing (CORS) Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request Processing Time Middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": str(exc),
                "path": request.url.path
            }
        )

    # Include Routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    app.include_router(ws_router, prefix=settings.API_V1_PREFIX)

    # Convenience root health redirect / metadata
    @app.get("/health", tags=["Health"], include_in_schema=False)
    async def root_health():
        from app.api.routes import health_check
        return await health_check()

    @app.get("/", tags=["Root"], include_in_schema=False)
    async def root():
        return {
            "service": "VoxSplit",
            "version": settings.APP_VERSION,
            "status": "operational",
            "docs": "/docs"
        }

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

