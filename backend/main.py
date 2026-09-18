import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# Ensure backend and src are in path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for p in [BACKEND_DIR, PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from routes.health import router as health_router
from routes.investigation import router as investigation_router
from routes.actions import router as actions_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ArthadrishtiAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Arthadrishti API backend...")
    logger.info("Arthadrishti API routes initialized successfully.")
    yield
    logger.info("Shutting down Arthadrishti API backend...")

app = FastAPI(
    title="Arthadrishti — AI-Powered Fraud Investigation Workspace API",
    description="Production-grade REST backend for LightGBM, SHAP, TigerGraph, and OpenRouter Multi-Model Evidence Engine.",
    version="1.12-rt",
    lifespan=lifespan
)

# Enable CORS for all local development servers (Live Server, Vite, Webpack, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(health_router)
app.include_router(investigation_router)
app.include_router(actions_router)

@app.get("/")
def serve_index():
    """
    Serves the pre-built code.html frontend directly at root.
    """
    html_path = os.path.join(PROJECT_ROOT, "code.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {
        "workspace": "Arthadrishti — AI-Powered Fraud Investigation Workspace",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
