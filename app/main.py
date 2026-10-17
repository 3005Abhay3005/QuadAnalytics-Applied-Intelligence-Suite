"""
main.py — FastAPI application entry point.
Serves the REST API on /api/* and static frontend files on /*.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import mimetypes
import logging

# Fix Windows registry MIME type association issues
mimetypes.init()
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("image/svg+xml", ".svg")

from app.routes import router

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load all model artefacts at startup to avoid cold-start latency."""
    logger.info("Pre-loading ML model artefacts...")
    try:
        from app.predictor import (
            _load_diabetes,
            _load_breast_cancer,
            _load_car_price,
            _load_ipl,
        )
        _load_diabetes()
        logger.info("  - Diabetes model loaded")
        _load_breast_cancer()
        logger.info("  - Breast Cancer model loaded")
        _load_car_price()
        logger.info("  - Car Price model loaded")
        _load_ipl()
        logger.info("  - IPL Score model loaded")
        logger.info("All models ready.")
    except FileNotFoundError as e:
        logger.warning(f"Some models not yet trained: {e}")
        logger.warning("Run: python ml-website/train_all.py")
    yield
    logger.info("Shutting down ML platform.")


app = FastAPI(
    title="Quad Analytics",
    description="A portfolio of Machine Learning prediction models",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Serve CSS, JS and other static assets at their natural paths
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/js",  StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")),  name="js")

# Serve any remaining static assets (images, fonts, etc.) under /assets if the folder exists
_assets_dir = os.path.join(FRONTEND_DIR, "assets")
if os.path.isdir(_assets_dir):
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/{page}.html")
def serve_page(page: str):
    path = os.path.join(FRONTEND_DIR, f"{page}.html")
    if os.path.exists(path):
        return FileResponse(path)
    not_found = os.path.join(FRONTEND_DIR, "404.html")
    if os.path.exists(not_found):
        return FileResponse(not_found, status_code=404)
    from fastapi.responses import JSONResponse
    return JSONResponse({"detail": "Page not found"}, status_code=404)
