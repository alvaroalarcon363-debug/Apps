# ════════════════════════════════════════════════════════
#  forecastpy_backend/app/main.py
#
#  Punto de entrada del servidor FastAPI.
#
#  Arrancar con:
#    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#
#  Documentación interactiva:
#    http://localhost:8000/docs   ← Swagger UI
#    http://localhost:8000/redoc  ← ReDoc
# ════════════════════════════════════════════════════════
 
import logging
import sys
from contextlib import asynccontextmanager
 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
 
from app.config import HOST, PORT, DEBUG, MODELS_DIR, CSV_PATH
from app.api.routes import router
 
# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("forecastpy")
 
 
# ── Lifespan (startup / shutdown) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("=" * 55)
    logger.info("  ForecastPY API — Iniciando servidor")
    logger.info(f"  CSV:     {CSV_PATH}")
    logger.info(f"  Modelos: {MODELS_DIR}")
    logger.info(f"  Debug:   {DEBUG}")
    logger.info("=" * 55)
 
    if not CSV_PATH.exists():
        logger.warning(f"⚠ CSV no encontrado: {CSV_PATH}")
        logger.warning("  Subí el CSV con POST /upload-csv")
 
    modelos = list(MODELS_DIR.glob("model_*.keras"))
    if modelos:
        logger.info(f"✓ Modelos: {[m.stem for m in modelos]}")
    else:
        logger.warning("⚠ Sin modelos entrenados. Ejecuta POST /retrain")
 
    yield
 
    # SHUTDOWN
    logger.info("ForecastPY API — Servidor detenido.")
 
 
# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="ForecastPY API",
    description=(
        "API de predicción de demanda, costo y rentabilidad "
        "para productos de agua mineral. "
        "LSTM + TensorFlow · Open-Meteo · Cordillera, Paraguay."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
 
# ── CORS: permite llamadas desde Flutter ──────────────────
# En producción reemplazar "*" por la IP/dominio específico
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Registrar rutas ───────────────────────────────────────
app.include_router(router)
 
 
# ── Ejecución directa ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info",
    )