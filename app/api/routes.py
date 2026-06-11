# ════════════════════════════════════════════════════════
#  forecastpy_backend/app/api/routes.py
#
#  CORRECCIÓN DEFINITIVA:
#  Eliminado JWT de /predict, /upload-csv y /reporte.
#  Todos los endpoints son accesibles sin token.
#
#  Razón: Flutter usa login SQLite local (db_helper.dart)
#  y no llama a POST /auth/login del backend, por lo que
#  nunca tiene un JWT válido para enviar.
#
#  Para MVP local esto es correcto. Si en el futuro
#  se despliega en producción, se re-activa el JWT.
# ════════════════════════════════════════════════════════

import logging
import asyncio
import shutil
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import List

from fastapi import (
    APIRouter, HTTPException, UploadFile, File,
    BackgroundTasks, status,
)
from fastapi.responses import FileResponse

from app.config import (
    CSV_PATH, REPORTS_DIR, PRODUCTOS_VALIDOS,
    SECRET_KEY, TOKEN_EXPIRE_MIN,
)
from app.schemas.schemas import (
    LoginRequest, TokenResponse,
    PredictRequest, PredictResponse, PuntoHistorico,
    WeatherResponse,
    RetrainResponse, CSVUploadResponse,
    ReporteRequest, ReporteResponse,
    ProductoInfo, ProductosResponse,
    HistoricoResponse,
)
from app.ml.lstm_model   import predecir, entrenar_todos, modelo_existe
from app.ml.weather      import obtener_clima_actual
from app.ml.preprocessor import (
    obtener_precios_recientes, obtener_historico_producto, cargar_csv,
)
from app.utils.report_generator import generar_pdf

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Auth simple con SHA-256 (sin bcrypt) ─────────────────
def _hash_password(password: str) -> str:
    return hmac.new(
        SECRET_KEY.encode("utf-8"),
        password.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

def _make_user(username: str, password: str) -> dict:
    return {
        "username":        username,
        "hashed_password": _hash_password(password),
    }

USERS_DB = {
    "admin":    _make_user("admin",    "admin1234"),
    "forecast": _make_user("forecast", "forecast123"),
}

# ─────────────────────────────────────────────────────────
NOMBRES = {
    "agua_500ml": "Agua Mineral 500 ml",
    "agua_2l":    "Agua Mineral 2 L",
    "agua_5l":    "Agua Mineral 5 L",
    "agua_20l":   "Agua Mineral 20 L",
}
VOLUMENES = {
    "agua_500ml": "500 ml",
    "agua_2l":    "2 L",
    "agua_5l":    "5 L",
    "agua_20l":   "20 L",
}


# ══════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════

@router.get("/", tags=["Sistema"])
async def health():
    """Verifica que el servidor responde."""
    modelos_listos = [
        pid for pid in PRODUCTOS_VALIDOS if modelo_existe(pid)
    ]
    return {
        "status":         "ok",
        "app":            "ForecastPY API",
        "version":        "1.0.0",
        "modelos_listos": modelos_listos,
        "total_modelos":  len(modelos_listos),
    }


# ══════════════════════════════════════════════════════════
#  AUTH — solo para referencia, no bloquea nada en MVP
# ══════════════════════════════════════════════════════════

@router.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(req: LoginRequest):
    """
    Login opcional para Swagger /docs.
    En el MVP Flutter usa SQLite local para autenticar.

    Credenciales:
      admin / admin1234
      forecast / forecast123
    """
    user = USERS_DB.get(req.username)
    if not user:
        raise HTTPException(status_code=401,
                            detail="Usuario o contraseña incorrectos")

    hashed = _hash_password(req.password)
    if not hmac.compare_digest(hashed, user["hashed_password"]):
        raise HTTPException(status_code=401,
                            detail="Usuario o contraseña incorrectos")

    # Token simple (no requerido para usar la API en MVP)
    import json, base64
    payload = {
        "sub": req.username,
        "exp": (datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MIN)).isoformat()
    }
    token = base64.b64encode(
        json.dumps(payload).encode()
    ).decode()

    logger.info(f"Login: {req.username}")
    return TokenResponse(access_token=token, username=req.username)


# ══════════════════════════════════════════════════════════
#  PRODUCTOS — sin auth
# ══════════════════════════════════════════════════════════

@router.get("/productos", response_model=ProductosResponse, tags=["Productos"])
async def listar_productos():
    """Lista los 4 productos con precios del CSV."""
    items: List[ProductoInfo] = []
    for pid in PRODUCTOS_VALIDOS:
        try:
            p    = obtener_precios_recientes(pid)
            rent = round(
                ((p["precio_venta"] - p["precio_costo"]) / p["precio_venta"]) * 100, 1
            ) if p["precio_venta"] > 0 else 0.0
            items.append(ProductoInfo(
                id=pid,
                nombre=NOMBRES[pid],
                volumen=VOLUMENES[pid],
                precio_venta=p["precio_venta"],
                precio_costo=p["precio_costo"],
                margen_gs=p["margen_gs"],
                rentabilidad=rent,
                ultimo_mes=p["ultimo_mes"],
                modelo_listo=modelo_existe(pid),
            ))
        except Exception as e:
            logger.error(f"Error productos {pid}: {e}")
    return ProductosResponse(productos=items)


# ══════════════════════════════════════════════════════════
#  CLIMA — sin auth
# ══════════════════════════════════════════════════════════

@router.get("/clima", response_model=WeatherResponse, tags=["Clima"])
async def clima():
    """Condiciones climáticas actuales de Caacupé, Cordillera, PY."""
    datos = obtener_clima_actual()
    return WeatherResponse(**datos)


# ══════════════════════════════════════════════════════════
#  PREDICCIÓN — SIN AUTH (corrección principal)
# ══════════════════════════════════════════════════════════

@router.post("/predict", response_model=PredictResponse, tags=["Predicción"])
async def predict(req: PredictRequest):
    """
    Ejecuta el modelo LSTM para el producto y período solicitado.
    NO requiere autenticación — Flutter llama directo sin JWT.
    """
    pid = req.producto_id.value

    if not modelo_existe(pid):
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail=(
                f"Modelo no entrenado para '{pid}'. "
                f"Ejecutá primero POST /retrain."
            ),
        )

    # Clima: override manual o Open-Meteo real
    if req.temp_max_override is not None and req.precip_mm_override is not None:
        temp   = req.temp_max_override
        precip = req.precip_mm_override
    else:
        clima_datos = obtener_clima_actual()
        temp        = clima_datos["temp_max_absoluta"]
        precip      = clima_datos["precip_acumulada_mm"]

    try:
        resultado = predecir(
            producto_id=pid,
            meses_futuro=req.meses_futuro.value,
            temp_max_actual=temp,
            precip_mm_actual=precip,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=424, detail=str(e))
    except Exception as e:
        logger.error(f"Error predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    historico = [
        PuntoHistorico(fecha=r["fecha"], cantidad=int(r["cantidad"]))
        for r in resultado.pop("historico")
    ]
    return PredictResponse(**resultado, historico=historico)


# ══════════════════════════════════════════════════════════
#  HISTORIAL — sin auth
# ══════════════════════════════════════════════════════════

@router.get(
    "/historico/{producto_id}",
    response_model=HistoricoResponse,
    tags=["Datos"],
)
async def historico(producto_id: str):
    """Serie histórica completa del producto desde el CSV."""
    if producto_id not in PRODUCTOS_VALIDOS:
        raise HTTPException(
            status_code=404,
            detail=f"Producto no encontrado: {producto_id}",
        )
    registros = obtener_historico_producto(producto_id)
    return HistoricoResponse(producto_id=producto_id, registros=registros)


# ══════════════════════════════════════════════════════════
#  RETRAIN — sin auth
# ══════════════════════════════════════════════════════════

@router.post("/retrain", response_model=RetrainResponse, tags=["Modelo"])
async def retrain(bg: BackgroundTasks):
    """
    Reentrena los 4 modelos LSTM. Sin auth. Background.
    Tiempo estimado: 2-5 minutos.
    """
    logger.info("Retrain iniciado")
    bg.add_task(_bg_retrain)
    return RetrainResponse(
        status="iniciado",
        resultados=[{
            "mensaje": (
                "Entrenando en background. "
                "Revisá la consola del servidor para ver el progreso."
            )
        }],
    )


async def _bg_retrain():
    loop       = asyncio.get_event_loop()
    resultados = await loop.run_in_executor(None, entrenar_todos)
    logger.info(f"Retrain completado: {resultados}")


# ══════════════════════════════════════════════════════════
#  UPLOAD CSV — sin auth
# ══════════════════════════════════════════════════════════

@router.post("/upload-csv", response_model=CSVUploadResponse, tags=["Datos"])
async def upload_csv(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Reemplaza el CSV y reentrena automáticamente."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="El archivo debe ser .csv")

    contenido = await file.read()

    # Backup
    backup = CSV_PATH.parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if CSV_PATH.exists():
        shutil.copy(CSV_PATH, backup)

    CSV_PATH.write_bytes(contenido)

    try:
        df = cargar_csv(CSV_PATH)
    except Exception as e:
        if backup.exists():
            shutil.copy(backup, CSV_PATH)
        raise HTTPException(status_code=422, detail=f"CSV inválido: {e}")

    filas     = len(df)
    productos = df["producto"].unique().tolist()
    fecha_min = str(df["fecha"].min())
    fecha_max = str(df["fecha"].max())

    bg.add_task(_bg_retrain)
    logger.info(f"CSV nuevo: {filas} filas. Reentrenando.")

    return CSVUploadResponse(
        status="ok",
        filas=filas,
        productos=productos,
        rango_fechas={"inicio": fecha_min, "fin": fecha_max},
        mensaje=f"CSV cargado con {filas} registros. Reentrenamiento iniciado.",
    )


# ══════════════════════════════════════════════════════════
#  REPORTE PDF — sin auth
# ══════════════════════════════════════════════════════════

@router.post("/reporte", response_model=ReporteResponse, tags=["Reportes"])
async def crear_reporte(req: ReporteRequest):
    """Genera PDF con gráfico y tabla de KPIs del forecast."""
    pid = req.producto_id.value
    if not modelo_existe(pid):
        raise HTTPException(
            status_code=424,
            detail=f"Modelo no entrenado para '{pid}'",
        )

    clima_datos = obtener_clima_actual()
    resultado   = predecir(
        producto_id=pid,
        meses_futuro=req.meses_futuro.value,
        temp_max_actual=clima_datos["temp_max_absoluta"],
        precip_mm_actual=clima_datos["precip_acumulada_mm"],
    )
    nombre = generar_pdf(resultado, clima_datos)

    return ReporteResponse(
        status="ok",
        archivo=nombre,
        url_descarga=f"/reporte/{nombre}",
    )


@router.get("/reporte/{nombre_archivo}", tags=["Reportes"])
async def descargar_reporte(nombre_archivo: str):
    """Descarga el PDF generado."""
    ruta = REPORTS_DIR / nombre_archivo
    if not ruta.exists():
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    return FileResponse(
        path=str(ruta),
        media_type="application/pdf",
        filename=nombre_archivo,
    )