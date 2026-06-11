# ════════════════════════════════════════════════════════
#  forecastpy_backend/app/schemas/schemas.py
#
#  Modelos Pydantic v2 para validación de request/response.
# ════════════════════════════════════════════════════════
 
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
 
 
# ── Enums ─────────────────────────────────────────────────
 
class ProductoID(str, Enum):
    agua_500ml = "agua_500ml"
    agua_2l    = "agua_2l"
    agua_5l    = "agua_5l"
    agua_20l   = "agua_20l"
 
class MesesFuturo(int, Enum):
    uno  = 1
    seis = 6
 
 
# ══════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════
 
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4)
 
class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str
 
 
# ══════════════════════════════════════════════════════════
#  PREDICCIÓN
# ══════════════════════════════════════════════════════════
 
class PredictRequest(BaseModel):
    producto_id:        ProductoID
    meses_futuro:       MesesFuturo  = MesesFuturo.uno
    temp_max_override:  Optional[float] = Field(None, ge=-10, le=55)
    precip_mm_override: Optional[float] = Field(None, ge=0, le=1000)
 
class PuntoHistorico(BaseModel):
    fecha:    str
    cantidad: int
 
class PredictResponse(BaseModel):
    producto_id:      str
    meses:            List[str]
    cantidades:       List[int]
    costos_unitarios: List[int]
    precios_venta:    List[int]
    ingresos:         List[int]
    costos_total:     List[int]
    margenes:         List[int]
    rentabilidades:   List[float]
    ajuste_clima_pct: float
    temp_max_usada:   float
    precip_mm_usada:  float
    historico:        List[PuntoHistorico]
    meses_futuro:     int
 
 
# ══════════════════════════════════════════════════════════
#  CLIMA
# ══════════════════════════════════════════════════════════
 
class WeatherResponse(BaseModel):
    temp_max_promedio:    float
    temp_max_absoluta:    float
    temp_min_promedio:    float
    precip_acumulada_mm:  float
    humedad_promedio_pct: float
    dias_medidos:         int
    es_sequia:            bool
    nivel_alerta:         str
    ajuste_demanda_pct:   float
    ubicacion:            str
    coordenadas:          Dict[str, float]
    periodo:              str
 
 
# ══════════════════════════════════════════════════════════
#  RETRAIN / CSV
# ══════════════════════════════════════════════════════════
 
class RetrainResponse(BaseModel):
    status:     str
    resultados: List[Dict[str, Any]]
 
class CSVUploadResponse(BaseModel):
    status:       str
    filas:        int
    productos:    List[str]
    rango_fechas: Dict[str, str]
    mensaje:      str
 
 
# ══════════════════════════════════════════════════════════
#  REPORTES
# ══════════════════════════════════════════════════════════
 
class ReporteRequest(BaseModel):
    producto_id:  ProductoID
    meses_futuro: MesesFuturo = MesesFuturo.uno
 
class ReporteResponse(BaseModel):
    status:       str
    archivo:      str
    url_descarga: str
 
 
# ══════════════════════════════════════════════════════════
#  PRODUCTOS
# ══════════════════════════════════════════════════════════
 
class ProductoInfo(BaseModel):
    id:           str
    nombre:       str
    volumen:      str
    precio_venta: float
    precio_costo: float
    margen_gs:    float
    rentabilidad: float
    ultimo_mes:   str
    modelo_listo: bool
 
class ProductosResponse(BaseModel):
    productos: List[ProductoInfo]
 
 
# ══════════════════════════════════════════════════════════
#  HISTORIAL
# ══════════════════════════════════════════════════════════
 
class HistoricoResponse(BaseModel):
    producto_id: str
    registros:   List[Dict[str, Any]]