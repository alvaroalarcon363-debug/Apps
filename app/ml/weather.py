# ════════════════════════════════════════════════════════
#  forecastpy_backend/app/ml/weather.py
#
#  CORRECCIÓN: ahora muestra temperatura REAL de hoy,
#  no el máximo del mes.
#
#  Usa DOS endpoints de Open-Meteo:
#    1. forecast API  → temperatura de HOY (en tiempo real)
#    2. archive API   → historial del mes para el ajuste
#       de demanda (acumulado mensual)
#
#  El chip en Flutter mostrará la temp de hoy.
#  El ajuste de demanda LSTM usa el promedio del mes.
# ════════════════════════════════════════════════════════
 
import logging
import requests
import requests_cache
from datetime import date, timedelta
from typing import Dict
 
from app.config import WEATHER_LAT, WEATHER_LON, WEATHER_TZ, DROUGHT_MM
 
logger = logging.getLogger(__name__)
 
# Caché de 30 minutos para no sobrecargar Open-Meteo
# (reducido de 1h a 30min para que la temp de hoy se actualice más seguido)
requests_cache.install_cache(
    "openmeteo_cache",
    expire_after=1800,   # 30 minutos
    backend="memory",
)
 
 
# ══════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════
 
def obtener_clima_actual() -> Dict:
    """
    Retorna el resumen climático combinando:
      - Temperatura de HOY (forecast API) → para el chip del AppBar
      - Historial del mes (archive API)   → para ajuste de demanda LSTM
 
    En caso de error devuelve defaults conservadores.
    """
    hoy   = date.today()
    ayer  = hoy - timedelta(days=1)
    inicio_mes = hoy.replace(day=1)
 
    # Si es el primer día del mes, usamos ayer como rango
    if inicio_mes == hoy:
        inicio_mes = ayer
 
    # ── 1. Temperatura de HOY desde forecast API ──────────
    temp_hoy_max  = None
    temp_hoy_min  = None
    precip_hoy    = None
 
    try:
        datos_hoy = _consultar_forecast(hoy)
        temp_hoy_max = datos_hoy.get("temp_max_hoy")
        temp_hoy_min = datos_hoy.get("temp_min_hoy")
        precip_hoy   = datos_hoy.get("precip_hoy", 0.0)
        logger.info(
            f"Temperatura hoy: max={temp_hoy_max}°C "
            f"min={temp_hoy_min}°C precip={precip_hoy}mm"
        )
    except Exception as e:
        logger.warning(f"Forecast API falló: {e}. Intentando archive.")
 
    # ── 2. Historial del mes desde archive API ────────────
    temp_max_mes  = None
    precip_mes    = None
    humedad_mes   = None
 
    try:
        # Solo consultamos hasta ayer (archive no tiene hoy aún)
        if inicio_mes <= ayer:
            datos_mes = _consultar_archive(inicio_mes.isoformat(), ayer.isoformat())
            temps_mes  = [v for v in datos_mes.get("temperature_2m_max", []) if v is not None]
            precips_mes = [v for v in datos_mes.get("precipitation_sum", []) if v is not None]
            humeds_mes  = [v for v in datos_mes.get("relative_humidity_2m_mean", []) if v is not None]
 
            temp_max_mes = round(max(temps_mes), 1)   if temps_mes  else None
            precip_mes   = round(sum(precips_mes), 1) if precips_mes else 0.0
            humedad_mes  = round(sum(humeds_mes) / len(humeds_mes), 1) if humeds_mes else 70.0
    except Exception as e:
        logger.warning(f"Archive API falló: {e}")
 
    # ── 3. Combinar: priorizar datos de HOY ───────────────
    # Para el chip del AppBar usamos la temp de HOY
    # Para el ajuste de demanda usamos el máximo del mes
    temp_display  = temp_hoy_max  if temp_hoy_max  is not None else (temp_max_mes or 25.0)
    temp_min_disp = temp_hoy_min  if temp_hoy_min  is not None else 15.0
    temp_max_abs  = temp_max_mes  if temp_max_mes  is not None else temp_display
    precip_total  = precip_mes    if precip_mes    is not None else (precip_hoy or 0.0)
    humedad_prom  = humedad_mes   if humedad_mes   is not None else 70.0
 
    # Si no hay datos de ninguna fuente, usar defaults
    if temp_display is None:
        logger.warning("Ninguna API respondió. Usando defaults.")
        return _defaults()
 
    # ── 4. Calcular nivel de alerta y ajuste ──────────────
    # El ajuste de demanda se basa en temp_max_abs (máximo del mes)
    # porque representa el comportamiento general del período
    es_sequia = precip_total < DROUGHT_MM
 
    if temp_max_abs >= 38.0 or es_sequia:
        nivel  = "critico"
        ajuste = 20.0
    elif temp_max_abs >= 35.0:
        nivel    = "alto"
        fraccion = (temp_max_abs - 35.0) / 3.0
        ajuste   = round(10.0 + fraccion * 10.0, 1)
    else:
        nivel  = "normal"
        ajuste = 0.0
 
    return {
        # Temperatura de HOY → para el chip del AppBar en Flutter
        "temp_max_promedio":    round(temp_display, 1),
        "temp_max_absoluta":    round(temp_display, 1),  # HOY
        "temp_max_mes":         round(temp_max_abs, 1),  # máximo del mes
        "temp_min_promedio":    round(temp_min_disp, 1),
        # Precipitación acumulada del mes
        "precip_acumulada_mm":  round(precip_total, 1),
        "humedad_promedio_pct": round(humedad_prom, 1),
        "dias_medidos":         (ayer - inicio_mes).days + 1 if inicio_mes <= ayer else 0,
        "es_sequia":            es_sequia,
        "nivel_alerta":         nivel,
        "ajuste_demanda_pct":   ajuste,
        "ubicacion":            "Caacupé, Cordillera, Paraguay",
        "coordenadas":          {"lat": WEATHER_LAT, "lon": WEATHER_LON},
        "periodo":              str(hoy),
        "fuente":               "Open-Meteo forecast + archive",
    }
 
 
# ══════════════════════════════════════════════════════════
#  FORECAST API — temperatura de HOY
# ══════════════════════════════════════════════════════════
 
def _consultar_forecast(hoy: date) -> Dict:
    """
    Consulta el endpoint forecast de Open-Meteo para obtener
    la temperatura máxima y mínima de HOY.
 
    Este endpoint devuelve datos del día actual sin necesidad
    de esperar a que se confirmen en el archivo histórico.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":              WEATHER_LAT,
        "longitude":             WEATHER_LON,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
        ],
        "timezone":              WEATHER_TZ,
        "temperature_unit":      "celsius",
        "precipitation_unit":    "mm",
        "forecast_days":         1,   # solo hoy
    }
 
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
 
    if "daily" not in data:
        raise ValueError(f"Respuesta inesperada del forecast: {data}")
 
    daily = data["daily"]
 
    # Extraer el primer (y único) día
    temps_max = [v for v in daily.get("temperature_2m_max", []) if v is not None]
    temps_min = [v for v in daily.get("temperature_2m_min", []) if v is not None]
    precips   = [v for v in daily.get("precipitation_sum",  []) if v is not None]
 
    logger.info(f"Forecast OK: hoy={hoy}")
 
    return {
        "temp_max_hoy": round(temps_max[0], 1) if temps_max else None,
        "temp_min_hoy": round(temps_min[0], 1) if temps_min else None,
        "precip_hoy":   round(precips[0],   1) if precips   else 0.0,
    }
 
 
# ══════════════════════════════════════════════════════════
#  ARCHIVE API — historial del mes para ajuste de demanda
# ══════════════════════════════════════════════════════════
 
def _consultar_archive(fecha_inicio: str, fecha_fin: str) -> Dict:
    """
    Consulta datos históricos confirmados de Open-Meteo.
    Solo tiene datos hasta ayer (no incluye el día actual).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":              WEATHER_LAT,
        "longitude":             WEATHER_LON,
        "start_date":            fecha_inicio,
        "end_date":              fecha_fin,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "relative_humidity_2m_mean",
        ],
        "timezone":              WEATHER_TZ,
        "temperature_unit":      "celsius",
        "precipitation_unit":    "mm",
    }
 
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
 
    if "daily" not in data:
        raise ValueError(f"Respuesta inesperada del archive: {data}")
 
    logger.info(f"Archive OK: {fecha_inicio}→{fecha_fin}")
    return data["daily"]
 
 
# ══════════════════════════════════════════════════════════
#  DEFAULTS — cuando ambas APIs fallan
# ══════════════════════════════════════════════════════════
 
def _defaults() -> Dict:
    """
    Valores por defecto cuando Open-Meteo no responde.
    Representa un día templado típico de mayo en Paraguay
    (otoño, temperaturas moderadas).
    """
    return {
        "temp_max_promedio":    20.0,   # mayo es otoño en PY
        "temp_max_absoluta":    20.0,
        "temp_max_mes":         22.0,
        "temp_min_promedio":    10.0,
        "precip_acumulada_mm":  30.0,
        "humedad_promedio_pct": 65.0,
        "dias_medidos":         0,
        "es_sequia":            False,
        "nivel_alerta":         "normal",
        "ajuste_demanda_pct":   0.0,
        "ubicacion":            "Caacupé, Cordillera, Paraguay",
        "coordenadas":          {"lat": WEATHER_LAT, "lon": WEATHER_LON},
        "periodo":              str(date.today()),
        "fuente":               "default (API no disponible)",
    }
 
 
# ══════════════════════════════════════════════════════════
#  FUNCIÓN AUXILIAR para otros módulos
# ══════════════════════════════════════════════════════════
 
def obtener_clima_mes(anio: int, mes: int) -> Dict:
    """Retorna el resumen climático de un mes específico pasado."""
    import calendar
    _, ultimo_dia = calendar.monthrange(anio, mes)
    inicio = date(anio, mes, 1)
    fin    = min(date(anio, mes, ultimo_dia), date.today() - timedelta(days=1))
 
    try:
        datos = _consultar_archive(inicio.isoformat(), fin.isoformat())
        temps = [v for v in datos.get("temperature_2m_max", []) if v is not None]
        precs = [v for v in datos.get("precipitation_sum",  []) if v is not None]
        hums  = [v for v in datos.get("relative_humidity_2m_mean", []) if v is not None]
 
        if not temps:
            return _defaults()
 
        temp_max_abs = round(max(temps), 1)
        precip_total = round(sum(precs), 1) if precs else 0.0
        humedad_prom = round(sum(hums) / len(hums), 1) if hums else 70.0
        es_sequia    = precip_total < DROUGHT_MM
 
        if temp_max_abs >= 38.0 or es_sequia:
            nivel = "critico"; ajuste = 20.0
        elif temp_max_abs >= 35.0:
            nivel = "alto"
            ajuste = round(10.0 + ((temp_max_abs - 35.0) / 3.0) * 10.0, 1)
        else:
            nivel = "normal"; ajuste = 0.0
 
        return {
            "temp_max_promedio":    round(sum(temps) / len(temps), 1),
            "temp_max_absoluta":    temp_max_abs,
            "temp_max_mes":         temp_max_abs,
            "temp_min_promedio":    15.0,
            "precip_acumulada_mm":  precip_total,
            "humedad_promedio_pct": humedad_prom,
            "dias_medidos":         len(temps),
            "es_sequia":            es_sequia,
            "nivel_alerta":         nivel,
            "ajuste_demanda_pct":   ajuste,
            "ubicacion":            "Caacupé, Cordillera, Paraguay",
            "coordenadas":          {"lat": WEATHER_LAT, "lon": WEATHER_LON},
            "periodo":              str(inicio),
        }
    except Exception as e:
        logger.warning(f"Error clima {anio}-{mes:02d}: {e}")
        return _defaults()