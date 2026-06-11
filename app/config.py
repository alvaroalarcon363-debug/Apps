# ════════════════════════════════════════════════════════
#  forecastpy_backend/app/config.py
#
#  Configuración centralizada. Lee el .env una sola vez.
#  TODOS los módulos importan desde aquí.
# ════════════════════════════════════════════════════════
 
import os
from pathlib import Path
from dotenv import load_dotenv
 
# Raíz del proyecto (un nivel arriba de /app)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
 
# ── Servidor ─────────────────────────────────────────────
HOST:  str  = os.getenv("HOST", "0.0.0.0")
PORT:  int  = int(os.getenv("PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
 
# ── JWT ──────────────────────────────────────────────────
SECRET_KEY:      str = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
ALGORITHM:       str = os.getenv("ALGORITHM", "HS256")
TOKEN_EXPIRE_MIN: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
 
# ── Rutas absolutas ───────────────────────────────────────
CSV_PATH:   Path = BASE_DIR / os.getenv("CSV_PATH",   "data/ventas_aguasanjose_2023_2026.csv")
MODELS_DIR: Path = BASE_DIR / os.getenv("MODELS_DIR", "models_saved")
REPORTS_DIR:Path = BASE_DIR / os.getenv("REPORTS_DIR","reports")
 
# Crear carpetas si no existen al importar
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
 
# ── Open-Meteo — Cordillera, Paraguay ────────────────────
WEATHER_LAT: float = float(os.getenv("WEATHER_LAT", "-25.3867"))
WEATHER_LON: float = float(os.getenv("WEATHER_LON", "-57.1417"))
WEATHER_TZ:  str   = os.getenv("WEATHER_TIMEZONE", "America/Asuncion")
 
# ── LSTM Hiperparámetros ──────────────────────────────────
LOOKBACK:     int = int(os.getenv("LOOKBACK", "12"))
EPOCHS:       int = int(os.getenv("EPOCHS", "150"))
BATCH_SIZE:   int = int(os.getenv("BATCH_SIZE", "8"))
LSTM_UNITS_1: int = int(os.getenv("LSTM_UNITS_1", "64"))
LSTM_UNITS_2: int = int(os.getenv("LSTM_UNITS_2", "32"))
 
# ── Reglas climáticas Paraguay ────────────────────────────
TEMP_LOW:   float = float(os.getenv("TEMP_THRESHOLD_LOW",  "35.0"))
TEMP_HIGH:  float = float(os.getenv("TEMP_THRESHOLD_HIGH", "38.0"))
DROUGHT_MM: float = float(os.getenv("DROUGHT_PRECIP_MM",   "5.0"))
 
CLIMATE_FACTOR_MID: float = 0.10   # +10% entre 35-38 °C
CLIMATE_FACTOR_MAX: float = 0.20   # +20% sobre 38 °C o sequía
 
# ── Productos válidos (deben existir en el CSV) ───────────
PRODUCTOS_VALIDOS: list[str] = [
    "agua_500ml",
    "agua_2l",
    "agua_5l",
    "agua_20l",
]