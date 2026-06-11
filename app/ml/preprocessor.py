# ════════════════════════════════════════════════════════
#  forecastpy_backend/app/ml/preprocessor.py
#
#  Lee el CSV real, construye features y ventanas para LSTM.
#
#  CSV esperado (columnas):
#    fecha        → "YYYY-MM"
#    producto     → "agua_500ml" | "agua_2l" | "agua_5l" | "agua_20l"
#    cantidad     → unidades vendidas (TARGET del LSTM)
#    precio_costo → costo en Gs.
#    precio_venta → precio de venta en Gs.
# ════════════════════════════════════════════════════════
 
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List
 
from app.config import CSV_PATH, MODELS_DIR, LOOKBACK, PRODUCTOS_VALIDOS
 
logger = logging.getLogger(__name__)
 
# Columnas requeridas en el CSV
REQUIRED_COLS = ["fecha", "producto", "cantidad", "precio_costo", "precio_venta"]
 
# Features que entran al LSTM — orden fijo (el scaler usa este orden)
FEATURE_COLS = [
    "cantidad",      # [0] TARGET — columna que predice el modelo
    "precio_venta",  # [1]
    "precio_costo",  # [2]
    "margen_gs",     # [3] precio_venta - precio_costo
    "mes",           # [4] 1–12
    "trimestre",     # [5] 1–4
    "tendencia",     # [6] índice ordinal del mes (0, 1, 2, …)
]
N_FEATURES = len(FEATURE_COLS)  # 7
 
 
# ══════════════════════════════════════════════════════════
#  CARGA Y VALIDACIÓN DEL CSV
# ══════════════════════════════════════════════════════════
 
def cargar_csv(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """
    Lee y valida el CSV. Retorna DataFrame con columna 'fecha_dt'
    (Period mensual de pandas) agregada.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
 
    df = pd.read_csv(csv_path)
    logger.info(f"CSV cargado: {df.shape[0]} filas")
 
    # Validar columnas
    faltantes = set(REQUIRED_COLS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {faltantes}")
 
    # Parsear "YYYY-MM" a período mensual
    df["fecha_dt"] = pd.to_datetime(df["fecha"], format="%Y-%m").dt.to_period("M")
    df = df.sort_values(["producto", "fecha_dt"]).reset_index(drop=True)
    return df
 
 
# ══════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
 
def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas que necesita el LSTM."""
    df = df.copy()
    df["margen_gs"]  = df["precio_venta"] - df["precio_costo"]
    df["mes"]        = df["fecha_dt"].dt.month    # 1–12
    df["trimestre"]  = df["fecha_dt"].dt.quarter  # 1–4
    # Tendencia: índice ordinal dentro de cada producto
    df["tendencia"]  = df.groupby("producto").cumcount()
    return df
 
 
# ══════════════════════════════════════════════════════════
#  PREPARAR DATOS PARA ENTRENAMIENTO
# ══════════════════════════════════════════════════════════
 
def preparar_producto(
    df: pd.DataFrame,
    producto_id: str,
    lookback: int = LOOKBACK,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Para un producto dado:
      1. Filtra sus filas
      2. Normaliza con MinMaxScaler [0,1]
      3. Construye ventanas deslizantes (X, y)
 
    Retorna:
        X      → shape (n, lookback, N_FEATURES)
        y      → shape (n,)  cantidad normalizada del mes siguiente
        scaler → MinMaxScaler ajustado (guardado en .pkl)
    """
    if producto_id not in PRODUCTOS_VALIDOS:
        raise ValueError(f"Producto inválido: {producto_id}")
 
    prod_df = df[df["producto"] == producto_id].copy()
    prod_df = prod_df.sort_values("fecha_dt").reset_index(drop=True)
 
    if len(prod_df) < lookback + 1:
        raise ValueError(
            f"Datos insuficientes para '{producto_id}': "
            f"{len(prod_df)} filas, mínimo {lookback + 1}"
        )
 
    data = prod_df[FEATURE_COLS].values.astype(np.float32)
 
    scaler = MinMaxScaler(feature_range=(0, 1))
    if fit_scaler:
        data_scaled = scaler.fit_transform(data)
        _guardar_scaler(scaler, producto_id)
    else:
        scaler = _cargar_scaler(producto_id)
        data_scaled = scaler.transform(data)
 
    X, y = _crear_ventanas(data_scaled, lookback)
    logger.info(f"[{producto_id}] X:{X.shape} y:{y.shape}")
    return X, y, scaler
 
 
def _crear_ventanas(
    data_scaled: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte serie normalizada en pares (ventana, target).
    X[i] = data[i-lookback : i]  →  shape (lookback, N_FEATURES)
    y[i] = data[i, 0]            →  cantidad del mes siguiente
    """
    X_list, y_list = [], []
    for i in range(lookback, len(data_scaled)):
        X_list.append(data_scaled[i - lookback : i, :])
        y_list.append(data_scaled[i, 0])
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )
 
 
def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    División temporal (no aleatoria).
    Los últimos test_ratio% son el set de test.
    """
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]
 
 
# ══════════════════════════════════════════════════════════
#  PREPARAR VENTANA PARA PREDICCIÓN
# ══════════════════════════════════════════════════════════
 
def preparar_ventana_prediccion(
    producto_id: str,
) -> Tuple[np.ndarray, MinMaxScaler, pd.DataFrame]:
    """
    Prepara la ventana de los últimos LOOKBACK meses del CSV
    para alimentar el modelo en modo predicción.
 
    Retorna:
        ventana  → shape (1, LOOKBACK, N_FEATURES)
        scaler   → MinMaxScaler cargado del .pkl
        prod_df  → DataFrame del producto (para precios recientes)
    """
    df = cargar_csv()
    df = construir_features(df)
 
    prod_df = df[df["producto"] == producto_id].sort_values("fecha_dt")
 
    if len(prod_df) < LOOKBACK:
        raise ValueError(
            f"'{producto_id}' tiene {len(prod_df)} meses, "
            f"se necesitan {LOOKBACK}."
        )
 
    ultimos  = prod_df.tail(LOOKBACK)
    data     = ultimos[FEATURE_COLS].values.astype(np.float32)
    scaler   = _cargar_scaler(producto_id)
    data_sc  = scaler.transform(data)
    ventana  = np.expand_dims(data_sc, axis=0)  # (1, LOOKBACK, N_FEATURES)
 
    return ventana, scaler, prod_df
 
 
# ══════════════════════════════════════════════════════════
#  DESNORMALIZACIÓN
# ══════════════════════════════════════════════════════════
 
def desnormalizar_cantidad(
    valor_norm: float,
    scaler: MinMaxScaler,
) -> float:
    """
    Convierte el valor normalizado [0,1] del LSTM a unidades reales.
    Usa un array dummy con ceros en todas las columnas excepto col 0.
    """
    dummy = np.zeros((1, N_FEATURES), dtype=np.float32)
    dummy[0, 0] = valor_norm
    real = scaler.inverse_transform(dummy)
    return max(0.0, float(real[0, 0]))
 
 
# ══════════════════════════════════════════════════════════
#  PERSISTENCIA DEL SCALER
# ══════════════════════════════════════════════════════════
 
def _guardar_scaler(scaler: MinMaxScaler, producto_id: str) -> None:
    ruta = MODELS_DIR / f"scaler_{producto_id}.pkl"
    joblib.dump(scaler, ruta)
    logger.info(f"Scaler guardado: {ruta}")
 
 
def _cargar_scaler(producto_id: str) -> MinMaxScaler:
    ruta = MODELS_DIR / f"scaler_{producto_id}.pkl"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Scaler no encontrado para '{producto_id}'. "
            f"Ejecuta primero POST /retrain."
        )
    return joblib.load(ruta)
 
 
# ══════════════════════════════════════════════════════════
#  UTILIDADES PARA LA API
# ══════════════════════════════════════════════════════════
 
def obtener_precios_recientes(producto_id: str) -> Dict[str, float]:
    """Retorna precio_venta, precio_costo y margen del último mes del CSV."""
    df = cargar_csv()
    prod_df = df[df["producto"] == producto_id].sort_values("fecha_dt")
    if prod_df.empty:
        raise ValueError(f"Producto '{producto_id}' no encontrado en el CSV.")
    ultimo = prod_df.iloc[-1]
    return {
        "precio_venta": float(ultimo["precio_venta"]),
        "precio_costo": float(ultimo["precio_costo"]),
        "margen_gs":    float(ultimo["precio_venta"] - ultimo["precio_costo"]),
        "ultimo_mes":   str(ultimo["fecha"]),
    }
 
 
def obtener_historico_producto(producto_id: str) -> List[dict]:
    """Retorna la serie histórica completa para el gráfico de línea."""
    df = cargar_csv()
    prod_df = df[df["producto"] == producto_id].sort_values("fecha_dt")
    return prod_df[["fecha", "cantidad", "precio_venta", "precio_costo"]].to_dict("records")