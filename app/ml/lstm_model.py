# ════════════════════════════════════════════════════════
#  forecastpy_backend/app/ml/lstm_model.py
#
#  Arquitectura LSTM, entrenamiento y predicción multi-paso.
#
#  ARQUITECTURA:
#    Input  (LOOKBACK=12, N_FEATURES=7)
#    LSTM_1  64 unidades, return_sequences=True
#    Dropout 0.2
#    LSTM_2  32 unidades, return_sequences=False
#    Dropout 0.2
#    Dense   16 relu
#    Dense   1  linear  →  cantidad normalizada
# ════════════════════════════════════════════════════════
 
import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
 
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
 
from app.config import (
    MODELS_DIR, LOOKBACK, EPOCHS, BATCH_SIZE,
    LSTM_UNITS_1, LSTM_UNITS_2, PRODUCTOS_VALIDOS,
    TEMP_LOW, TEMP_HIGH, DROUGHT_MM,
    CLIMATE_FACTOR_MID, CLIMATE_FACTOR_MAX,
)
from app.ml.preprocessor import (
    N_FEATURES, cargar_csv, construir_features,
    preparar_producto, preparar_ventana_prediccion,
    desnormalizar_cantidad, obtener_precios_recientes,
    split_train_test,
)
 
logger = logging.getLogger(__name__)
 
tf.random.set_seed(42)
np.random.seed(42)
 
 
# ══════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL MODELO
# ══════════════════════════════════════════════════════════
 
def construir_modelo(
    lookback: int = LOOKBACK,
    n_features: int = N_FEATURES,
) -> Sequential:
    """
    Construye y compila la red LSTM.
    Dos capas LSTM apiladas + Dropout + Dense de salida.
    Loss: MAE (más robusto que MSE con pocos datos mensuales).
    """
    model = Sequential([
        Input(shape=(lookback, n_features)),
 
        LSTM(
            units=LSTM_UNITS_1,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        ),
        Dropout(0.2),
 
        LSTM(
            units=LSTM_UNITS_2,
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        ),
        Dropout(0.2),
 
        Dense(16, activation="relu"),
        Dense(1,  activation="linear"),
    ])
 
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mae",
        metrics=["mse"],
    )
    logger.info(f"Modelo construido: {model.count_params()} parámetros")
    return model
 
 
# ══════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════════════════════
 
def entrenar_producto(producto_id: str) -> Dict:
    """
    Entrena el modelo LSTM para un producto.
    Guarda: models_saved/model_{producto_id}.keras
            models_saved/scaler_{producto_id}.pkl
    Retorna dict con métricas de entrenamiento.
    """
    logger.info(f"=== Entrenando: {producto_id} ===")
 
    # 1. Preparar datos
    df = cargar_csv()
    df = construir_features(df)
    X, y, scaler = preparar_producto(
        df, producto_id, lookback=LOOKBACK, fit_scaler=True)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
 
    logger.info(f"[{producto_id}] train:{X_train.shape} test:{X_test.shape}")
 
    # 2. Modelo
    model = construir_modelo()
 
    # 3. Callbacks
    ruta_ckpt = str(MODELS_DIR / f"model_{producto_id}.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=ruta_ckpt,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
 
    # 4. Entrenar
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,   # datos temporales: NO barajar
        verbose=1,
    )
 
    # 5. Evaluar
    test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
    test_rmse   = float(np.sqrt(test_mse))
    epocas_real = len(history.history["loss"])
    best_val    = float(min(history.history["val_loss"]))
 
    logger.info(
        f"[{producto_id}] mae={test_loss:.4f} rmse={test_rmse:.4f} "
        f"épocas={epocas_real}"
    )
    return {
        "producto_id":   producto_id,
        "epocas":        epocas_real,
        "test_mae":      round(test_loss, 4),
        "test_rmse":     round(test_rmse, 4),
        "best_val_loss": round(best_val, 4),
        "modelo_path":   ruta_ckpt,
        "train_samples": int(X_train.shape[0]),
        "test_samples":  int(X_test.shape[0]),
    }
 
 
def entrenar_todos() -> List[Dict]:
    """Entrena los 4 modelos de manera secuencial."""
    resultados = []
    for pid in PRODUCTOS_VALIDOS:
        try:
            resultados.append(entrenar_producto(pid))
        except Exception as e:
            logger.error(f"Error entrenando '{pid}': {e}")
            resultados.append({"producto_id": pid, "error": str(e)})
    return resultados
 
 
# ══════════════════════════════════════════════════════════
#  CARGA DEL MODELO
# ══════════════════════════════════════════════════════════
 
def cargar_modelo(producto_id: str) -> Sequential:
    """Carga el .keras desde disco. Lanza error si no existe."""
    ruta = MODELS_DIR / f"model_{producto_id}.keras"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado para '{producto_id}'. "
            f"Ejecuta primero POST /retrain."
        )
    return load_model(str(ruta))
 
 
def modelo_existe(producto_id: str) -> bool:
    """True si el .keras ya fue entrenado."""
    return (MODELS_DIR / f"model_{producto_id}.keras").exists()
 
 
# ══════════════════════════════════════════════════════════
#  PREDICCIÓN MULTI-PASO
# ══════════════════════════════════════════════════════════
 
def predecir(
    producto_id: str,
    meses_futuro: int = 1,
    temp_max_actual: float = 30.0,
    precip_mm_actual: float = 50.0,
) -> Dict:
    """
    Predicción iterativa para 1 o 6 meses.
 
    Estrategia multi-paso:
      - Predice mes N+1 con la ventana actual.
      - Usa esa predicción como entrada para predecir N+2.
      - Repite hasta completar meses_futuro.
      - Aplica ajuste climático al final (post-proceso).
 
    Retorna dict con cantidades, KPIs financieros e histórico.
    """
    if meses_futuro not in (1, 6):
        raise ValueError("meses_futuro debe ser 1 o 6")
 
    # 1. Cargar modelo y preparar ventana inicial
    model   = cargar_modelo(producto_id)
    ventana, scaler, prod_df = preparar_ventana_prediccion(producto_id)
    precios = obtener_precios_recientes(producto_id)
 
    precio_venta = precios["precio_venta"]
    precio_costo = precios["precio_costo"]
 
    # 2. Calcular ajuste climático
    ajuste = _calcular_ajuste_clima(temp_max_actual, precip_mm_actual)
    logger.info(f"[{producto_id}] ajuste_clima={ajuste:.2%} temp={temp_max_actual}°C")
 
    # 3. Predicción iterativa
    ventana_actual = ventana.copy()
    cantidades_norm: List[float] = []
 
    for _ in range(meses_futuro):
        pred_norm = float(model.predict(ventana_actual, verbose=0)[0, 0])
        cantidades_norm.append(pred_norm)
 
        # Construir nuevo timestep con la cantidad predicha
        ultimo_ts    = ventana_actual[0, -1, :].copy()
        nuevo_ts     = ultimo_ts.copy()
        nuevo_ts[0]  = pred_norm  # col 0 = cantidad predicha
 
        # Avanzar mes y tendencia (aproximación normalizada)
        mes_norm     = ultimo_ts[4]
        mes_real     = round(mes_norm * 11 + 1)
        nuevo_mes    = (mes_real % 12) + 1
        nuevo_ts[4]  = (nuevo_mes - 1) / 11
 
        trim_real    = (nuevo_mes - 1) // 3 + 1
        nuevo_ts[5]  = (trim_real - 1) / 3
        nuevo_ts[6]  = ultimo_ts[6] + (1 / 40)  # tendencia avanza
 
        # Deslizar la ventana
        ventana_actual = np.concatenate([
            ventana_actual[:, 1:, :],
            nuevo_ts.reshape(1, 1, N_FEATURES),
        ], axis=1)
 
    # 4. Desnormalizar
    cantidades_raw = [
        desnormalizar_cantidad(v, scaler) for v in cantidades_norm
    ]
 
    # 5. Aplicar ajuste climático
    cantidades_pred = [round(c * (1 + ajuste)) for c in cantidades_raw]
 
    # 6. Fechas de los meses predichos
    ultimo_periodo = prod_df["fecha_dt"].iloc[-1]
    meses_labels   = [str(ultimo_periodo + i) for i in range(1, meses_futuro + 1)]
 
    # 7. KPIs financieros
    ingresos      = [round(c * precio_venta) for c in cantidades_pred]
    costos_total  = [round(c * precio_costo) for c in cantidades_pred]
    margenes      = [i - c for i, c in zip(ingresos, costos_total)]
    rentabilidades = [
        round((m / i) * 100, 2) if i > 0 else 0.0
        for m, i in zip(margenes, ingresos)
    ]
 
    # 8. Histórico para el gráfico
    historico = prod_df[["fecha", "cantidad"]].to_dict("records")
 
    return {
        "producto_id":      producto_id,
        "meses":            meses_labels,
        "cantidades":       cantidades_pred,
        "costos_unitarios": [int(precio_costo)] * meses_futuro,
        "precios_venta":    [int(precio_venta)] * meses_futuro,
        "ingresos":         ingresos,
        "costos_total":     costos_total,
        "margenes":         margenes,
        "rentabilidades":   rentabilidades,
        "ajuste_clima_pct": round(ajuste * 100, 1),
        "temp_max_usada":   temp_max_actual,
        "precip_mm_usada":  precip_mm_actual,
        "historico":        historico,
        "meses_futuro":     meses_futuro,
    }
 
 
# ══════════════════════════════════════════════════════════
#  AJUSTE CLIMÁTICO — Reglas Paraguay Cordillera
# ══════════════════════════════════════════════════════════
 
def _calcular_ajuste_clima(temp_max: float, precip_mm: float) -> float:
    """
    Retorna la fracción de ajuste de demanda según clima.
 
    Reglas:
      precip < DROUGHT_MM (5mm)  → sequía → +20%
      temp ≥ 38°C                → +20%
      35°C ≤ temp < 38°C         → interpolación lineal 10%-20%
      temp < 35°C                → 0%
    """
    if precip_mm < DROUGHT_MM:
        logger.info(f"Sequía ({precip_mm}mm) → ajuste máximo {CLIMATE_FACTOR_MAX:.0%}")
        return CLIMATE_FACTOR_MAX
 
    if temp_max >= TEMP_HIGH:
        return CLIMATE_FACTOR_MAX
    elif temp_max >= TEMP_LOW:
        fraccion = (temp_max - TEMP_LOW) / (TEMP_HIGH - TEMP_LOW)
        return CLIMATE_FACTOR_MID + fraccion * (CLIMATE_FACTOR_MAX - CLIMATE_FACTOR_MID)
    else:
        return 0.0