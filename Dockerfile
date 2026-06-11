FROM python:3.11-slim
# Imagen base: Python 3.11 liviana (sin cosas extras de desktop)

WORKDIR /app
# Todas las operaciones se hacen desde /app dentro del contenedor

RUN apt-get update && apt-get install -y \
    libgomp1 \
    # Necesario para TensorFlow (OpenMP)
    gcc \
    # Compilador C necesario para algunas librerías Python
    && rm -rf /var/lib/apt/lists/*
    # Limpiar caché de apt para reducir el tamaño de la imagen

COPY requirements.txt .
# Copiar solo requirements.txt primero (optimización de caché)

RUN pip install --no-cache-dir -r requirements.txt
# Instalar dependencias (esto tarda varios minutos la primera vez)
# --no-cache-dir reduce el tamaño final de la imagen

COPY . .
# Copiar todo el código fuente al contenedor

RUN mkdir -p /data/models_saved /data/reports
# Crear las carpetas que va a usar el backend
# Las montamos en /data para que el volumen de Railway las persista

EXPOSE 8000
# Indicar que la app escucha en el puerto 8000

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     # 1 solo worker: TensorFlow no es thread-safe en multi-proceso
     "--timeout-keep-alive", "120"]
     # 120 segundos de keep-alive para requests largos (entrenamiento)