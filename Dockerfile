## Dockerfile para empaquetar el servicio FastAPI y el modelo
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para algunas wheels (pandas/xgboost)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar sólo requirements primero para aprovechar cache de capas
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el código y modelos
COPY . /app

# Puerto expuesto por la app
EXPOSE 8000

# Comando por defecto para ejecutar el servidor
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
