FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para OpenCV y otras librerías
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar el archivo de requerimientos e instalar las dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código y los modelos
COPY . .

# Exponer los puertos según la configuración de settings.json (HTTP, gRPC, Metrics)
EXPOSE 8080 8081 8082

# Comando para iniciar mlserver con el archivo de configuración
CMD ["mlserver", "start", "--settings", "settings.json"]
