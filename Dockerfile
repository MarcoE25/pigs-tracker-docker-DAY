# 1. Imagen base
FROM python:3.9-slim

# 2. Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Instalamos librerías del sistema
# CORRECCIÓN: Cambiamos 'libgl1-mesa-glx' por 'libgl1'
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Carpeta de trabajo
WORKDIR /app

# 5. Copiamos requerimientos e instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiamos el código
COPY . .

# 7. Carpetas necesarias
RUN mkdir -p output data models

# 8. Comando de inicio
CMD ["python", "main.py"]