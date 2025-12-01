# pigs_trackerDIA/config.py

import os

# --- 1. CONFIGURACIÓN DE RUTAS DINÁMICAS ---

# Obtenemos la ruta absoluta de ESTE archivo (config.py)
# En tu caso: .../pigs_trackerDIA/config.py
CURRENT_FILE = os.path.abspath(__file__)

# Como config.py está en la raíz del proyecto, el directorio base es simplemente su carpeta contenedora.
# BASE_DIR será: .../pigs_trackerDIA
BASE_DIR = os.path.dirname(CURRENT_FILE)

# --- DEBUG (Opcional: Imprime esto si sigue fallando para ver dónde busca) ---
print(f"📂 Raíz del proyecto detectada: {BASE_DIR}")
print(f"🔍 Buscando modelo en: {os.path.join(BASE_DIR, 'models', 'best.pt')}")

# --- Definición de Rutas Relativas ---

# Modelo: Ahora buscará en .../pigs_trackerDIA/models/best.pt
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# Video: Buscará en .../pigs_trackerDIA/data/video_dia.avi
VIDEO_PATH = os.path.join(BASE_DIR, "data", "video_dia.avi")

# Salida: .../pigs_trackerDIA/output
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Aseguramos que la carpeta de salida exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Galería
GALLERY_PATH = os.path.join(OUTPUT_DIR, "gallery.npz")
# Sample frame (opcional)
SAMPLE_FRAME_PATH = os.path.join(BASE_DIR, "data", "sample_frame.jpg")

# --- Configuración Global ---
TRACKER_TYPE = "bytetrack.yaml" 
SAVE_VIDEO = True 

# --- Parámetros de Filtrado y Detección ---
MIN_AREA = 15000 
MAX_ASPECT_RATIO = 4.0 
MIN_ASPECT_RATIO = 0.4 
IOU_MERGE = 0.7 

# --- Parámetros de Seguimiento (Kalman / Pérdida) ---
FPS_GUESS = 30.0 
MAX_MISSED_SECONDS = 13.0 
MAX_SPEED_PIXELS_PER_FRAME = 80.0 
ID_LOST_THRESHOLD = 3.0 

# --- Parámetros de ReID y Matching ---
WARMUP_SECONDS = 5.0 
REID_THRESHOLD = 0.72 
ALPHA = 0.6 
BETA = 0.4 
MOTION_GATE = 0.25 
OCCLUSION_IOU = 0.45 
EMBED_UPDATE_MOMENTUM = 0.6
'''
import os

# --- Rutas de Archivos y Configuración Global ---
MODEL_PATH = r"D:\puerquitos_Marco\pigs_trackerDIA\models\best.pt"
VIDEO_PATH = r"D:\puerquitos_Marco\video_dia.avi"
OUTPUT_DIR = r"D:\puerquitos_Marco\Resultados\Modelo_dia_reid_kalman_finetuning2"
GALLERY_PATH = os.path.join(OUTPUT_DIR, "gallery.npz")
SAMPLE_FRAME_PATH = "/mnt/data/03c0713d-2685-4fde-b472-af9b3cd0d50f.jpg"
TRACKER_TYPE = "bytetrack.yaml" # Usado en la llamada a model.track (YOLO)
SAVE_VIDEO = True # Bandera para guardar el video de salida

# --- Parámetros de Filtrado y Detección ---
MIN_AREA = 15000 # Área mínima del bounding box
MAX_ASPECT_RATIO = 4.0 # Relación de aspecto máxima (ancho/alto)
MIN_ASPECT_RATIO = 0.4 # Relación de aspecto mínima
IOU_MERGE = 0.7 # Umbral IOU para fusionar cajas muy solapadas entre mas alto mas dificl que se dupliquen y entre mas bajo mas facil que se dupliquen

# --- Parámetros de Seguimiento (Kalman / Pérdida) ---
FPS_GUESS = 30.0 # Estimación inicial de FPS
MAX_MISSED_SECONDS = 13.0 # Tolerancia de pérdida en segundos
MAX_SPEED_PIXELS_PER_FRAME = 80.0 # Límite de velocidad (pixeles/frame) para evitar "teleport"
ID_LOST_THRESHOLD = 3.0 # Tiempo en segundos para enviar alerta de "perdido"

# --- Parámetros de ReID y Matching ---
WARMUP_SECONDS = 5.0 # Tiempo para construir la galería inicial
REID_THRESHOLD = 0.72 # Umbral de similitud coseno para identidad
ALPHA = 0.6 # Peso de la apariencia (ReID) en el costo combinado
BETA = 0.4 # Peso del movimiento (Kalman) en el costo combinado
MOTION_GATE = 0.25 # Distancia normalizada máxima permitida
OCCLUSION_IOU = 0.45 # Umbral de IOU para considerar solapamiento/oclusión
EMBED_UPDATE_MOMENTUM = 0.6 # Momentum para suavizar el prototipo ReID del track
'''