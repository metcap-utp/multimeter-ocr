import os
import random
from ultralytics import YOLO

# Carga el modelo entrenado
model = YOLO("runs/detect/train/weights/best.pt")

# Carpeta de validación
val_dir = "dataset/images/val"
images = [
    f for f in os.listdir(val_dir) if f.endswith((".jpg", ".jpeg", ".png"))
]

# Escoge una imagen al azar
sample = random.choice(images)
image_path = os.path.join(val_dir, sample)

print(f"Probando con imagen: {image_path}")

# Ejecuta la predicción y guarda el resultado
model.predict(image_path, save=True, conf=0.25)

# Los resultados se guardan en runs/detect/predict/
