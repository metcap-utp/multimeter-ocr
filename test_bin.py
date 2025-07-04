import cv2
import pytesseract
import re

ruta = "test/input/crops/crop_081365.jpg"  # Ajusta esto según corresponda
img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"No se pudo cargar la imagen: {ruta}")
    exit(1)

texto = pytesseract.image_to_string(img, config="--psm 7")
texto = texto.strip().replace(" ", "").replace("\n", "")
print("Texto extraído:", texto)
match = re.search(r"\d+\.\d+", texto)
print("Lectura detectada:", match.group() if match else "Nada")
