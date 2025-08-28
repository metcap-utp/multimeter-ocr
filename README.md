# Multimeter OCR

Extrae lecturas numéricas de videos de multímetros usando YOLO + OCR.

## Instalación

```bash
conda env create -f environment.yml
conda activate extract-current
```

## Uso

### Preparación
1. Coloca tus videos en la carpeta `input/` (formatos: .mp4, .avi, .mov)
2. Asegúrate de que el modelo YOLO esté en `models/best.pt`

### Comando básico
```bash
python main.py [--file VIDEO.mp4] [--ocr_engine ENGINE] [--debug]
```

### Parámetros
- `--file`: Video específico dentro de `input/` (opcional, procesa todos si se omite)
- `--ocr_engine`: 
  - `easyocr` (default): Recomendado para displays de 7 segmentos
  - `tesseract`: OCR tradicional
  - `both`: Genera resultados con ambos motores
- `--debug`: Guarda imágenes intermedias en `test/` para análisis

### Ejemplos
```bash
# Procesar video.mp4 con EasyOCR (modo recomendado)
python main.py --file video.mp4 --ocr_engine easyocr

# Comparar ambos motores OCR
python main.py --file video.mp4 --ocr_engine both

# Modo debug: guarda crops, binarizadas y frames con bounding boxes
python main.py --file video.mp4 --ocr_engine easyocr --debug

# Procesar todos los videos de input/ con ambos motores
python main.py --ocr_engine both
```

## Estructura
```
input/     # Videos (.mp4, .avi, .mov)
output/    # CSVs con resultados
test/      # Imágenes debug (con --debug)
models/    # Modelo YOLO (best.pt)
```

## Salida
- **CSVs**: `output/{video}_{motor}.csv` con formato `Time (ms), Reading`
- **Debug** (solo con `--debug`): Carpetas en `test/{video}_{motor}/` con:
  - `crops/`: Detecciones YOLO de pantallas
  - `refined_crops/`: Área recortada del display
  - `bins/`: Imágenes binarizadas enviadas al OCR
  - `frames/`: Videos con bounding boxes dibujados

## Sistema de validación
- Procesa cada 5 frames
- Detecta y confirma saltos grandes válidos (cambios de rango)
- Filtra outliers esporádicos con mediana móvil
- Valida estabilidad temporal
