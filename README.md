# Multimeter OCR

## Instalación

```bash
conda env create -f environment.yml
conda activate extract-current
```

## Preparación

- Videos en `input/` (`.mp4`, `.avi`, `.mov`).
- Modelos: `models/screen.pt` y `models/digit_detection.pt`.

## Ejecutar `main.py`

```bash
python main.py RUTA_AL_VIDEO.mp4 \
  [--screen-model models/screen.pt] \
  [--digits-model models/digit_detection.pt]
```

- Resultados: recorte en `preproc/`, video con detecciones en `predictions/` y CSV en `output/`.
