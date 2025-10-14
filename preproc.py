import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MARGIN_PERCENT = 0.05


def get_screen_bbox(
    frame: np.ndarray, model: YOLO
) -> Optional[Tuple[int, int, int, int]]:
    """Detecta la pantalla del multímetro y retorna su bounding box."""
    results = model.predict(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        # Tomar la detección con mayor confianza
        confidences = result.boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        box = result.boxes[best_idx]

        conf = float(box.conf.item())
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        return int(x1), int(y1), int(x2), int(y2)

    return None


def add_margin(
    bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Agrega un margen al bounding box para asegurar que toda la pantalla esté incluida."""
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape[:2]

    width = x2 - x1
    height = y2 - y1

    margin_w = int(width * MARGIN_PERCENT)
    margin_h = int(height * MARGIN_PERCENT)

    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(frame_w, x2 + margin_w)
    y2 = min(frame_h, y2 + margin_h)

    return x1, y1, x2, y2


def get_output_path(input_path: Path, output_dir: Path) -> Path:
    """Genera una ruta de salida única basada en el nombre del archivo de entrada."""
    output_dir.mkdir(exist_ok=True)

    base_name = input_path.stem
    extension = ".mp4"

    output_path = output_dir / f"{base_name}{extension}"
    if not output_path.exists():
        return output_path

    counter = 1
    while True:
        output_path = output_dir / f"{base_name}_{counter}{extension}"
        if not output_path.exists():
            return output_path
        counter += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesamiento: detectar y recortar pantalla de multímetro"
    )
    parser.add_argument("video", type=str, help="Ruta al video de entrada")
    parser.add_argument(
        "--model",
        type=str,
        default="models/screen.pt",
        help="Ruta al modelo de detección de pantalla",
    )
    args = parser.parse_args()

    # Ruta al modelo entrenado para detectar la pantalla completa
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo de pantalla en {model_path}. "
            "Verifica que el archivo exista en la ubicación correcta."
        )

    model = YOLO(model_path.as_posix())

    # Video de entrada
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el video en {video_path}.")

    # Crear carpeta de salida y generar nombre único
    preproc_dir = Path("preproc")
    output_path = get_output_path(video_path, preproc_dir)

    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print("Detectando pantalla y recortando video...")

    cap = cv2.VideoCapture(video_path.as_posix())

    # Detectar la pantalla en el primer frame para obtener el bbox
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("No se pudo leer el primer frame del video")

    bbox = get_screen_bbox(first_frame, model)
    if bbox is None:
        raise ValueError(
            "No se pudo detectar la pantalla en el primer frame. "
            "Verifica que el modelo esté entrenado correctamente y que la pantalla sea visible."
        )

    # Agregar margen al bbox
    bbox = add_margin(bbox, first_frame.shape)
    x1, y1, x2, y2 = bbox

    crop_width = x2 - x1
    crop_height = y2 - y1

    print(
        f"Pantalla detectada en: x={x1}, y={y1}, w={crop_width}, h={crop_height}"
    )

    # Reiniciar el video al inicio
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Configurar el video de salida con las dimensiones recortadas
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(
        output_path.as_posix(),
        fourcc,
        fps,
        (crop_width, crop_height),
    )

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Recortar el frame
        cropped_frame = frame[y1:y2, x1:x2]
        writer.write(cropped_frame)

        frame_count += 1
        if frame_count % 30 == 0 or frame_count == total_frames:
            progress = (
                (frame_count / total_frames) * 100 if total_frames > 0 else 0
            )
            print(
                f"Procesando: {frame_count}/{total_frames} frames ({progress:.1f}%)"
            )

    cap.release()
    writer.release()

    print(f"Video recortado guardado en: {output_path}")


if __name__ == "__main__":
    main()
