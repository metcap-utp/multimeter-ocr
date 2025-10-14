import argparse
import csv
import statistics
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Tuple

import cv2
from ultralytics import YOLO


CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 20
MEDIAN_WINDOW_SIZE = 5


def decode_reading(detections: Iterable[Tuple[float, str]]) -> str:
    """Return the ordered reading string from (x_center, label) pairs."""
    ordered = sorted(detections, key=lambda item: item[0])
    raw = "".join(label for _, label in ordered)
    return normalize_reading(raw)


def normalize_reading(raw: str) -> str:
    """Format the raw label sequence to show all detected digits with decimal point.

    El último dígito detectado siempre será después del punto decimal.
    La detección del punto se ignora completamente.
    """
    if not raw:
        return "0.0"

    # Extraer solo los dígitos, ignorando el punto
    digits = [ch for ch in raw if ch.isdigit()]

    if not digits:
        return "0.0"

    # El último dígito siempre va después del punto
    if len(digits) == 1:
        before = "0"
        after = digits[0]
    else:
        before = "".join(digits[:-1])
        after = digits[-1]

    return f"{before}.{after}"


def annotate_frame(frame, reading: str) -> None:
    text = reading if reading else "0.0"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    frame_h, frame_w = frame.shape[:2]
    padding = 10
    x1 = frame_w - text_w - 2 * padding
    y1 = padding
    x2 = frame_w - padding
    y2 = text_h + 2 * padding

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x1 + padding, y2 - padding),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def get_output_path(
    input_path: Path, output_dir: Path, extension: str = ".mp4"
) -> Path:
    """Generate a unique output path based on input filename."""
    output_dir.mkdir(exist_ok=True)

    base_name = input_path.stem

    # Intentar sin número primero
    output_path = output_dir / f"{base_name}{extension}"
    if not output_path.exists():
        return output_path

    # Si existe, agregar números hasta encontrar uno disponible
    counter = 1
    while True:
        output_path = output_dir / f"{base_name}_{counter}{extension}"
        if not output_path.exists():
            return output_path
        counter += 1


class MedianFilter:
    """Filtro de mediana adaptativo para detectar y corregir outliers aislados."""

    def __init__(self, window_size: int = MEDIAN_WINDOW_SIZE):
        self.window_size = window_size
        self.readings: Deque[float] = deque(maxlen=window_size)
        self.consecutive_similar = 0
        self.last_outlier_value = None
        self.similarity_threshold = 3

    def add_reading(self, reading: str) -> str:
        """Agrega una lectura y retorna la lectura filtrada."""
        try:
            value = float(reading)
        except ValueError:
            return reading

        # Si no tenemos suficientes lecturas, construir historial
        if len(self.readings) < 3:
            self.readings.append(value)
            self.consecutive_similar = 0
            return reading

        # Calcular estadísticas de la ventana
        median_value = statistics.median(self.readings)
        mean_value = statistics.mean(self.readings)

        # Calcular MAD (Median Absolute Deviation)
        deviations = [abs(r - median_value) for r in self.readings]
        mad = statistics.median(deviations)

        # Determinar si es un outlier con criterios más estrictos
        is_outlier = False

        if mad > 0.01:
            # Usar z-score modificado con MAD
            z_score = abs(value - median_value) / (1.4826 * mad)
            is_outlier = z_score > 2.5
        else:
            # Si MAD es muy pequeño (valores muy estables), ser más estricto
            deviation_percent = (
                abs(value - median_value) / median_value
                if median_value > 0
                else 0
            )
            is_outlier = deviation_percent > 0.15

        # Detectar cambios de orden de magnitud (ej: 1.8 vs 18.6)
        if median_value > 0.5:
            ratio = value / median_value
            # Si el ratio es cercano a 10, probablemente es un error de detección de dígito
            if 8 < ratio < 12 or 0.083 < ratio < 0.125:
                is_outlier = True
            # Cambios muy extremos también son outliers
            elif ratio > 15 or ratio < 0.067:
                is_outlier = True

        if is_outlier:
            # Si el outlier es similar al anterior, incrementar contador
            if (
                self.last_outlier_value is not None
                and abs(value - self.last_outlier_value) < 0.5
            ):
                self.consecutive_similar += 1
            else:
                self.consecutive_similar = 1
                self.last_outlier_value = value

            # Solo aceptar si hay suficientes outliers similares consecutivos
            if self.consecutive_similar >= self.similarity_threshold:
                # Es un cambio real confirmado
                self.readings.append(value)
                self.consecutive_similar = 0
                self.last_outlier_value = None
                return reading
            else:
                # Outlier no confirmado, usar mediana
                return self._format_value(median_value)
        else:
            # Lectura normal
            self.consecutive_similar = 0
            self.last_outlier_value = None
            self.readings.append(value)
            return reading

    def _format_value(self, value: float) -> str:
        """Formatea un valor flotante al formato X.Y"""
        rounded = round(value, 1)
        return f"{rounded:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferencia de lectura de multímetro en video"
    )
    parser.add_argument("video", type=str, help="Ruta al video de entrada")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/multimeter-yolov8n/weights/best.pt",
        help="Ruta al modelo entrenado",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en {model_path}."
        )

    model = YOLO(model_path.as_posix())
    class_map: Dict[int, str] = model.names  # type: ignore[assignment]

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el video en {video_path}.")

    # Crear carpeta de predicciones y generar nombre único para el video
    predictions_dir = Path("predictions")
    output_path = get_output_path(
        video_path, predictions_dir, extension=".mp4"
    )

    # Crear carpeta output y generar nombre único para el CSV
    output_dir = Path("output")
    csv_path = get_output_path(video_path, output_dir, extension=".csv")

    print(f"Input: {video_path}")
    print(f"Output video: {output_path}")
    print(f"Output CSV: {csv_path}")

    cap = cv2.VideoCapture(video_path.as_posix())
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(
        output_path.as_posix(),
        fourcc,
        fps,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    # Abrir archivo CSV para escritura
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Time (ms)", "Reading"])

    # Inicializar filtro de mediana
    median_filter = MedianFilter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            verbose=False,
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf < CONF_THRESHOLD:
                    continue
                cls_idx = int(box.cls.item())
                label = class_map.get(cls_idx, "?")
                x_center = float(box.xywh[0][0].item())
                detections.append((x_center, label))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{label}:{conf:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        reading = decode_reading(detections)

        # Aplicar filtro de mediana para detectar outliers
        filtered_reading = median_filter.add_reading(reading)

        # Calcular tiempo en milisegundos
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        time_ms = int((frame_count / fps) * 1000)

        # Guardar en CSV
        csv_writer.writerow([time_ms, filtered_reading])

        annotate_frame(frame, filtered_reading)
        writer.write(frame)

    cap.release()
    writer.release()
    csv_file.close()

    print(f"Video procesado guardado en: {output_path}")
    print(f"Datos guardados en: {csv_path}")


if __name__ == "__main__":
    main()
