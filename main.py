import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Importar funciones de preproc
from preproc import get_screen_bbox, add_margin

# Importar funciones de infer_video
from infer_video import (
    decode_reading,
    annotate_frame,
    get_output_path,
    MedianFilter,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
)


def preprocess_video(
    video_path: Path, screen_model_path: Path, preproc_dir: Path
) -> Path:
    """Preprocesa el video detectando y recortando la pantalla."""
    print(f"Preprocesando: {video_path}")

    if not screen_model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo de pantalla en {screen_model_path}."
        )

    model = YOLO(screen_model_path.as_posix())
    output_path = get_output_path(video_path, preproc_dir, extension=".mp4")

    cap = cv2.VideoCapture(video_path.as_posix())

    # Detectar la pantalla en el primer frame
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("No se pudo leer el primer frame del video")

    bbox = get_screen_bbox(first_frame, model)
    if bbox is None:
        raise ValueError(
            "No se pudo detectar la pantalla. "
            "Verifica que el modelo esté entrenado correctamente."
        )

    bbox = add_margin(bbox, first_frame.shape)
    x1, y1, x2, y2 = bbox
    crop_width = x2 - x1
    crop_height = y2 - y1

    # Reiniciar el video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Configurar video de salida
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(
        output_path.as_posix(), fourcc, fps, (crop_width, crop_height)
    )

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[y1:y2, x1:x2]
        writer.write(cropped_frame)

        frame_count += 1

    cap.release()
    writer.release()

    print(f"Video recortado: {output_path}")
    return output_path


def detect_readings(
    video_path: Path,
    model_path: Path,
    predictions_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Detecta las lecturas en el video preprocesado."""
    print(f"Detectando lecturas: {video_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo de dígitos en {model_path}."
        )

    model = YOLO(model_path.as_posix())
    class_map = model.names

    # Generar rutas de salida
    output_video_path = get_output_path(
        video_path, predictions_dir, extension=".mp4"
    )
    csv_path = get_output_path(video_path, output_dir, extension=".csv")

    cap = cv2.VideoCapture(video_path.as_posix())
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video_path.as_posix(),
        fourcc,
        fps,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    # Abrir CSV
    import csv

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Time (ms)", "Reading"])

    # Inicializar filtro
    median_filter = MedianFilter()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
        filtered_reading = median_filter.add_reading(reading)

        frame_count += 1
        time_ms = int((frame_count / fps) * 1000)

        csv_writer.writerow([time_ms, filtered_reading])
        annotate_frame(frame, filtered_reading)
        writer.write(frame)

    cap.release()
    writer.release()
    csv_file.close()

    print(f"Video anotado: {output_video_path}")
    print(f"CSV guardado: {csv_path}")

    return output_video_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline completo: preprocesamiento y detección de lecturas"
    )
    parser.add_argument("video", type=str, help="Ruta al video de entrada")
    parser.add_argument(
        "--screen-model",
        type=str,
        default="models/screen.pt",
        help="Ruta al modelo de detección de pantalla",
    )
    parser.add_argument(
        "--digits-model",
        type=str,
        default="models/digit_detection.pt",
        help="Ruta al modelo de detección de dígitos",
    )
    parser.add_argument(
        "--skip-preproc",
        action="store_true",
        help="Saltar preprocesamiento (usar si el video ya está recortado)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: No se encontró el video en {video_path}")
        sys.exit(1)

    screen_model_path = Path(args.screen_model)
    digits_model_path = Path(args.digits_model)

    try:
        # Preprocesamiento (opcional)
        if not args.skip_preproc:
            preproc_dir = Path("preproc")
            preprocessed_video = preprocess_video(
                video_path, screen_model_path, preproc_dir
            )
        else:
            preprocessed_video = video_path

        # Detección de lecturas
        predictions_dir = Path("predictions")
        output_dir = Path("output")

        output_video, csv_file = detect_readings(
            preprocessed_video,
            digits_model_path,
            predictions_dir,
            output_dir,
        )

        print(f"\nCompletado. CSV: {csv_file}")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
