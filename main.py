import os
import cv2
import csv
import re
import argparse
import easyocr
from ultralytics import YOLO
import sys
from io import StringIO
import pytesseract

# --- CONFIGURATIONS ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_PATH = "models/best.pt"
CONFIDENCE_THRESHOLD = 0.3
TEST_DIR = "test"

# --- CROPPING CONFIGURATIONS ADJUSTED ---
CROP_START_X = 0.24
CROP_END_X = 0.95
CROP_START_Y = 0.42
CROP_END_Y = 0.85

# --- ERROR CHECKING CONFIGURATIONS ---
FRAME_SKIP = 1  # Procesar cada N frames (reducir frecuencia de muestreo)
MAX_CHANGE_PERCENT = (
    50.0  # Máximo cambio porcentual permitido entre lecturas consecutivas
)
MEDIAN_WINDOW_SIZE = 5  # Tamaño de ventana para filtro de mediana móvil
MIN_STABLE_COUNT = (
    2  # Mínimo de lecturas estables antes de aceptar un cambio grande
)
LARGE_JUMP_THRESHOLD = 90.0  # Porcentaje que se considera un "salto grande"
LARGE_JUMP_CONFIRMATION_FRAMES = (
    3  # Frames consecutivos para confirmar un salto grande
)
OUTLIER_ISOLATION_THRESHOLD = (
    2  # Frames consecutivos diferentes para considerar outlier aislado
)


model = YOLO(MODEL_PATH)
easyocr_reader = None


def extract_decimal(text):
    """
    Extrae un número decimal (digitos.digitos) de una cadena de texto.
    """
    match = re.search(r"\d+\.\d+", text)
    return match.group() if match else None


def correct_common_chars(s: str) -> str:
    if not s:
        return s
    corrections = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5"}
    for k, v in corrections.items():
        s = s.replace(k, v)
    return s


def is_valid_reading_change(
    new_reading, prev_reading, max_change_percent=MAX_CHANGE_PERCENT
):
    """
    Verifica si el cambio entre dos lecturas es válido.
    Retorna True si el cambio es aceptable, False si es un outlier.
    """
    if prev_reading is None:
        return True

    try:
        new_val = float(new_reading)
        prev_val = float(prev_reading)

        # Si alguno de los valores es 0, usar diferencia absoluta
        if prev_val == 0:
            return (
                abs(new_val - prev_val) <= 10.0
            )  # Máximo 10 unidades de diferencia

        # Calcular cambio porcentual
        change_percent = abs((new_val - prev_val) / prev_val) * 100
        return change_percent <= max_change_percent
    except (ValueError, ZeroDivisionError):
        return False


def is_large_jump(new_reading, prev_reading):
    """
    Determina si hay un salto grande entre dos lecturas.
    """
    if prev_reading is None:
        return False

    try:
        new_val = float(new_reading)
        prev_val = float(prev_reading)

        if prev_val == 0:
            return abs(new_val) > 10.0

        change_percent = abs((new_val - prev_val) / prev_val) * 100
        return change_percent > LARGE_JUMP_THRESHOLD
    except (ValueError, ZeroDivisionError):
        return False


def validate_large_jump(
    new_reading,
    confirmation_buffer,
    min_confirmations=LARGE_JUMP_CONFIRMATION_FRAMES,
):
    """
    Valida si un salto grande es legítimo basándose en confirmaciones consecutivas.
    Retorna True si el salto es válido (confirmado por lecturas consecutivas similares).
    """
    if len(confirmation_buffer) < min_confirmations:
        return False

    try:
        new_val = float(new_reading)

        # Verificar que las últimas lecturas sean similares a la nueva
        similar_count = 0
        for reading in confirmation_buffer[-min_confirmations:]:
            if (
                reading and abs(float(reading) - new_val) <= 0.3
            ):  # Tolerancia más amplia
                similar_count += 1

        return similar_count >= min_confirmations - 1
    except (ValueError, TypeError):
        return False


def is_isolated_outlier(new_reading, recent_readings, future_buffer=None):
    """
    Detecta si una lectura es un outlier aislado (diferente a las anteriores Y posteriores).
    Si future_buffer tiene lecturas, las usa para validación adicional.
    """
    if len(recent_readings) < 2:
        return False

    try:
        new_val = float(new_reading)

        # Verificar diferencia con lecturas anteriores
        different_from_past = True
        for reading in recent_readings[-2:]:
            if reading and abs(float(reading) - new_val) <= 0.5:
                different_from_past = False
                break

        # Si no es muy diferente del pasado, no es outlier
        if not different_from_past:
            return False

        # Si tenemos lecturas futuras, verificar también con ellas
        if future_buffer and len(future_buffer) >= 2:
            different_from_future = True
            for reading in future_buffer[:2]:
                if reading and abs(float(reading) - new_val) <= 0.5:
                    different_from_future = False
                    break

            # Es outlier si es diferente tanto del pasado como del futuro
            return different_from_past and different_from_future

        # Sin futuro, solo podemos basarnos en el pasado
        return different_from_past
    except (ValueError, TypeError):
        return False


def apply_median_filter(readings_buffer):
    """
    Aplica un filtro de mediana a un buffer de lecturas.
    Retorna la mediana si hay suficientes valores, None si no.
    """
    if len(readings_buffer) < 3:
        return None

    # Convertir a float y ordenar
    try:
        float_readings = [float(r) for r in readings_buffer if r is not None]
        if len(float_readings) < 3:
            return None

        float_readings.sort()
        mid = len(float_readings) // 2

        if len(float_readings) % 2 == 0:
            median = (float_readings[mid - 1] + float_readings[mid]) / 2
        else:
            median = float_readings[mid]

        # Retornar como string con formato original
        return f"{median:.1f}"
    except (ValueError, IndexError):
        return None


def validate_reading_stability(
    new_reading, recent_readings, min_stable_count=MIN_STABLE_COUNT
):
    """
    Verifica si una lectura es estable comparándola con lecturas recientes.
    Retorna True si la lectura es consistente.
    """
    if len(recent_readings) < min_stable_count:
        return True

    try:
        new_val = float(new_reading)
        similar_count = 0

        for reading in recent_readings[-min_stable_count:]:
            if (
                reading and abs(float(reading) - new_val) <= 0.2
            ):  # Tolerancia de 0.2
                similar_count += 1

        return similar_count >= min_stable_count - 1
    except (ValueError, TypeError):
        return False


def extract_reading(text: str):
    """Extrae una lectura con formato nnn.n o mmm.n desde text.
    Retorna la primera coincidencia o None.
    """
    if not text:
        return None
    t = text.replace(" ", "").replace("\n", "")
    # buscar formato 1-3 digitos + punto + 1 digito
    m = re.search(r"\b\d{1,3}\.\d\b", t)
    if m:
        return m.group()

    t2 = correct_common_chars(t)
    m = re.search(r"\b\d{1,3}\.\d\b", t2)
    if m:
        return m.group()
    return None


def perform_ocr(image, ocr_engine):
    """
    Realiza el reconocimiento OCR en la imagen usando el motor especificado.
    Retorna el texto extraído o None si no se encuentra un decimal válido.
    """
    extracted_text = None

    if ocr_engine == "easyocr":
        global easyocr_reader
        if easyocr_reader is None:
            try:
                easyocr_reader = easyocr.Reader(["en"], gpu=True)
            except Exception:
                easyocr_reader = easyocr.Reader(["en"], gpu=False)
            print("EasyOCR inicializado.")

        try:
            ocr_results = easyocr_reader.readtext(
                image, detail=0, allowlist="0123456789."
            )
        except TypeError:
            ocr_results = easyocr_reader.readtext(image, detail=0)

        extracted_text = "\n".join(ocr_results)

        extracted_text = extracted_text.replace(" ", "").replace("\n", "")
        extracted_text = extracted_text.replace(",", ".")

        extracted_text = correct_common_chars(extracted_text)

        if extracted_text.count(".") > 1:
            parts = extracted_text.split(".")
            extracted_text = (
                "".join(parts[:-1]).replace(".", "") + "." + parts[-1]
            )

    elif ocr_engine == "tesseract":
        try:
            raw_text = pytesseract.image_to_string(
                image,
                config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.",
            )
            extracted_text = (
                raw_text.strip().replace(" ", "").replace("\n", "")
            )
        except pytesseract.TesseractNotFoundError:
            print(
                "Error: Tesseract no encontrado. Asegúrate de que está instalado y en tu PATH."
            )
            print(
                "Si lo instalaste manualmente, especifica la ruta con 'pytesseract.pytesseract.tesseract_cmd = r\"ruta\\a\\tesseract.exe\"'"
            )
            return None
        except Exception as e:
            print(f"Error al usar Tesseract: {e}")
    else:
        print(f"OCR engine '{ocr_engine}' not supported. Returning None.")
        return None

    if extracted_text:
        reading = extract_reading(extracted_text)
        if reading:
            return reading

        only_digits = re.sub(r"[^0-9]", "", extracted_text)
        if len(only_digits) >= 2:
            inferred = only_digits[:-1] + "." + only_digits[-1]
            return inferred

    return None


def process_video_with_engine(file_name, debug, current_ocr_engine):
    """
    Lógica principal de procesamiento de video para un motor OCR específico.
    """
    base_name = os.path.splitext(file_name)[0]
    csv_path = os.path.join(
        OUTPUT_DIR, f"{base_name}_{current_ocr_engine}.csv"
    )

    # Si el CSV ya existe, crear uno con número entre paréntesis
    if os.path.exists(csv_path):
        counter = 1
        while os.path.exists(csv_path):
            csv_path = os.path.join(
                OUTPUT_DIR, f"{base_name}_{current_ocr_engine}({counter}).csv"
            )
            counter += 1

    base_test_path = os.path.join(
        TEST_DIR,
        f"{base_name}_{current_ocr_engine}",
    )

    current_debug_path = None
    if debug:
        original_base_test_path = base_test_path
        counter = 1
        while os.path.exists(base_test_path):
            base_test_path = f"{original_base_test_path}({counter})"
            counter += 1
        current_debug_path = base_test_path

        print(
            f"Creando carpeta de depuración para {current_ocr_engine}: {current_debug_path}"
        )

        path_crops = os.path.join(current_debug_path, "crops")
        path_refined_crops = os.path.join(current_debug_path, "refined_crops")
        path_bins = os.path.join(current_debug_path, "bins")
        path_frames = os.path.join(current_debug_path, "frames")

        os.makedirs(path_crops, exist_ok=True)
        os.makedirs(path_refined_crops, exist_ok=True)
        os.makedirs(path_bins, exist_ok=True)
        os.makedirs(path_frames, exist_ok=True)
    else:
        path_crops = None
        path_refined_crops = None
        path_bins = None
        path_frames = None

    if not os.path.isfile(os.path.join(INPUT_DIR, file_name)):
        print(f"File not found: {os.path.join(INPUT_DIR, file_name)}")
        return

    # --- INICIO DEL BLOQUE DE IMPRESIÓN PARA CADA VIDEO/MOTOR ---
    print("\n" + "=" * 70)
    print(f" Starting processing of: {file_name}")
    print(f" Current OCR engine: {current_ocr_engine.upper()}")
    print("=" * 70 + "\n")
    # --- FIN DEL BLOQUE DE IMPRESIÓN ---

    cap = cv2.VideoCapture(os.path.join(INPUT_DIR, file_name))
    prev_value = None
    results = []
    frame_count = 0

    # Buffers para el sistema de chequeo de errores
    readings_buffer = []  # Buffer para filtro de mediana
    recent_readings = []  # Buffer para validación de estabilidad
    rejected_readings = []  # Para debug: lecturas rechazadas
    confirmation_buffer = []  # Buffer para confirmar saltos grandes
    pending_large_jump = (
        None  # Lectura con salto grande pendiente de confirmación
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Reducir frecuencia de muestreo: procesar solo cada FRAME_SKIP frames
        if frame_count % FRAME_SKIP != 0:
            continue

        time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        time_s = time_ms / 1000
        valid_reading = None
        raw_reading = None

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            detections = model(frame)[0]
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        for box in detections.boxes:
            if box.conf < CONFIDENCE_THRESHOLD:
                continue

            x1_yolo, y1_yolo, x2_yolo, y2_yolo = map(int, box.xyxy[0])
            yolo_screen_crop = frame[y1_yolo:y2_yolo, x1_yolo:x2_yolo]

            h, w = yolo_screen_crop.shape[:2]

            x_start_refined = int(w * CROP_START_X)
            x_end_refined = int(w * CROP_END_X)
            y_start_refined = int(h * CROP_START_Y)
            y_end_refined = int(h * CROP_END_Y)

            x_start_refined = max(0, min(x_start_refined, w))
            x_end_refined = max(0, min(x_end_refined, w))
            y_start_refined = max(0, min(y_start_refined, h))
            y_end_refined = max(0, min(y_end_refined, h))

            refined_screen = yolo_screen_crop[
                y_start_refined:y_end_refined, x_start_refined:x_end_refined
            ]

            if refined_screen.shape[0] == 0 or refined_screen.shape[1] == 0:
                if debug:
                    tag = f"{time_ms:06d}"
                    cv2.imwrite(
                        os.path.join(path_crops, f"crop_yolo_{tag}.jpg"),
                        yolo_screen_crop,
                    )
                continue

            gray = cv2.cvtColor(refined_screen, cv2.COLOR_BGR2GRAY)

            binarized = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                21,
                5,
            )

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binarized = cv2.dilate(binarized, kernel, iterations=1)

            raw_reading = perform_ocr(binarized, current_ocr_engine)

            # Aplicar sistema de chequeo de errores inteligente
            if raw_reading:
                # 1. Verificar si hay un salto grande
                has_large_jump = is_large_jump(raw_reading, prev_value)

                if has_large_jump:
                    # Es un salto grande - necesita confirmación
                    confirmation_buffer.append(raw_reading)

                    # Si tenemos suficientes confirmaciones, validar el salto
                    if validate_large_jump(raw_reading, confirmation_buffer):
                        # Salto grande confirmado - es válido
                        valid_reading = raw_reading

                        # Limpiar buffer de confirmación
                        confirmation_buffer = []
                        pending_large_jump = None

                        # Resetear buffers para el nuevo nivel de lectura
                        readings_buffer = [raw_reading]
                        recent_readings = [raw_reading]

                        print(
                            f"  ✓ Large jump confirmed: {prev_value} → {raw_reading}"
                        )
                    else:
                        # Salto grande no confirmado aún - esperar más frames
                        pending_large_jump = raw_reading
                        valid_reading = None
                        print(
                            f"  ⏳ Large jump pending confirmation: {raw_reading}"
                        )
                else:
                    # No es un salto grande - aplicar validaciones normales
                    is_valid_change = is_valid_reading_change(
                        raw_reading, prev_value
                    )
                    is_stable = validate_reading_stability(
                        raw_reading, recent_readings
                    )

                    # Limpiar cualquier salto pendiente si esta lectura es consistente con el estado anterior
                    if pending_large_jump and is_valid_change:
                        confirmation_buffer = []
                        pending_large_jump = None

                    # Agregar a buffer para filtro de mediana
                    readings_buffer.append(raw_reading)
                    if len(readings_buffer) > MEDIAN_WINDOW_SIZE:
                        readings_buffer.pop(0)

                    # Aplicar filtro de mediana si hay suficientes lecturas
                    filtered_reading = apply_median_filter(readings_buffer)

                    # Decidir si aceptar la lectura
                    if is_valid_change and is_stable:
                        valid_reading = (
                            filtered_reading
                            if filtered_reading
                            else raw_reading
                        )

                        # Agregar a lecturas recientes para futuras validaciones
                        recent_readings.append(valid_reading)
                        if len(recent_readings) > MIN_STABLE_COUNT * 2:
                            recent_readings.pop(0)
                    else:
                        # Rechazar la lectura por ser inconsistente
                        rejected_readings.append(
                            (
                                time_ms,
                                raw_reading,
                                not is_valid_change,
                                not is_stable,
                            )
                        )
                        valid_reading = None
            else:
                # No hay lectura OCR - limpiar confirmaciones pendientes
                if len(confirmation_buffer) > 0:
                    confirmation_buffer.pop()

            if valid_reading and valid_reading != prev_value:
                results.append((time_ms, valid_reading))
                prev_value = valid_reading

            if debug:
                tag = f"{time_ms:06d}"
                cv2.imwrite(
                    os.path.join(path_crops, f"crop_yolo_{tag}.jpg"),
                    yolo_screen_crop,
                )
                cv2.imwrite(
                    os.path.join(
                        path_refined_crops, f"crop_refined_{tag}.jpg"
                    ),
                    refined_screen,
                )
                cv2.imwrite(
                    os.path.join(path_bins, f"bin_{tag}.jpg"), binarized
                )

                frame_bbox = frame.copy()
                cv2.rectangle(
                    frame_bbox,
                    (x1_yolo, y1_yolo),
                    (x2_yolo, y2_yolo),
                    (0, 255, 0),
                    2,
                )
                x1_refined_abs = x1_yolo + x_start_refined
                y1_refined_abs = y1_yolo + y_start_refined
                x2_refined_abs = x1_yolo + x_end_refined
                y2_refined_abs = y1_yolo + y_end_refined
                cv2.rectangle(
                    frame_bbox,
                    (x1_refined_abs, y1_refined_abs),
                    (x2_refined_abs, y2_refined_abs),
                    (255, 0, 0),
                    2,
                )
                cv2.imwrite(
                    os.path.join(path_frames, f"frame_{tag}.jpg"), frame_bbox
                )

        # --- IMPRESIÓN POR CADA EVENTO CON SEPARADORES COMPLETOS ---
        print("-" * 30)  # Divisor antes
        if valid_reading:
            print(f"Time: {time_s:.1f} s | Reading: {valid_reading}")
            if raw_reading != valid_reading:
                print(f"  └─ Raw: {raw_reading} → Filtered: {valid_reading}")
        else:
            print(f"Time: {time_s:.1f} s | No reading")
            if raw_reading:
                if pending_large_jump:
                    print(
                        f"  └─ Pending large jump: {raw_reading} (need {LARGE_JUMP_CONFIRMATION_FRAMES - len(confirmation_buffer)} more confirmations)"
                    )
                else:
                    print(
                        f"  └─ Rejected: {raw_reading} (outlier or unstable)"
                    )
        print("-" * 30)  # Divisor después
        # --- FIN IMPRESIÓN ---

    cap.release()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (ms)", "Reading"])
        writer.writerows(results)

    # --- FINAL DEL BLOQUE DE IMPRESIÓN PARA CADA VIDEO/MOTOR ---
    print(f"\nCSV saved: {csv_path}")
    print(f"Total valid readings: {len(results)}")
    print(f"Total rejected readings: {len(rejected_readings)}")

    if rejected_readings and debug:
        print("\nRejected readings summary:")
        for time_ms, reading, invalid_change, unstable in rejected_readings[
            :10
        ]:  # Mostrar solo los primeros 10
            reasons = []
            if invalid_change:
                reasons.append("invalid_change")
            if unstable:
                reasons.append("unstable")
            print(f"  {time_ms/1000:.1f}s: {reading} ({', '.join(reasons)})")
        if len(rejected_readings) > 10:
            print(f"  ... and {len(rejected_readings) - 10} more")

    print("\n" + "=" * 70)
    print(
        f" Processing of '{file_name}' with {current_ocr_engine.upper()} completed."
    )
    print("=" * 70 + "\n")


def process_video(file_name, debug=False, ocr_engine="easyocr"):
    """
    Función wrapper para manejar la opción 'both'.
    """
    if ocr_engine == "both":
        print(
            f"\nDetected 'both' option. Processing '{file_name}' with EasyOCR and Tesseract."
        )
        process_video_with_engine(file_name, debug, "easyocr")
        process_video_with_engine(file_name, debug, "tesseract")
    else:
        process_video_with_engine(file_name, debug, ocr_engine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video(s) with YOLOv8 and a selectable OCR engine."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Name of the file to process (in input/ folder). If omitted, all will be processed.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug images (crops, bins, frames) in a numbered folder.",
    )
    parser.add_argument(
        "--ocr_engine",
        type=str,
        default="easyocr",
        choices=["easyocr", "tesseract", "both"],
        help="OCR engine to use: 'easyocr', 'tesseract', or 'both'. Default is 'easyocr'.",
    )
    args = parser.parse_args()

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    if args.file:
        process_video(args.file, debug=args.debug, ocr_engine=args.ocr_engine)
    else:
        for file in os.listdir(INPUT_DIR):
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                process_video(
                    file, debug=args.debug, ocr_engine=args.ocr_engine
                )
