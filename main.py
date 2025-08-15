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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        time_s = time_ms / 1000
        valid_reading = None

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

            valid_reading = perform_ocr(binarized, current_ocr_engine)

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
        else:
            print(f"Time: {time_s:.1f} s | No reading")
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
