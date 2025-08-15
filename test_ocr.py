import argparse
import os
import sys
import subprocess  # Necesario para ejecutar comandos externos como ssocr

import re
import cv2
import easyocr
import pytesseract

# Inicialización de EasyOCR (se inicializa solo una vez si se usa)
easyocr_reader = None


def preprocess_image(image_or_path, resize_factor=3):
    """Carga (si es ruta) y preprocesa la imagen para OCR.
    Devuelve imagen binarizada adecuada para OCR.
    """
    img = (
        cv2.imread(image_or_path, cv2.IMREAD_GRAYSCALE)
        if isinstance(image_or_path, str)
        else image_or_path
    )
    if img is None:
        return None

    # upscale to help OCR on small seven-seg
    if resize_factor != 1:
        img = cv2.resize(
            img,
            None,
            fx=resize_factor,
            fy=resize_factor,
            interpolation=cv2.INTER_CUBIC,
        )

    # equalize / CLAHE
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    except Exception:
        img = cv2.equalizeHist(img)

    # binarize
    th = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    # morphology to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    return th


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
    # intentar correcciones comunes y reintentar
    t2 = correct_common_chars(t)
    m = re.search(r"\b\d{1,3}\.\d\b", t2)
    if m:
        return m.group()
    return None


def ocr_with_tesseract_full(img):
    """Ejecuta Tesseract sobre la imagen completa con whitelist.
    Devuelve texto (str).
    """
    try:
        text = pytesseract.image_to_string(
            img,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.",
        )
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        print(
            "Error: Tesseract no encontrado. Asegúrate de que está instalado y en tu PATH."
        )
        return None
    except Exception as e:
        print(f"Error al usar Tesseract: {e}")
        return None


def segment_and_ocr_tesseract(img):
    """Segmenta blobs en la imagen binarizada y aplica Tesseract por carácter (psm 10).
    Devuelve la concatenación de caracteres reconocidos.
    """
    # find contours on binary image
    contours, _ = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # heurística: filtrar contornos pequeños o extremadamente altos/anchos
        if area < 30:
            continue
        if h < 5 or w < 3:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return ""

    # ordenar left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])

    chars = []
    for x, y, w, h in boxes:
        pad_x = max(1, int(w * 0.15))
        pad_y = max(1, int(h * 0.2))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x + w + pad_x)
        y2 = min(img.shape[0], y + h + pad_y)
        roi = img[y1:y2, x1:x2]
        # invert back to white text on black if needed for psm 10
        roi_for_tess = cv2.bitwise_not(roi)
        # resize small crops
        roi_for_tess = cv2.resize(
            roi_for_tess, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
        ch = pytesseract.image_to_string(
            roi_for_tess,
            config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.",
        )
        ch = ch.strip()
        if ch:
            chars.append(ch)

    return "".join(chars)


def perform_ocr(image_path, ocr_engine, save_preproc=False):
    """
    Realiza el reconocimiento OCR en la imagen usando el motor especificado.
    Retorna todo el texto extraído (raw) y además imprime el intento de formato válido.
    """
    extracted_text = None

    # preprocesar imagen (binarizada)
    bin_img = preprocess_image(image_path)
    if bin_img is None:
        return None

    # guardar imagen preprocesada si se solicitó
    if save_preproc and isinstance(image_path, str):
        try:
            base, ext = os.path.splitext(image_path)
            out_name = f"{base}_PREPROC_{ocr_engine}.png"
            # convertir a formato visual (invertir para ver texto blanco sobre negro si se desea)
            vis = bin_img.copy()
            # guardamos la binaria tal cual para diagnóstico
            cv2.imwrite(out_name, vis)
            print(f"Imagen preprocesada guardada en: {out_name}")
        except Exception as e:
            print(f"No se pudo guardar la imagen preprocesada: {e}")

    # Para EasyOCR necesitamos la imagen en escala de grises o BGR
    if ocr_engine == "easyocr":
        global easyocr_reader
        if easyocr_reader is None:
            print("Inicializando EasyOCR (puede tardar la primera vez)...")
            # GPU puede fallar en entornos sin CUDA; si da error, cambia a gpu=False
            try:
                easyocr_reader = easyocr.Reader(["en"], gpu=True)
            except Exception:
                easyocr_reader = easyocr.Reader(["en"], gpu=False)
            print("EasyOCR inicializado.")

        # easyocr espera imagen en BGR o gray (no binaria invertida idealmente)
        # convertimos bin_img (binary inverted) a formato esperado
        proc = cv2.bitwise_not(bin_img)
        try:
            ocr_results = easyocr_reader.readtext(
                proc, detail=0, allowlist="0123456789."
            )
        except TypeError:
            # versiones antiguas pueden no aceptar allowlist
            ocr_results = easyocr_reader.readtext(proc, detail=0)

        # join results and apply EasyOCR-specific normalizations
        extracted_text = "\n".join(ocr_results)
        # basic normalization: remove spaces/newlines, commas -> dot
        extracted_text = extracted_text.replace(" ", "").replace("\n", "")
        extracted_text = extracted_text.replace(",", ".")
        # correct common misread characters (O->0, I->1, etc.)
        extracted_text = correct_common_chars(extracted_text)
        # if multiple dots, keep only the last as decimal separator
        if extracted_text.count(".") > 1:
            parts = extracted_text.split(".")
            extracted_text = (
                "".join(parts[:-1]).replace(".", "") + "." + parts[-1]
            )

    elif ocr_engine == "tesseract":
        # primer intento: tesseract sobre la imagen completa
        text_full = ocr_with_tesseract_full(cv2.bitwise_not(bin_img))
        extracted_text = text_full

        # si no se encuentra lectura válida, intentar segmentación por dígito
        if extract_reading(text_full) is None:
            seg = segment_and_ocr_tesseract(bin_img)
            if seg:
                extracted_text = seg

    elif ocr_engine == "ssocr":
        # ssocr trabaja directamente con el archivo de imagen.
        # Necesitamos asegurarnos de que image_path sea una ruta de archivo.
        if not isinstance(image_path, str):
            print("Error: ssocr requiere una ruta de archivo de imagen.")
            return None
        try:
            command = ["ssocr", "-d", "auto", image_path]
            result = subprocess.run(
                command, capture_output=True, text=True, check=True
            )
            extracted_text = result.stdout.strip()
        except FileNotFoundError:
            print(
                "Error: ssocr no encontrado. Asegúrate de que está instalado y en tu PATH."
            )
            print(
                "Puedes descargarlo de: https://www.unix-ag.uni-kl.de/~auerswal/ssocr/"
            )
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar ssocr: {e}")
            print(f"Stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"Error inesperado al usar ssocr: {e}")

    else:
        print(
            f"Motor OCR '{ocr_engine}' no soportado. Por favor, elige 'easyocr', 'tesseract' o 'ssocr'."
        )
        return None

    # intentar extraer lectura en formato nnn.n
    if extracted_text:
        reading = extract_reading(extracted_text)
        if reading:
            return reading

        # si no se detectó el punto, intentar inferirlo cuando la salida sean sólo dígitos
        only_digits = re.sub(r"[^0-9]", "", extracted_text)
        if len(only_digits) >= 2:
            # suponer que el último dígito es la parte decimal: e.j. '655' -> '65.5'
            inferred = only_digits[:-1] + "." + only_digits[-1]
            print(f"[inferred decimal] {inferred} from raw '{extracted_text}'")
            return inferred

    return extracted_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Probar motores OCR (EasyOCR/Tesseract/ssocr) en una imagen binarizada preprocesada."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ruta a la imagen binarizada (por ejemplo, 'bin_041142.jpg').",
    )
    parser.add_argument(
        "--ocr_engine",
        type=str,
        default="all",  # Cambiado de 'both' a 'all'
        choices=[
            "easyocr",
            "tesseract",
            "ssocr",
            "all",
        ],  # Cambiado 'both' por 'all'
        help="Motor OCR a utilizar: 'easyocr', 'tesseract', 'ssocr' o 'all'. Por defecto es 'all'.",
    )
    parser.add_argument(
        "--save_preproc",
        action="store_true",
        help="Guardar la imagen preprocesada con sufijo _PREPROC_<engine>.png",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: La imagen '{args.image}' no se encontró.")
        sys.exit(1)

    print(f"\n--- Probando OCR en la imagen: {args.image} ---")

    engines_to_test = []
    if args.ocr_engine == "all":  # Condición cambiada a 'all'
        engines_to_test = ["easyocr", "tesseract", "ssocr"]
    else:
        engines_to_test = [args.ocr_engine]

    for engine in engines_to_test:
        print(f"\n===== Motor: {engine.upper()} =====")

        extracted_text = perform_ocr(
            args.image, engine, save_preproc=args.save_preproc
        )

        if extracted_text:
            print(f"Texto extraído:\n{extracted_text}")
        else:
            print("No se extrajo texto.")
        print("=" * 30)

    print("\n--- Prueba de OCR finalizada ---")
