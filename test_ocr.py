import argparse
import os
import re
import sys
import subprocess  # Necesario para ejecutar comandos externos como ssocr

import cv2
import easyocr
import pytesseract

# Inicialización de EasyOCR (se inicializa solo una vez si se usa)
easyocr_reader = None


def perform_ocr(image_path, ocr_engine):
    """
    Realiza el reconocimiento OCR en la imagen usando el motor especificado.
    Retorna todo el texto extraído.
    """
    extracted_text = None

    if ocr_engine == "easyocr":
        global easyocr_reader
        if easyocr_reader is None:
            print("Inicializando EasyOCR (puede tardar la primera vez)...")
            # Para pruebas rápidas, considera gpu=False si no tienes una GPU potente
            easyocr_reader = easyocr.Reader(["en"], gpu=True)
            print("EasyOCR inicializado.")

        # EasyOCR funciona bien con imágenes binarizadas o en escala de grises
        # Ajustamos para que reciba la imagen directamente como objeto de OpenCV
        image = (
            cv2.imread(image_path)
            if isinstance(image_path, str)
            else image_path
        )
        if image is None:
            return "Error: No se pudo cargar la imagen para EasyOCR."

        # Convertir a escala de grises si es una imagen a color para mejor rendimiento con OCR
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ocr_results = easyocr_reader.readtext(image, detail=0)
        # Unimos todos los resultados en una sola cadena para una salida más limpia
        extracted_text = "\n".join(ocr_results)

    elif ocr_engine == "tesseract":
        try:
            image = (
                cv2.imread(image_path)
                if isinstance(image_path, str)
                else image_path
            )
            if image is None:
                return "Error: No se pudo cargar la imagen para Tesseract."
            # Convertir a escala de grises si es una imagen a color para mejor rendimiento con OCR
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Tesseract también funciona bien con imágenes binarizadas.
            extracted_text = pytesseract.image_to_string(
                image,
                config="--psm 6 --oem 3",  # Se mantienen otros parámetros de configuración si son relevantes para la calidad general.
            )
            extracted_text = extracted_text.strip()
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

    elif ocr_engine == "ssocr":
        # ssocr trabaja directamente con el archivo de imagen.
        # Necesitamos asegurarnos de que image_path sea una ruta de archivo.
        if not isinstance(image_path, str):
            print("Error: ssocr requiere una ruta de archivo de imagen.")
            return None
        try:
            # Puedes añadir más opciones de ssocr aquí si las necesitas, por ejemplo, -d (digits)
            # o -t (threshold) si la binarización de la imagen no es perfecta.
            # Para este ejemplo, se usa una configuración básica.
            # La salida de ssocr es típicamente solo el número reconocido.
            command = [
                "ssocr",
                "-d",
                "auto",
                image_path,
            ]
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

        extracted_text = perform_ocr(args.image, engine)

        if extracted_text:
            print(f"Texto extraído:\n{extracted_text}")
        else:
            print("No se extrajo texto.")
        print("=" * 30)

    print("\n--- Prueba de OCR finalizada ---")
