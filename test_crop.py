import cv2
import os

# --- CONFIGURACIONES ---
INPUT_IMAGE_PATH = "test/input(1)/crops/crop_000033.jpg"  # Usaremos esta imagen para probar el recorte
OUTPUT_IMAGE_PATH = "recorte_lectura.jpg"  # Nombre para la imagen de salida

# Porcentajes de recorte (0.0 a 1.0) - AJUSTADOS PARA LA LECTURA DE CORRIENTE
CROP_START_X = 0.24
CROP_END_X = 0.92
CROP_START_Y = 0.42
CROP_END_Y = 0.85

# --- SCRIPT (el resto del código es el mismo que te di antes) ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(
            f"Error: La imagen de entrada '{INPUT_IMAGE_PATH}' no se encontró."
        )
        print(
            "Por favor, asegúrate de colocarla en el mismo directorio que este script."
        )
    else:
        # Cargar la imagen
        image = cv2.imread(INPUT_IMAGE_PATH)

        if image is None:
            print(
                f"Error: No se pudo cargar la imagen '{INPUT_IMAGE_PATH}'. ¿Formato correcto?"
            )
        else:
            h, w = image.shape[:2]

            # Calcular las coordenadas de recorte en píxeles
            x_start = int(w * CROP_START_X)
            x_end = int(w * CROP_END_X)
            y_start = int(h * CROP_START_Y)
            y_end = int(h * CROP_END_Y)

            # Asegurarse de que las coordenadas sean válidas y dentro de los límites
            x_start = max(0, min(x_start, w))
            x_end = max(0, min(x_end, w))
            y_start = max(0, min(y_start, h))
            y_end = max(0, min(y_end, h))

            # Aplicar el recorte
            cropped_image = image[y_start:y_end, x_start:x_end]

            if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                print(
                    "Advertencia: El recorte resultó en una imagen vacía. Ajusta los porcentajes de recorte."
                )
            else:
                # Guardar la imagen recortada
                cv2.imwrite(OUTPUT_IMAGE_PATH, cropped_image)
                print(f"Imagen original: {w}x{h} píxeles.")
                print(
                    f"Recortada de ({x_start},{y_start}) a ({x_end},{y_end})."
                )
                print(
                    f"Imagen recortada ({cropped_image.shape[1]}x{cropped_image.shape[0]} píxeles) guardada como '{OUTPUT_IMAGE_PATH}'"
                )

                # # Opcional: Mostrar la imagen recortada (COMENTAR O ELIMINAR ESTAS LÍNEAS si te da el error de GUI)
                # cv2.imshow("Imagen Recortada", cropped_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
