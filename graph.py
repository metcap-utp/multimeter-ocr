import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import argparse


def plot_csv(csv_path, save=False):
    """
    Grafica los datos de un archivo CSV con tiempo en segundos.
    """
    if not os.path.exists(csv_path):
        print(f"Error: El archivo '{csv_path}' no existe.")
        return

    try:
        # Leer el CSV
        df = pd.read_csv(csv_path)

        # Verificar que tenga las columnas necesarias
        if "Time (ms)" not in df.columns or "Reading" not in df.columns:
            print(
                "Error: El CSV debe contener las columnas 'Time (ms)' y 'Reading'"
            )
            return

        # Convertir tiempo de milisegundos a segundos
        df["Time (s)"] = df["Time (ms)"] / 1000.0

        # Crear la gráfica
        plt.figure(figsize=(12, 6))
        plt.plot(
            df["Time (s)"],
            df["Reading"],
            marker="o",
            linestyle="-",
            markersize=4,
        )
        plt.xlabel("Tiempo (s)", fontsize=12)
        plt.ylabel("Lectura", fontsize=12)
        plt.title(
            f"Lecturas vs Tiempo - {os.path.basename(csv_path)}", fontsize=14
        )
        plt.grid(True, alpha=0.3)

        # Establecer límite mínimo del eje Y en 0.0
        y_max = df["Reading"].max()
        plt.ylim(0.0, y_max * 1.1)

        plt.tight_layout()

        # Mostrar estadísticas
        print(f"\n{'='*50}")
        print(f"Archivo: {csv_path}")
        print(f"Total de lecturas: {len(df)}")
        print(
            f"Rango de tiempo: {df['Time (s)'].min():.2f}s - {df['Time (s)'].max():.2f}s"
        )
        print(f"Valor mínimo: {df['Reading'].min()}")
        print(f"Valor máximo: {df['Reading'].max()}")
        print(f"Valor promedio: {df['Reading'].mean():.2f}")
        print(f"{'='*50}\n")

        if save:
            # Crear carpeta graphs si no existe
            graphs_dir = "graphs"
            os.makedirs(graphs_dir, exist_ok=True)

            # Generar nombre del archivo
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            output_path = os.path.join(graphs_dir, f"{base_name}.png")

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Gráfica guardada en: {output_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"Error al procesar el archivo: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grafica lecturas de corriente desde archivos CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python graph.py output/mesa_vertical_both.csv
  python graph.py data/test_easyocr.csv -s
  python graph.py /ruta/completa/archivo.csv --save
  python graph.py -h  # Muestra esta ayuda

Con -s/--save, la gráfica se guarda en la carpeta 'graphs/' en alta resolución.

Formato esperado del CSV:
  - Columna 1: Time (ms) - Tiempo en milisegundos
  - Columna 2: Reading - Valor de la lectura (formato decimal)
        """,
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Ruta al archivo CSV (ej: output/datos.csv)",
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Guardar gráfica en carpeta 'graphs/' en lugar de mostrarla",
    )

    args = parser.parse_args()
    plot_csv(args.csv_path, save=args.save)
