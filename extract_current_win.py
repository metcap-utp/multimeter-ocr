"""
extract_current.py
==================
Extracts current readings from a clamp‑meter display (Fluke 376 FC) that were captured on video
and writes them to a CSV file with 10 readings per second.

Usage
-----
python extract_current.py --video "input.mp4" --csv "output.csv"

Requires:
    * Python 3.8+
    * opencv‑python (cv2)
    * pytesseract  +  Tesseract‑OCR engine installed and in PATH

The script is calibrated for the sample frame provided; if the camera position changes
significantly you can fine‑tune the `ROI` values below.
"""

import argparse
from pathlib import Path
import cv2
import pytesseract
import csv
import math

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# --------------------------- helper functions ---------------------------
def digits_from_frame(frame, roi, thresh_block=11):
    """
    Crop the display region, run OCR, and return the numeric value as float.
    Returns None if OCR fails.
    """
    x, y, w, h = roi
    crop = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # improve contrast & threshold
    gray = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        thresh_block,
        2,
    )

    # OCR – whitelist digits and decimal point
    config = "--psm 7 -c tessedit_char_whitelist=0123456789."
    txt = pytesseract.image_to_string(thresh, config=config).strip()

    # keep only first number found
    try:
        val = float(txt)
    except ValueError:
        return None
    return val


def round_sig(x, sig=2):
    if x == 0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


# ------------------------------ main script -----------------------------
def main(args):
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"No such video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise RuntimeError("Cannot read FPS from the video.")

    # step to hit ~10 Hz
    step = max(1, round(fps / 10.0))

    # ROI (x, y, w, h) – calibrated from sample image
    ROI = (150, 690, 270, 90)  # adjust if necessary

    readings = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            t = frame_idx / fps
            val = digits_from_frame(frame, ROI)
            if val is not None:
                val = round_sig(val, 2)
                readings.append((t, val))

        frame_idx += 1

    cap.release()

    # write CSV
    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tiempo_s", "corriente_A"])
        writer.writerows(readings)

    print(f"Done! {len(readings)} rows written to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract current readings from video."
    )
    parser.add_argument("--video", required=True, help="Path to input .mp4")
    parser.add_argument("--csv", required=True, help="Path to output .csv")
    args = parser.parse_args()
    main(args)
