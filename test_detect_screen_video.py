from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture("input.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ejecuta detección
    results = model(frame)

    for r in results:
        for box in r.boxes.xyxy:  # coordenadas (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box)
            pantalla = frame[y1:y2, x1:x2]

            # Aquí puedes guardar, mostrar o pasar a OCR
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Pantalla detectada", pantalla)

    cv2.imshow("Video con detección", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
