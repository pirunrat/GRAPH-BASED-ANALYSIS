import cv2
import numpy as np

def preprocess(frame):
    """Convert to grayscale and apply thresholding."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    return binary

def detect_objects(binary):
    """Apply connected component labeling."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
    return num_labels, labels, stats, centroids

def draw_boxes(frame, stats):
    """Draw bounding boxes for each component (skip background)."""
    for i in range(1, len(stats)):  # skip label 0 (background)
        x, y, w, h, area = stats[i]
        if area > 100:  # filter small blobs
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {i}', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        binary = preprocess(frame)
        num_labels, labels, stats, centroids = detect_objects(binary)
        draw_boxes(frame, stats)

        cv2.imshow("Real-Time Detection", frame)
        cv2.imshow("Binary Mask", binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
