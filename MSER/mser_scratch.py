import cv2
import numpy as np

def naive_mser(gray, delta=5, min_area=100, max_area=10000):
    regions = []
    visited = np.zeros_like(gray, dtype=np.uint8)

    for thresh in range(0, 255 - delta, delta):
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(cnt)

                # Prevent overlapping regions
                region_mask = np.zeros_like(gray)
                cv2.drawContours(region_mask, [cnt], -1, 255, -1)
                overlap = cv2.bitwise_and(region_mask, visited)

                if np.count_nonzero(overlap) == 0:
                    regions.append((x, y, w, h))
                    visited = cv2.bitwise_or(visited, region_mask)

    return regions

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        regions = naive_mser(gray)

        for (x, y, w, h) in regions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Real-Time MSER Approximation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
