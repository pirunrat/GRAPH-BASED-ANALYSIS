import cv2
import numpy as np

image = cv2.imread('img.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER_create()
regions, _ = mser.detectRegions(gray)

# Draw regions
for p in regions:
    hull = cv2.convexHull(p.reshape(-1, 1, 2))
    cv2.polylines(image, [hull], True, (0, 255, 0), 2)

cv2.imshow("MSER", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
