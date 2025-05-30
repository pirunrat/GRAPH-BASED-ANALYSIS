import cv2
import numpy as np
import matplotlib.pyplot as plt


def mser_sift_hybrid(image_path):
    # Load grayscale image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: MSER region detection
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    # Convert regions to keypoints
    keypoints = []
    for region in regions:
        for x, y in region:
            kp = cv2.KeyPoint(float(x), float(y), 5)  # ✅ Fix here
            keypoints.append(kp)

    # Step 2: Remove duplicates (optional)
    pts = cv2.KeyPoint_convert(keypoints)
    unique_pts = np.unique(pts, axis=0)
    keypoints = [cv2.KeyPoint(float(x), float(y), 5) for x, y in unique_pts]

    # Step 3: Compute SIFT descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray, keypoints)

    print(f"🟢 MSER found {len(regions)} regions")
    print(f"🟡 SIFT computed {len(keypoints)} descriptors")

    # Visualization
    img_out = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(10, 6))
    plt.title("MSER + SIFT Hybrid Features")
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return keypoints, descriptors


# === Usage ===
if __name__ == "__main__":
    image_path = "./lena.png"  # Replace with your image
    mser_sift_hybrid(image_path)
