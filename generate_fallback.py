import cv2
import numpy as np
import os

if not os.path.exists("static/img"):
    os.makedirs("static/img")

image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(image, "Webcam Disconnected", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
cv2.putText(image, "Make sure no other software is using the camera.", (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(image, "Or you may need to allow site permissions!", (50, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imwrite("static/img/fallback.jpg", image)
print("Created static/img/fallback.jpg")
