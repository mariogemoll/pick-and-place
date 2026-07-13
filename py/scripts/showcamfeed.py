# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import sys

import cv2

# Usage:
# python showcamfeed.py 0
# python showcamfeed.py 1

camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"Failed to open camera {camera_id}")
    sys.exit(1)

print(f"Showing camera {camera_id}")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow(f"Camera {camera_id}", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()
