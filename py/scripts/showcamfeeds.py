# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import cv2

MAX_CAMERAS = 10

cameras = []

for cam_id in range(MAX_CAMERAS):
    cap = cv2.VideoCapture(cam_id)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            window_name = f"Camera {cam_id}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cameras.append((cam_id, cap, window_name))
            print(f"Opened camera {cam_id}")
        else:
            cap.release()
    else:
        cap.release()

if not cameras:
    print("No cameras found")
    raise SystemExit(1)

print("Press q in any window to quit")

while True:
    for cam_id, cap, window_name in cameras:
        ret, frame = cap.read()
        if ret:
            cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for _, cap, _ in cameras:
    cap.release()

cv2.destroyAllWindows()
