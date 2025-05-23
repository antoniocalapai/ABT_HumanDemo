import cv2
import numpy as np
import os

# --- Configuration ---
home = os.path.expanduser("~")
video_path = os.path.join(home, "ownCloud", "Shared", "PriCaB", "_HomeCage", "250523b", "Calibration__101_20250523144844.mp4")
output_dir = os.path.join(home, "Desktop", "manual_annotations_cam101")
board_size = (14, 9)

os.makedirs(output_dir, exist_ok=True)

def annotate_frame(frame, frame_idx):
    clone = frame.copy()
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < board_size[0] * board_size[1]:
                points.append((x, y))
                cv2.circle(clone, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(clone, str(len(points)), (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Annotate", clone)

    cv2.imshow("Annotate", clone)
    cv2.setMouseCallback("Annotate", on_click)

    print(f"🖱️ Frame {frame_idx}: Click {board_size[0] * board_size[1]} corners. Press ESC to skip.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or len(points) == board_size[0] * board_size[1]:
            break

    cv2.destroyAllWindows()

    if len(points) == board_size[0] * board_size[1]:
        filename = f"frame_{frame_idx:04d}.npy"
        save_path = os.path.join(output_dir, filename)
        np.save(save_path, np.array(points, dtype=np.float32).reshape(-1, 1, 2))
        print(f"✅ Saved {len(points)} points to {save_path}")
    else:
        print("⚠️ Incomplete annotation. Frame skipped.")

# --- Open video and process ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video file: {video_path}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    frame_idx += 1
    annotate_frame(frame, frame_idx)

cap.release()
print("✅ Done with all frames.")