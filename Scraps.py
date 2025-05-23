import cv2
import os
import numpy as np

# Input video directory
home = os.path.expanduser("~")
video_dir = os.path.join(home, "ownCloud/Shared/PriCaB/_HomeCage/250523b")

# Output collage folder
output_dir = os.path.join(home, "Desktop", "collages")
os.makedirs(output_dir, exist_ok=True)

# Define 4 specific camera videos
camera_ids = ["101", "102", "113", "126"]
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
video_map = {}

# Match camera ID to video file
for cam_id in camera_ids:
    match = next((f for f in video_files if f"Calibration__{cam_id}" in f), None)
    if not match:
        print(f"❌ Video for camera {cam_id} not found.")
        exit()
    video_map[cam_id] = cv2.VideoCapture(os.path.join(video_dir, match))

# Determine minimum frame count across all cameras
frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in video_map.values()]
min_frames = min(frame_counts)
print(f"📏 Will extract {min_frames} synchronized frames.")

# Process frames
for idx in range(min_frames):
    frames = []
    for cam_id in camera_ids:
        ret, frame = video_map[cam_id].read()
        if not ret:
            print(f"❌ Failed to read frame {idx+1} from camera {cam_id}")
            break
        frame_resized = cv2.resize(frame, (640, 480))  # Resize to fit in collage
        cv2.putText(frame_resized, f"Camera {cam_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frames.append(frame_resized)

    if len(frames) < 4:
        break

    # Build 2x2 collage
    top = np.hstack((frames[0], frames[1]))
    bottom = np.hstack((frames[2], frames[3]))
    collage = np.vstack((top, bottom))

    # Save
    filename = os.path.join(output_dir, f"frame_{idx+1:04d}.jpg")
    cv2.imwrite(filename, collage)
    print(f"💾 Saved {filename}")

# Release video objects
for cap in video_map.values():
    cap.release()

print("✅ Done.")