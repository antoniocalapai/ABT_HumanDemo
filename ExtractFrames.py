import os
import cv2
import glob

video_dir = '/Users/antoninocalapai/_local/'
video_paths = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))

# 📂 Output base directory for extracted frames
output_base = os.path.join(video_dir, 'frames')
os.makedirs(output_base, exist_ok=True)

for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_output_dir = os.path.join(output_base, video_name)
    os.makedirs(frame_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_idx = 0

    print(f"Extracting frames from {video_name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= 60:  # Skip the first 60 frames
            frame_filename = os.path.join(frame_output_dir, f'frame_{saved_idx:05d}.png')
            cv2.imwrite(frame_filename, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved_idx} frames (after skipping 60) to: {frame_output_dir}")
