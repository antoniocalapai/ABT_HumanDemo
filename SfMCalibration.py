import os
import cv2
from moviepy.editor import VideoFileClip

# --- Config ---
input_folder = os.path.expanduser("~/ownCloud/Shared/PriCaB/_HomeCage/250519")
clipped_folder = os.path.join(input_folder, "clipped")
sfm_folder = os.path.join(clipped_folder, "sfm_images")
start_time = 5    # seconds
end_time = 15     # seconds
frame_number = 1  # which frame to extract for SfM

# --- Ensure folders exist ---
os.makedirs(clipped_folder, exist_ok=True)
os.makedirs(sfm_folder, exist_ok=True)

# --- Step 1: Clip videos to 5–15s ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(clipped_folder, f"clip_{filename}")

        print(f"🎬 Clipping {start_time}-{end_time}s from {filename}...")

        try:
            clip = VideoFileClip(input_path).subclip(start_time, end_time)
            clip.write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)
            print(f"✅ Saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed on {filename}: {e}")

# --- Step 2: Extract frame from each clipped video ---
for filename in sorted(os.listdir(clipped_folder)):
    if filename.endswith(".mp4"):
        video_path = os.path.join(clipped_folder, filename)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if ret:
            out_name = os.path.splitext(filename)[0] + ".jpg"
            out_path = os.path.join(sfm_folder, out_name)
            cv2.imwrite(out_path, frame)
            print(f"✅ Saved frame {frame_number} of {filename} → {out_path}")
        else:
            print(f"❌ Failed to read frame {frame_number} from {filename}")
        cap.release()