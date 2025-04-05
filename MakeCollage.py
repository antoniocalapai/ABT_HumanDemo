import os
import glob
import cv2
import numpy as np

# ---------------------------------------
# 🔧 Setup
# ---------------------------------------
base_input_dir = '/Users/antoninocalapai/_local/frames'
all_folders = sorted([f for f in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, f))])
grid_size = 4  # 4x4 layout
target_size = (1920, 1080)
fps = 20

# 📁 Collage output directory
collage_output_dir = os.path.join(base_input_dir, 'collage_frames')
os.makedirs(collage_output_dir, exist_ok=True)

print("\n🧩 Starting collage creation...")

# Step 1: Collect all 'with_keypoints/visualizations' folders
keypoint_dirs = [os.path.join(base_input_dir, f, 'with_keypoints', 'visualizations') for f in all_folders]
frame_lists = []
for kp_dir in keypoint_dirs:
    frames = sorted(glob.glob(os.path.join(kp_dir, 'frame_*.png')))
    if frames:
        frame_lists.append(frames)
    else:
        print(f"⚠️ Skipping empty folder: {kp_dir}")

# Step 2: Pad missing video folders
expected_videos = grid_size * grid_size
if len(frame_lists) < expected_videos:
    print(f"⚠️ Only {len(frame_lists)} videos found, padding with empty placeholders.")
    while len(frame_lists) < expected_videos:
        frame_lists.append([])

# Step 3: Find smallest frame count to sync all videos
min_frame_count = min(len(f) for f in frame_lists if f)
print(f"🧮 Minimum synchronized frames: {min_frame_count}")

# Step 4: Get size from first frame
first_valid = next(f for f_list in frame_lists if f_list for f in f_list)
h, w, _ = cv2.imread(first_valid).shape

# Step 5: Generate collage PNGs
for i in range(min_frame_count):
    frames = []
    for vid_idx in range(expected_videos):
        if i < len(frame_lists[vid_idx]):
            frame = cv2.imread(frame_lists[vid_idx][i])
        else:
            frame = 255 * np.ones((h, w, 3), dtype=np.uint8)
        frames.append(frame)

    # Build collage grid
    grid_rows = []
    for row in range(grid_size):
        row_imgs = frames[row * grid_size:(row + 1) * grid_size]
        if len(row_imgs) < grid_size:
            row_imgs += [255 * np.ones((h, w, 3), dtype=np.uint8)] * (grid_size - len(row_imgs))
        grid_rows.append(cv2.hconcat(row_imgs))
    collage = cv2.vconcat(grid_rows)

    # Save PNG
    collage_path = os.path.join(collage_output_dir, f'collage_{i:04d}.png')
    cv2.imwrite(collage_path, collage)
    print(f"🖼️ Saved collage frame: {collage_path}")

print("\n🎬 Creating final collage video...")

# Step 6: Compile PNGs into video with resize
collage_frames = sorted(glob.glob(os.path.join(collage_output_dir, 'collage_*.png')))
video_path = os.path.join(base_input_dir, 'video_collage.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, target_size)

for frame_path in collage_frames:
    img = cv2.imread(frame_path)
    resized = cv2.resize(img, target_size)
    out.write(resized)

out.release()
print(f"✅ Final collage video saved at: {video_path}")
