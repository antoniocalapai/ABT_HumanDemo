import os
import cv2
import glob
import torch
import shutil
from tqdm import tqdm
from mmpose.apis import MMPoseInferencer

# ENV PATCHING to force CPU-only on macOS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if torch.backends.mps.is_available():
    torch.device("cpu")
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = False

# 📁 Path Setup
input_dir = '/Users/antoninocalapai/Desktop/250404_HumanTest_2'
output_dir = os.path.join(input_dir, 'processed_videos')
temp_frames_dir = os.path.join(input_dir, 'temp_frames')
temp_vis_dir = os.path.join(input_dir, 'temp_vis')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_frames_dir, exist_ok=True)
os.makedirs(temp_vis_dir, exist_ok=True)

# 🔁 Get video files
video_paths = sorted(glob.glob(os.path.join(input_dir, '*.mp4')))
print(f"🎬 Found {len(video_paths)} videos.")

# 🧠 Load the model
inferencer = MMPoseInferencer('human', device='cpu')

for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n🎥 Processing: {video_name}")

    # Clean temp folders
    for f in glob.glob(os.path.join(temp_frames_dir, '*')): os.remove(f)
    for f in glob.glob(os.path.join(temp_vis_dir, '*')): os.remove(f)

    # 1️⃣ Extract frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx = 0
    frame_paths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_frames_dir, f'frame_{idx:05d}.png')
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        idx += 1
    cap.release()

    print(f"🖼️ Extracted {len(frame_paths)} frames.")

    # 2️⃣ Run inference per PNG
    for img_path in tqdm(frame_paths, desc="🔍 Inferring keypoints"):
        try:
            result = next(inferencer(img_path))
            vis_frame = result['visualization'][0]
            out_path = os.path.join(temp_vis_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, vis_frame)
        except Exception as e:
            print(f"⚠️ Error on {img_path}: {e}")

    # 3️⃣ Recombine into video
    vis_frames = sorted(glob.glob(os.path.join(temp_vis_dir, '*.png')))
    out_path = os.path.join(output_dir, f"{video_name}_pose.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for vis_frame_path in vis_frames:
        img = cv2.imread(vis_frame_path)
        out.write(img)
    out.release()

    print(f"✅ Saved video: {out_path}")

# 🧹 Clean temp
shutil.rmtree(temp_frames_dir)
shutil.rmtree(temp_vis_dir)

print("\n🎯 All videos processed successfully.")
