import os
import glob
import cv2
import torch
from mmpose.apis import MMPoseInferencer

# ---------------------------------------
# 🔧 Environment setup to disable GPU/MPS
# ---------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

if torch.backends.mps.is_available():
    torch.device("cpu")
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = False

# ---------------------------------------
# 📂 Input parent directory containing all video folders
# ---------------------------------------
base_input_dir = '/Users/antoninocalapai/_local/frames'

# ---------------------------------------
# 🚀 Instantiate the inferencer
# ---------------------------------------
inferencer = MMPoseInferencer('human', device='cpu')

# ---------------------------------------
# 🔁 Process each folder in base_input_dir
# ---------------------------------------
all_folders = sorted([f for f in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, f))])

for folder in all_folders:
    frame_dir = os.path.join(base_input_dir, folder)
    output_dir = os.path.join(frame_dir, 'with_keypoints')
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.png')))
    print(f"\n📂 Processing folder: {folder} — {len(frame_paths)} frames")

    for i, img_path in enumerate(frame_paths):
        try:
            result_generator = inferencer(img_path, show=False, out_dir=output_dir, radius=14, thickness=8)
            result = next(result_generator)
        except Exception as e:
            print(f"⚠️ Error on frame {i}: {e}")
        finally:
            os.remove(img_path)