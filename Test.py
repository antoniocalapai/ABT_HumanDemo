import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
if torch.backends.mps.is_available():
    torch.device("cpu")
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = False

from mmpose.apis import MMPoseInferencer

# 🖼️ Image path
img_path = '/Users/antoninocalapai/Desktop/250404_HumanTest_2/processed_frames/250404_HumanTest_2_101_20250404143242/frame_0241.png'

# 🚀 Instantiate inferencer using model alias, force CPU
inferencer = MMPoseInferencer('human', device='cpu')

# 🔍 Run inference with visualization
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
