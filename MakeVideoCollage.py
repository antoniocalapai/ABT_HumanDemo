import subprocess
import os
import glob

# Path to the directory containing PNGs
collage_dir = '/Users/antoninocalapai/_local/frames/collage_frames'
output_video = '/Users/antoninocalapai/_local/frames/video_collage_ffmpeg.mp4'

# Make sure the path exists
if not os.path.exists(collage_dir):
    raise FileNotFoundError(f"❌ Directory not found: {collage_dir}")

# Count total number of frames
frame_paths = sorted(glob.glob(os.path.join(collage_dir, 'collage_*.png')))
num_frames = len(frame_paths)
fps = 4.7
duration = num_frames / fps
trimmed_duration = max(0, duration - 10)

# Build ffmpeg command
cmd = [
    'ffmpeg',
    '-y',
    '-framerate', str(fps),
    '-pattern_type', 'glob',
    '-i', os.path.join(collage_dir, 'collage_*.png'),
    '-t', str(trimmed_duration),  # <-- Limit the output duration
    '-vf', 'scale=2048:1500',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    output_video
]

# Run ffmpeg
print(f"🚀 Running ffmpeg to create video (trimmed to {trimmed_duration:.2f} seconds)...")
subprocess.run(cmd, check=True)
print(f"✅ Video saved at: {output_video}")
