
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from moviepy.editor import ImageSequenceClip

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"{idx:05d}.png")
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()
    return fps

def run_pose_estimation(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import merge_data_samples
    from mmcv import imread
    from mmdet.apis import init_detector, inference_detector

    det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'                      'faster_rcnn_r50_fpn_1x_coco/'                      'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    pose_config = 'configs/body_2d_kpt_sview_rgb_img/topdown_heatmap/'                   'coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/'                       'hrnet_w32_coco_256x192-c78dce93_20200708.pth'

    det_model = init_detector(det_config, det_checkpoint, device='cpu')
    pose_model = init_model(pose_config, pose_checkpoint, device='cpu')

    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for image_path in tqdm(image_paths):
        image = imread(image_path)
        det_results = inference_detector(det_model, image)
        pred_instance = det_results.pred_instances.cpu().numpy()
        person_bboxes = pred_instance.bboxes[pred_instance.labels == 0]

        pose_results = []
        for bbox in person_bboxes:
            pose_result = inference_topdown(pose_model, image, [dict(bbox=bbox)])
            pose_results.append(pose_result[0])

        if pose_results:
            result = merge_data_samples(pose_results)
            pose_model.visualizer.add_datasample(
                name='result',
                image=image,
                data_sample=result,
                draw_gt=False,
                draw_bbox=True,
                show=False,
                out_file=os.path.join(output_dir, os.path.basename(image_path))
            )

def make_collage(input_dir, output_dir, cols=4, rows=4):
    os.makedirs(output_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    group_size = cols * rows
    for i in range(0, len(images), group_size):
        collage_images = images[i:i+group_size]
        imgs = [cv2.imread(img) for img in collage_images]
        h, w = imgs[0].shape[:2]
        collage = np.zeros((h*rows, w*cols, 3), dtype=np.uint8)
        for idx, img in enumerate(imgs):
            if img is None:
                continue
            row = idx // cols
            col = idx % cols
            collage[row*h:(row+1)*h, col*w:(col+1)*w] = img
        collage_path = os.path.join(output_dir, f"{i//group_size:05d}.png")
        cv2.imwrite(collage_path, collage)

def make_video(input_dir, output_path, fps):
    images = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output_path, codec='libx264')

def process_video(video_path, work_dir):
    base = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(work_dir, base, "frames")
    pose_dir = os.path.join(work_dir, base, "pose")
    collage_dir = os.path.join(work_dir, base, "collages")
    output_video = os.path.join(work_dir, base + "_final.mp4")

    fps = extract_frames(video_path, frames_dir)
    run_pose_estimation(frames_dir, pose_dir)
    make_collage(pose_dir, collage_dir)
    make_video(collage_dir, output_video, fps)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <video_path> <work_dir>")
        sys.exit(1)
    video_path = sys.argv[1]
    work_dir = sys.argv[2]
    process_video(video_path, work_dir)
