import cv2
import numpy as np
import os
import re

# --- Config ---
home = os.path.expanduser("~")
video_dir = os.path.join(home, "ownCloud/Shared/PriCaB/_HomeCage/250520")
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
video_files.sort()

checkerboard = (14, 9)
square_size_mm = 40.0
cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

def extract_id(name):
    match = re.search(r'Calibration__(\d+)', name)
    return int(match.group(1)) if match else -1

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 255, 0)
thickness = 1
line_type = cv2.LINE_AA

# Create folder for detected frames
detected_dir = os.path.join(video_dir, "detected_frames")
os.makedirs(detected_dir, exist_ok=True)

# --- Process each consecutive pair ---
for i in range(len(video_files) - 1):
    video1_file = video_files[i]
    video2_file = video_files[i + 1]
    video1_path = os.path.join(video_dir, video1_file)
    video2_path = os.path.join(video_dir, video2_file)
    cam1_id = extract_id(video1_file)
    cam2_id = extract_id(video2_file)

    print(f"\n🔧 Processing Camera {cam1_id} + Camera {cam2_id}")
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Could not open one or both video files.")
        continue

    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints = []
    imgpoints1 = []
    imgpoints2 = []

    frame_idx = 0
    total_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("🔚 End of video or failure.")
            break

        frame_idx += 1
        progress = int((frame_idx / total_frames) * 100)
        print(f"🔄 Processing frame {frame_idx}/{total_frames} ({progress}%)", end='\r')

        # Resize and grayscale
        gray1 = cv2.cvtColor(cv2.resize(frame1, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv2.resize(frame2, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        frame1 = cv2.resize(frame1, None, fx=0.5, fy=0.5)
        frame2 = cv2.resize(frame2, None, fx=0.5, fy=0.5)

        # Enhance
        gray1_proc = cv2.convertScaleAbs(gray1, alpha=1.5, beta=30)
        gray2_proc = cv2.convertScaleAbs(gray2, alpha=1.5, beta=30)
        gray1_proc = cv2.GaussianBlur(gray1_proc, (5, 5), 0)
        gray2_proc = cv2.GaussianBlur(gray2_proc, (5, 5), 0)

        # Detect
        found1, corners1 = cv2.findChessboardCornersSB(gray1_proc, checkerboard, cb_flags)
        found2, corners2 = cv2.findChessboardCornersSB(gray2_proc, checkerboard, cb_flags)

        if found1 and found2:
            corners1 = cv2.cornerSubPix(gray1_proc, corners1, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            corners2 = cv2.cornerSubPix(gray2_proc, corners2, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

            # Draw corners
            disp1 = cv2.drawChessboardCorners(frame1.copy(), checkerboard, corners1, found1)
            disp2 = cv2.drawChessboardCorners(frame2.copy(), checkerboard, corners2, found2)

            # Annotate
            label1 = f"{video1_file} | Frame {frame_idx}"
            label2 = f"{video2_file} | Frame {frame_idx}"
            cv2.putText(disp1, label1, (10, disp1.shape[0] - 10), font, font_scale, font_color, thickness, line_type)
            cv2.putText(disp2, label2, (10, disp2.shape[0] - 10), font, font_scale, font_color, thickness, line_type)

            # Save both frames
            fname1 = f"detected_cam{cam1_id:03d}_frame_{frame_idx:04d}.jpg"
            fname2 = f"detected_cam{cam2_id:03d}_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(os.path.join(detected_dir, fname1), disp1)
            cv2.imwrite(os.path.join(detected_dir, fname2), disp2)

            # Save collage
            collage = np.vstack((disp1, disp2))
            collage_name = f"collage_{cam1_id}_{cam2_id}_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(os.path.join(detected_dir, collage_name), collage)

            print(f"💾 Saved detection frames and collage at frame {frame_idx}")

    cap1.release()
    cap2.release()

    print(f"\n📦 Total valid stereo detections for {cam1_id}+{cam2_id}: {len(objpoints)}")

    if len(objpoints) < 3:
        print("❌ Not enough valid detections for stereo calibration.")
        continue

    image_size = gray1.shape[::-1]
    print("⚙️ Running stereo calibration...")
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        None, None, None, None,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    )

    save_path = os.path.join(video_dir, f"stereo_calibration_{cam1_id}_{cam2_id}.yaml")
    fs = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix1", mtx1)
    fs.write("distCoeffs1", dist1)
    fs.write("cameraMatrix2", mtx2)
    fs.write("distCoeffs2", dist2)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.release()

    print(f"✅ Calibration complete. Saved to {save_path}")