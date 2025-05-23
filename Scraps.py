import cv2
import numpy as np
import os

# --- Paths ---
home = os.path.expanduser("~")
video1_path = os.path.join(home, "ownCloud/Shared/PriCaB/_HomeCage/250520/Calibration__113.mp4")
video2_path = os.path.join(home, "ownCloud/Shared/PriCaB/_HomeCage/250520/Calibration__102.mp4")

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

if not cap1.isOpened() or not cap2.isOpened():
    print("❌ Could not open one or both video files.")
    exit()

# --- Checkerboard config ---
checkerboard = (14, 9)
square_size_mm = 40.0
cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

# --- Prepare object points ---
objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
objp *= square_size_mm

# --- Storage for calibration data ---
objpoints = []
imgpoints1 = []
imgpoints2 = []

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        print("🔚 End of video or failure.")
        break

    # Resize and convert
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, None, fx=0.5, fy=0.5)
    gray2 = cv2.resize(gray2, None, fx=0.5, fy=0.5)
    frame1 = cv2.resize(frame1, None, fx=0.5, fy=0.5)
    frame2 = cv2.resize(frame2, None, fx=0.5, fy=0.5)

    disp1 = frame1.copy()
    disp2 = frame2.copy()

    # Enhance
    gray1_proc = cv2.convertScaleAbs(gray1, alpha=1.5, beta=30)
    gray2_proc = cv2.convertScaleAbs(gray2, alpha=1.5, beta=30)
    gray1_proc = cv2.GaussianBlur(gray1_proc, (5, 5), 0)
    gray2_proc = cv2.GaussianBlur(gray2_proc, (5, 5), 0)

    # Try detection
    found1, corners1 = cv2.findChessboardCorners(gray1_proc, checkerboard, cb_flags)
    found2, corners2 = cv2.findChessboardCorners(gray2_proc, checkerboard, cb_flags)

    if (not found1 or not found2) and hasattr(cv2, 'findChessboardCornersSB'):
        if not found1:
            found1, corners1 = cv2.findChessboardCornersSB(gray1_proc, checkerboard, cb_flags)
        if not found2:
            found2, corners2 = cv2.findChessboardCornersSB(gray2_proc, checkerboard, cb_flags)

    # Refine if found
    if found1:
        corners1 = cv2.cornerSubPix(gray1_proc, corners1, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        disp1 = cv2.drawChessboardCorners(frame1.copy(), checkerboard, corners1, found1)

    if found2:
        corners2 = cv2.cornerSubPix(gray2_proc, corners2, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        disp2 = cv2.drawChessboardCorners(frame2.copy(), checkerboard, corners2, found2)

    # Display
    combined = np.vstack((np.hstack((cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR), disp1)),
                          np.hstack((cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR), disp2))))
    cv2.imshow("Grayscale vs Detected Corners (Cam1 top, Cam2 bottom)", combined)

    # --- Wait for keystroke only if BOTH views detected ---
    if found1 and found2:
        print("🛑 Checkerboard detected in both cameras. Press 's' to save, any key to skip, 'q' to quit.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)
            print("💾 Frame pair saved.")
        elif key == ord('q'):
            break
    else:
        # Keep advancing quickly
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

print(f"\n📦 Collected {len(objpoints)} stereo frame pairs for calibration.")

# --- Run stereo calibration ---
if len(objpoints) < 3:
    print("❌ Not enough valid pairs for calibration.")
else:
    image_size = gray1.shape[::-1]
    print("⚙️ Running stereo calibration...")
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        None, None, None, None,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    )

    # Save results
    fs = cv2.FileStorage("stereo_calibration.yaml", cv2.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix1", mtx1)
    fs.write("distCoeffs1", dist1)
    fs.write("cameraMatrix2", mtx2)
    fs.write("distCoeffs2", dist2)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.release()

    print("✅ Calibration complete. Results saved to stereo_calibration.yaml")