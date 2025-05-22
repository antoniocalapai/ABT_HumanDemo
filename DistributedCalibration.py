import cv2
import os
import numpy as np
from tqdm import tqdm

# --- Your Configuration ---
home = os.path.expanduser("~")

video1_path = os.path.join(home, "ownCloud/Shared/PriCaB/_HomeCage/250520/Calibration__113_20250520154614.mp4")
video2_path = os.path.join(home, "ownCloud/Shared/PriCaB/_HomeCage/250520/Calibration__126_20250520154614.mp4")

checkerboard = (10, 15)  # inner corners per chessboard row and column
square_size_mm = 40.0    # real-world size of a square

# --- Prepare object points ---
objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
objp *= square_size_mm

# --- Containers for points ---
objpoints = []
imgpoints1 = []
imgpoints2 = []

# --- Load videos ---
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

frame_count = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

with tqdm(total=frame_count, desc="Processing frames") as pbar:
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        found1, corners1 = cv2.findChessboardCorners(gray1, checkerboard)
        found2, corners2 = cv2.findChessboardCorners(gray2, checkerboard)

        if found1 and found2:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

            vis1 = cv2.drawChessboardCorners(frame1.copy(), checkerboard, corners1, found1)
            vis2 = cv2.drawChessboardCorners(frame2.copy(), checkerboard, corners2, found2)
            combined = np.hstack((vis1, vis2))
            cv2.imshow("Detected Checkerboards", combined)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        pbar.update(1)

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# --- Run stereo calibration ---
ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2,
    None, None, None, None,
    gray1.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC,
    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
)

# --- Save results ---
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

print("✅ Stereo calibration complete. Results saved to stereo_calibration.yaml")