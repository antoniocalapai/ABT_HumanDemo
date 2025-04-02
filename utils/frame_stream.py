from multiprocessing import Process, Queue, Value
import cv2
import time
import numpy as np

class OnlineStream:
    def __init__(self) -> None:
        self.cap = None

    def from_camera(self, camera):
        self.cap = camera
        self.frame_queue = Queue()

    def start(self, index):
        ret = self.cap.Start_grabbing(index, self.frame_queue)
        if ret != 0:
            raise Exception("camera failed starting to grab!")

    def get_frame_size(self):
        info = self.cap.get_camera_info()
        (self.frame_height, self.frame_width, self.frame_channel) = info.nHeight, info.nWidth, 3
        return (self.frame_width, self.frame_height)
    
    def get_fps(self):
        self.fps_video = 2
        print('video fps: ', self.fps_video)
        
        return self.fps_video

    def get_length(self):
        start_frame = 0
        end_frame = -1
        return (start_frame, end_frame)

    def get_frame(self):
        frame = self.frame_queue.get()
        if frame is None:
            raise Exception("Failed to read frame!")
        return frame
    
    def get_frame_number(self):
        return self.frame_num.value
    
    def release(self):
        ret = self.cap.Stop_grabbing()
        if ret != 0:
            raise Exception("camera failed to stop!")
        # self.frame_queue.put(None)

    def reset(self):
        while self.frame_queue.qsize()>0:
            _ = self.frame_queue.get()
        
class OfflineStream:
    def __init__(self) -> None:
        self.cap = None
        self.frame_queue = Queue()

    def from_video(self, video):
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise Exception("camera failed starting to grab!")
        

    def get_frame_size(self):
        self.frame_width, self.frame_height = int(self.cap.get(3)), int(self.cap.get(4))
        return (self.frame_width, self.frame_height)
    
    def get_fps(self):
        self.fps_video = self.cap.get(cv2.CAP_PROP_FPS)
        print('video fps: ', self.fps_video)
        
        return self.fps_video

    def set_length(self, start_second=-1, end_second=-1):
        start_second = round(start_second)
        end_second = np.floor(end_second)
        video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = self.get_fps()
        if start_second == -1:
            start_frame = 0
        else:
            start_frame = min(max(0, fps_video * start_second), video_length - 2)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        if end_second == -1:
            end_frame = video_length
        else:
            end_frame = max(min(video_length, fps_video * end_second), start_frame + 1)
        self.end_frame = end_frame

    def get_length(self, start_second=-1, end_second=-1):
        start_second = round(start_second)
        end_second = np.floor(end_second)
        video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = self.get_fps()
        if start_second == -1:
            start_frame = 0
        else:
            start_frame = min(max(0, fps_video * start_second), video_length - 2)
        if end_second == -1:
            end_frame = video_length
        else:
            end_frame = max(min(video_length, fps_video * end_second), start_frame + 1)
        
        return (0, end_frame-start_frame)

    def get_frame(self):
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.end_frame:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def release(self):
        ret = self.cap.release()

    def start(self):
        while True:
            f = self.get_frame()
            self.frame_queue.put(f)
            time.sleep(0.1)