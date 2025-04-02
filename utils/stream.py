import cv2
import sys

class Stream:
    def __init__(self) -> None:
        self.cap = None

    def from_video(self, video_file, start_second=-1, end_second=-1):
        source = video_file
        self.cap = cv2.VideoCapture(source)

        if (self.cap.isOpened() == False):
            raise Exception('Error while trying to read video. Please check path again')
        
        else:
            print(video_file,"is open")
        
        self.start_second = start_second
        self.end_second = end_second

    def get_frame_size(self):
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        print('self.frame_width, self.frame_height: ', (self.frame_width, self.frame_height))

        return (self.frame_width, self.frame_height)
    
    def get_fps(self):
        self.fps_video = int(self.cap.get(cv2.CAP_PROP_FPS))
        print('video fps: ', self.fps_video)
        
        return self.fps_video

    def get_length(self):
        video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.start_second == -1:
            start_frame = 0
        else:
            start_frame = min(max(0, self.fps_video * self.start_second), video_length - 2)
        if self.end_second == -1:
            end_frame = video_length
        else:
            end_frame = max(min(video_length, self.fps_video * self.end_second), start_frame + 1)
        
        return (start_frame, end_frame)
    
    def release(self):
        self.cap.release()

    def get_frame(self):
        # print("frmae number :", self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = self.cap.read()
        if not success:
            raise Exception("Failed to read frame!")
        
        return frame