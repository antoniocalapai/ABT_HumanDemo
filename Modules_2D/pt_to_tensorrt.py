import torch
from ultralytics import YOLO
import sys
def run(model_path, task):

    if task == 'box_pose':
        _task = 'pose'
        imgsz = 1280
        half = True
    else:
        _task = 'classify'
        imgsz = 224
        half = True

    # Load a model
    model = YOLO(model_path, task=_task)

    # Export the model
    model.export(format='engine', device=0, imgsz=imgsz, half=half)
    print('Process finished.')

if __name__ == "__main__":

    # check cuda
    device_cuda = torch.cuda.is_available()
    if not device_cuda:
        print('Cuda is not available!')
        sys.exit()

    # ---------------- Parameters ----------------
    model_path = './models/identifier/identifier_nathan_vin.pt'  # only pt
    # './models/box_pose_detector/monkey_box_pose.pt'
    # './models/identifier/identifier_nathan_vin.pt'
    model_task = 'identification'  # box_pose or identification

    run(model_path, model_task)