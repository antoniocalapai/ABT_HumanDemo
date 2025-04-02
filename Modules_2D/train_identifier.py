import torch
from ultralytics import YOLO
import sys, os
import traceback

def run(model_project, model_name, dataset_path, fine_tune, weights_path, backbone_size, epochs, batch_size, worker_num, output_data, is_running):

    def on_train_epoch_end(trainer):
        output_data.put((trainer.metrics['metrics/accuracy_top1'], trainer.tloss.detach().cpu().numpy(), trainer.metrics['val/loss']))

    try:
        # Load a model
        if fine_tune:
            model = YOLO(weights_path)
        else:
            if backbone_size == 'medium':
                model = YOLO(os.getcwd() + '/models/training_weights/medium.pt')
            elif backbone_size == 'small':
                model = YOLO(os.getcwd() + '/models/training_weights/small.pt')
            else:
                model = YOLO(os.getcwd() + '/models/training_weights/large.pt')

        # Train the model
        model.add_callback("on_fit_epoch_end", on_train_epoch_end)

        res = model.train(project=model_project, name=model_name, data=dataset_path, epochs=epochs, batch=batch_size, imgsz=224, workers=worker_num, device=0, lr0=0.01,
                    optimizer='SGD', pretrained=True, close_mosaic=max(10, int(epochs/10)),
                    degrees=45.0, translate=0.2, shear=0.1, perspective=0.1, flipud=0.5)
        print('Process finished.')
        print(res)
    except Exception as e:
        print("an error occured during training!", e)
        traceback.print_exc()
        
    print("done")
    is_running.value = False

if __name__ == "__main__":

    # check cuda
    device_cuda = torch.cuda.is_available()
    if not device_cuda:
        print('Cuda is not available!')
        sys.exit()

    # ---------------- Parameters ----------------
    model_name = 'Model_Nathan_Vin'  # dataset_folder path contain train and test folders
    dataset_path = os.getcwd() + '/Dataset_Nathan_Vin/'  # dataset_folder path contain train and test folders

    fine_tune = False # True: fine tune a trained identifier, False: train from scratch
    if fine_tune:
        weights_path = os.getcwd() + '/models/identifier/identifier_nathan_vin.pt' # only pt
        backbone_size = ''
    else:
        weights_path = ''
        backbone_size = 'small'  # selectable list: medium, small, large. default: medium
    epochs = 10 # int, default: 100
    batch_size = 64 # int, default: 8

    run(model_name, dataset_path, fine_tune, weights_path, backbone_size, epochs, batch_size)