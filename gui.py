
def rotation_x(tetha):
    tetha= np.radians(tetha)
    return [[1,0,0],[0,np.cos(tetha),-np.sin(tetha)],[0,np.sin(tetha),np.cos(tetha)]]
def rotation_y(tetha):
    tetha= np.radians(tetha)
    return [[np.cos(tetha),0,np.sin(tetha)],[0,1,0],[-np.sin(tetha),0,np.cos(tetha)]]
def rotation_z(tetha):
    tetha= np.radians(tetha)
    return [[np.cos(tetha),-np.sin(tetha),0],[np.sin(tetha),np.cos(tetha),0],[0,0,1]]

def streaming(video, start, end, input_queue, is_running):
    from utils.frame_stream import OfflineStream
    import time
    stream = OfflineStream()
    stream.from_video(video)
    stream.set_length(start, end)
    while is_running.value:
        if input_queue.qsize()>10:
            time.sleep(0.01)
            continue
        f = stream.get_frame()
        if f is None:
            input_queue.put(None)
            break
        input_queue.put(f)
        time.sleep(0.01)
    if not is_running.value:
        while input_queue.qsize() >0:
            _ = input_queue.get()
    stream.release()

def write_video(out_queue, path, fps, width, height, save_frames):
    import cv2
    import time
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(path, fourcc, fps, (width, height)) # ,(cv2.VIDEOWRITER_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)  
    sleep_time = (1 / fps) / 2
    while True:
        if out_queue.qsize()>0:
            img = out_queue.get()
            if img is None:
                video_out.release()
                break
            if isinstance(img, str):
                continue
            print(width, height, img.shape)
            if width!=img.shape[1] or height!=img.shape[0]:
                img = cv2.resize(img, (width, height))
            video_out.write(img)
            save_frames.value+=1
        else:
            time.sleep(sleep_time)

def write_video2(out_queue, path, fps, save_frames, is_done):
    import time
    import cv2
    defined_writer = 0
    sleep_time = (1 / fps) / 2
    while True:
        queue_s = out_queue.qsize()
        # print(ip+" queue_size: "+str(queue_s))
        if queue_s > 0:
            frame = out_queue.get()
            print("=========== save new frame ===========")
            if frame is None: # None signals the end of frames, so break the loop
                break
            elif defined_writer == 0:
                frame_height, frame_width, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # FFV1 XVID mp4v
                out = cv2.VideoWriter(path, fourcc, fps, (frame_width, frame_height))
                defined_writer = 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
            save_frames.value+=1
        else:
            time.sleep(sleep_time)
    if defined_writer == 1:
        out.release()
    is_done.value = True

def write_frames(out_queue, path, save_frames, fps, is_done):
    counter = 0
    sleep_time = (1 / fps) / 2
    while True:
        queue_s = out_queue.qsize()
        # print(ip+" queue_size: "+str(queue_s))
        if queue_s > 0:
            frame =out_queue.get()
            if frame is None: # None signals the end of frames, so break the loop
                break
            counter += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path + "/frame_" + str(counter) + ".bmp", frame)
            save_frames.value+=1
        else:
            time.sleep(sleep_time)
    is_done.value = True

def inference(file, classes, device=0):
    from ultralytics import YOLO
    import numpy as np
    try:

        classifier = YOLO(file, task='classify')
        imgsz_classifier = 224
        results = classifier.predict(np.zeros((imgsz_classifier, imgsz_classifier, 3)), device=device, task='classify', imgsz=imgsz_classifier, verbose=False)
        result = results[0]
        result = result.to('cpu')
        for l in result.names.values():
            classes.put(l)
            
    except:
        pass
    classes.put(None)
    return

def total_length_of_lists_in_queue(q):
    total_length = 0
    with q.mutex:
        for item in list(q.queue):
            total_length += len(item)
    return total_length

def validate_time(time_str):
    try:
        datetime.strptime(time_str, "%H:%M:%S")
        return True
    except ValueError:
        return False

class App:
    def __init__(self) -> None:
        self.step = 0
        w = screen.get_width()
        h = screen.get_height()
        offset_x = w//50
        offset_y = h//25
        y = h//20
        x = w//6

        self.menu_train_btn = Button("Model Training", w/4, h/10, (2*w//24, (h-4*h//10-3*h//20)//2), func=self.__select_option, option=20)
        self.menu_train_hint_tk = Toolkit(screen, 2*w//24, (h-4*h//10-3*h//20)//2+h//10, text="Train an identification model to identify monkeys form each other.")
        self.menu_quantize_btn = Button("Model Optimization", w/4, h/10, (2*w//24, (h-4*h//10-3*h//20)//2+h//10+h//20), func=self.__select_option, option=10)
        self.menu_quantize_hint_tk = Toolkit(screen, 2*w//24, (h-4*h//10-3*h//20)//2+h//10+h//20+h//10, text="Optimze models by quantizing pytorch models to tensorRT in order to improve the infrence speed on Jetson.")
        self.menu_box_pose_id_btn = Button("2D Prediction", w/4, h/10, (2*w//24, (h-4*h//10-3*h//20)//2+2*(h//10+h//20)), func=self.__select_option, option=30)
        self.menu_box_pos_id_hint_tk = Toolkit(screen, 2*w//24, (h-4*h//10-3*h//20)//2+2*(h//10+h//20)+h//10, "Detect and Identify monkeys in an arbitrary video input.")
        self.menu_jarvis_btn = Button("3D Calibration (JARVIS)", w/4, h/10, (9*w//24, (h-4*h//10-3*h//20)//2), func=self.__open_jarvis)
        self.menu_jarvis_hint_tk = Toolkit(screen, 9*w//24, (h-4*h//10-3*h//20)//2+h//10, "Open JARVIS to do calibration phase.")
        self.menu_jarvis_res_lbl = Label(9*w//24+50, (h-4*h//10-3*h//20)//2+h//10, color=(200,10,20))
        self.menu_calibration_btn = Button("3D Transformation", w/4, h/10, (9*w//24, (h-4*h//10-3*h//20)//2+h//10+h//20), func=self.__select_option, option=50)
        self.menu_calibration_hint_tk = Toolkit(screen, 9*w//24, (h-4*h//10-3*h//20)//2+h//10+h//20+h//10, text="Enter the camera's exact position and information for each pair of cameras to calculate the transformation matrix for each pair in order to convert their output to a global coordinate system. (repeat this step for each pair)")
        self.menu_localization_btn = Button("3D Localization", w/4, h/10, (9*w//24, (h-4*h//10-3*h//20)//2+2*(h//10+h//20)), func=self.__select_option, option=60)
        self.menu_localization_hint_tk = Toolkit(screen, 9*w//24, (h-4*h//10-3*h//20)//2+2*(h//10+h//20)+h//10, text="Select 2D keypoints data for different camera pairs and an unified 3D model will be generated and visualized.")        
        self.menu_visualization_btn = Button("3D Visualization", w/4, h/10, (9*w//24, (h-4*h//10-3*h//20)//2+3*(h//10+h//20)), func=self.__select_option, option=70)
        self.menu_visualization_hint_tk = Toolkit(screen, 9*w//24, (h-4*h//10-3*h//20)//2+3*(h//10+h//20)+h//10, text="Visualize 3D result from a reconstructed 3D data stored in a text file")
        self.menu_camera_btn = Button("Camera", w/4, h/10, (16*w//24, (h-4*h//10-3*h//20)//2), func=self.__select_option, option=80)
        self.menu_camera_hint_tk = Toolkit(screen, 16*w//24, (h-4*h//10-3*h//20)//2+h//10, "Store stream of synced GIGE cameras.")
        self.menu_frame_manager_btn = Button("Interaction with Jetsons", w/4, h/10, (16*w//24, (h-4*h//10-3*h//20)//2+h//10+h//20), func=self.__select_option, option=90)
        self.menu_frame_manager_tk = Toolkit(screen, 16*w//24, (h-4*h//10-3*h//20)//2+h//10+h//20+h//10, "Send videos to Jetsons and get results back.")

        self.rect_2d = pygame.Rect(2*w//24-w/50, (h-4*h//10-3*h//20)//2-2*h/20, w/4+2*w/50, 4*h/10+6*h/20)
        self.rect_3d = pygame.Rect(9*w//24-w/50, (h-4*h//10-3*h//20)//2-2*h/20, w/4+2*w/50, 4*h/10+6*h/20) 
        self.rect_cam = pygame.Rect(16*w//24-w/50, (h-4*h//10-3*h//20)//2-2*h/20, w/4+2*w/50, 4*h/10+6*h/20)        
        self.text_2d = pygame.font.Font(None, 36).render("2D POSE ESTIMATION", True, (0,0,0))
        self.r_2d = self.text_2d.get_rect()
        self.r_2d.width+=20
        self.r_2d.x = 2*w//24-w/50 + (w/4+2*w/50-self.r_2d.width)//2
        self.r_2d.y = (h-4*h//10-3*h//20)//2-2*h/20 - self.r_2d.height//2
        self.text_3d = pygame.font.Font(None, 36).render("3D TRIANGULATION", True, (0,0,0))
        self.r_3d = self.text_3d.get_rect()
        self.r_3d.width+=20
        self.r_3d.x = 9*w//24-w/50 + (w/4+2*w/50-self.r_3d.width)//2
        self.r_3d.y = (h-4*h//10-3*h//20)//2-2*h/20 - self.r_3d.height//2
        self.text_cam = pygame.font.Font(None, 36).render("CAMERA", True, (0,0,0))
        self.r_cam = self.text_cam.get_rect()
        self.r_cam.width+=20
        self.r_cam.x = 16*w//24-w/50 + (w/4+2*w/50-self.r_cam.width)//2
        self.r_cam.y = (h-4*h//10-3*h//20)//2-2*h/20 - self.r_cam.height//2
        
        self.back_to_menu_btn = Button("Back to Menu", w//10, y, (w//20, h-h//10), func=self.__wait_for_back)

        ## quantization
        self.cuda_status_lbl = Label(w//20, y)
        self.task_lbl = Label(w/10+w/50, h/10+offset_y, text="Select Model Type:")
        self.task_lst = DropDown(w/10+w/50, h/10+offset_y+offset_y, w/5-2*w/50, y, ["box_pose", "identification"])
        self.browse_pt_model_btn = Button("Select PyTorch Model", w/5-2*w/50, y, (w/10+w/5+w/50, h/10+offset_y+offset_y), func=self.__select_pt_model)
        self.pt_model_lbl = Label(w/10+w/5+w/50, h/10+offset_y+offset_y+y,w=2*w/5-50)
        self.quantize_btn = Button("Optimize", w/5-2*w/50, y, (w/10+w/50+3*w/5, h/10+offset_y+offset_y), func=self.__wait_for_process)
        self.quantize_done_lbl = Label(w/10+w/50, h/10+2*offset_y+4*y, w=4*w/5)
        
        ## train
        self.model_name_inp = InputBox(w/10+w/50, h/10+offset_y, w/5-2*w/50, y)
        self.model_name_lbl = Label(w/10+w/50, h/10, text="Model Name:")
        self.model_path_btn = Button("Select Output Path", w/5-2*w/50, y, (w/10+w/5+w/50, h/10+offset_y), func=self.__model_path)
        self.model_path_lbl = Label(w/10+w/5+w/50, h/10+offset_y+y)
        self.browse_dataset_btn = Button("Select Dataset", w/5-2*w/50, y, (w/10+w/50, 3*h/10), func=self.__browse_dataset)
        self.browse_dataset_lbl = Label(w/10+w/50, 3*h/10+y, w=4*w/5)
        self.browse_dataset_hink_tk = Toolkit(screen, w/5-2*w/50+w/10+w/50, 3*h/10, text="A folder with two subfolders for 'train' and 'test' data. In each, there must be an individual folder for each monkey with adequate number of images.")
        self.backbone_size_res_lbl = Label(w/10+w/50, 5*h/10+offset_y+offset_y)
        self.backbone_size_rb = RadioButton(screen, w/10+w/50, 5*h/10+offset_y, 2*w/5-2*w/50, y, ["small","medium","large"], default=1, func=self.__check_model_exist)
        self.backbone_size_lbl = Label(w/10+w/50, 5*h/10, text="Backbone Size:")
        self.backbone_size_hint_tk = Toolkit(screen, self.backbone_size_lbl.x+self.backbone_size_lbl.get_width()+10, 5*h/10, text="The size of backbone. More larger backbone results in more accurate but heavier network.")
        self.fine_tune_ckb = Checkbox(screen, w/10+2*w/5+w/50, 5*h/10, caption='Fine-tune', func=self.__enable_finetune)
        self.browse_pretrained_weight_btn = Button("Load Pretrained Model", w/5-2*w/50, y, (w/10+2*w/5+w/50, 5*h/10+offset_y), clickable=False, func=self.__load_weight)
        self.browse_pretrained_weight_lbl = Label(w/10+2*w/5+w/50, 5*h/10+offset_y+y, w=2*w/5)
        self.epoch_inp = InputBox(w/10+w/50, 7*h/10+offset_y, w/5-2*w/50, y, text="100")
        self.epoch_lbl = Label(w/10+w/50,7*h/10,text="Number of Epochs:")
        self.batch_inp = InputBox(w/10+w/5+w/50, 7*h/10+offset_y, w/5-2*w/50, y, text="12")
        self.batch_lbl = Label(w/10+w/5+w/50, 7*h/10, text="Batch Size:")
        self.worker_inp = InputBox(w/10+2*w/5+w/50, 7*h/10+offset_y, w/5-2*w/50, y, text="12")
        self.worker_lbl = Label(w/10+2*w/5+w/50, 7*h/10, text="Number of Workers:")
        self.train_btn = Button("Train", x, y, (w-x-w//20, h-h//10), func=self.__train)

        ## predict box, pose and id
        # online page
        self.search_camera_btn = Button("Search Cameras", w/6-2*w/50, y, (w/50, y+2*offset_y), func=self.__search_camera)
        self.no_camera_lbl = Label(w/50, y+offset_y, w = w/6-2*w/50, color=(200, 50, 0))
        self.trigger_option_lst = DropDown(w/6+w/50, y+2*offset_y, w/6-2*w/50, y, options=["Software Triggerig", "PTP"], default=0, enable=False)
        self.trigger_option_lbl = Label(w/6+w/50, y+offset_y, text="Sync Methods:")
        self.camera_result_lst = List(w/50, h/10+y+2*offset_y, w=w/6-2*w/50, h=y, height=h-2*h//10-2*y-2*offset_y, options=[], enable=True, func=self.__select_camera)
        self.select_camera_lbl = Label(w/50, h/10+y+offset_y, w = w/6-2*w/50, text="Select Camera:")
        self.open_cameras_btn = Button("Open Cameras", w/6-2*w/50, y, (w/6+w/50, 4*y+2*offset_y), clickable=False, func=self.__wait_for_process)
        self.specific_param_lbl = Label(3*w/6+w/50, y, text="Camera Specific Parameters")
        self.vertical_flip_ckb = Checkbox(screen, 3*w/6+w/50, 3*h/10+y+2*offset_y, enable=False, caption="Flip Vertically", func=self.__reset_list)
        self.horizental_flip_ckb = Checkbox(screen, 3*w/6+w/50, 3*h/10+2*y+2*offset_y, enable=False, caption="Flip Horizentally", func=self.__reset_list)
        self.apply_flip_lbl = Label(3*w/6+w/50, 5*h/10+y+offset_y, w=w/6+w/50, text="Apply to:")
        self.apply_flip_lst = DropDown(3*w/6+w/50, 5*h/10+y+2*offset_y, w/6-2*w/50, y, options=["None", "All"], default=1, enable=False, func=self.__set_camera_specific_parameters, scrollable=True, height=250)
        self.exposure_inp = InputBox(3*w/6+w/50, y+2*offset_y, w/6-2*w/50, y, func=self.__reset_list)
        self.auto_exposure_ckb = Checkbox(screen, 3*w/6+w/50, 2*y+2*offset_y, caption="Auto Exposure", enable=False, default=True, func=self.__auto_exposure)
        self.exposure_lbl = Label(3*w/6+w/50, y+offset_y, text="Exposure Time: (micro sec)")
        self.gain_inp = InputBox(3*w/6+w/50, h/10+y+3*offset_y, w/6-2*w/50, y, func=self.__reset_list)
        self.auto_gain_ckb = Checkbox(screen, 3*w/6+w/50, h/10+2*y+3*offset_y, caption="Auto Gain", enable=False, default=True, func=self.__auto_gain)
        self.gain_lbl = Label(3*w/6+w/50, h/10+y+2*offset_y, text="Gain:")
        self.general_param_lbl = Label(2*w/6+w/50, y, text="General Parameters")
        self.frame_rate_inp = InputBox(2*w/6+w/50, y+2*offset_y, w/6-2*w/50, y)
        self.frame_rate_lbl = Label(2*w/6+w/50, y+offset_y, text="Frame Rate:")
        self.binning_rb = RadioButton(screen, 2*w/6+w/50, h/10+y+2*offset_y, w/6-2*w/50, y, options=["1","2","4"], default=1)
        self.binning_lbl = Label(2*w/6+w/50, h/10+y+offset_y, text="Binning:")
        self.action_device_key_inp = InputBox(2*w/6+w/50, 2*h/10+y+2*offset_y, w/6-2*w/50, y)
        self.action_device_key_lbl = Label(2*w/6+w/50, 2*h/10+y+offset_y, text="Action Device Key:")
        self.action_group_key_inp = InputBox(2*w/6+w/50, 3*h/10+y+2*offset_y, w/6-2*w/50, y)
        self.action_group_key_lbl = Label(2*w/6+w/50, 3*h/10+y+offset_y, text="Action Group Key:")
        self.action_group_mask_inp = InputBox(2*w/6+w/50, 4*h/10+y+2*offset_y, w/6-2*w/50, y)
        self.action_group_mask_lbl = Label(2*w/6+w/50, 4*h/10+y+offset_y, text="Action Group Mask:")
        self.save_option_rb = RadioButton(screen, 4*w/6+w/50, y+2*offset_y, w/6-2*w/50, y, options=["Save Frames Seperately", "Save Video and CSV", "No Save"], default=2, dir="V")
        self.save_option_lbl = Label(4*w/6+w/50, y+offset_y, w=w/6-2*w/50, text="Save Options:")
        self.save_param_project_name_lbl = Label(4*w/6+w/50, 2*h/10+y+offset_y, w = w/6-2*w/50, text="Project Name:")
        self.save_param_project_name_inp = InputBox(4*w/6+w/50, 2*h/10+y+2*offset_y, w = w/6-2*w/50, h=y, enable=False)
        self.save_param_session_lbl = Label(4*w/6+w/50, 3*h/10+y+offset_y, w = w/6-2*w/50, text="Session:")
        self.save_param_session_inp = InputBox(4*w/6+w/50, 3*h/10+y+2*offset_y, w = w/6-2*w/50, h=y, enable=False)
        self.save_output_btn = Button("Browse Save Path", w/6-2*w/50, y, (4*w/6+w/50, 4*h/10+y+2*offset_y), clickable=False, func=self.__save_path)
        self.save_output_lbl = Label(4*w/6+w/50, 4*h/10+2*y+2*offset_y, w=w/6-2*w/50)
        self.set_param_btn = Button("Set Parameters", w/6-2*w/50, y, (5*w/6+w/50, y+2*offset_y), func=self.__set_camera_parameter)
        self.get_param_btn = Button("Get Parameters", w/6-2*w/50, y, (5*w/6+w/50, h/10+y+2*offset_y), func=self.__get_camera_parameter)
        self.load_param_btn = Button("Load Parameters", w/6-2*w/50, y, (5*w/6+w/50, 2*h/10+y+2*offset_y), func=self.__load_camera_parameter)
        self.save_param_btn = Button("Save Parameters", w/6-2*w/50, y, (5*w/6+w/50, 3*h/10+y+2*offset_y), func=self.__save_camera_parameter)
        self.start_grabbing_btn = Button("Start Grabbing", w//10, y, (w-w//10-w//20, h-h//10-y-offset_y), clickable=True, func=self.__start_grabbing_pre)
        self.stop_grabbing_btn = Button("Stop Grabbing", w//10, y, (w-w//10-w//20, h-h//10), clickable=False, func=self.__stop_grabbing_pre)
        self.scheduled_ckb = Checkbox(screen, w-w//10-w//20, h-h//10-9*y-offset_y, caption="Scheduled Grabbing", func=self.__enable_scheduling)
        self.start_time_title_lbl = Label(w-w//10-w//20, h-h//10-8*y-offset_y, w//10, text="Start Time:")
        self.start_time_inp = InputBox(w-w//10-w//20, h-h//10-7*y-offset_y, w//10, y, text="HH:MM:SS", enable=False)
        self.start_time_lbl = Label(w-w//10-w//20, h-h//10-6*y-offset_y, w//10)
        self.end_time_title_lbl = Label(w-w//10-w//20, h-h//10-5*y-offset_y, w//10, text="Period Time:")
        self.end_time_inp = InputBox(w-w//10-w//20, h-h//10-4*y-offset_y, w//10, y, text="HH:MM:SS", enable=False)
        self.end_time_lbl = Label(w-w//10-w//20, h-h//10-3*y-offset_y, w//10)
        self.every_day_ckb = Checkbox(screen, w-w//10-w//20, h-h//10-2*y-offset_y, caption="Evryday", enable=False)
        self.back_to_params_btn = Button("Back", w//10, y, (w//20, h-h//10-y-offset_y), func=self.__go_back)
        self.initialize_save_btn = Button("Initialize", x, y, (w-x-w//20, h-h//10), clickable=False, func=self.__initiate_camera)

        # offline page
        self.browse_video_btn = Button("Select Input Video", w/5-2*w/50, y, (w/50, h/10+offset_y+offset_y), func=self.__select_video)
        self.browse_video_lbl = Label(w/50+w/5-2*w/50, h/10+offset_y+offset_y, w=3*w/10-w/50)
        self.start_sec_inp = InputBox(w/50, 5*h/20+offset_y, w/5-2*w/50, y, text='-1', func=self.__get_input, input="start_sec")
        self.start_sec_lbl = Label(w/50, 5*h/20,text="Start Second:")
        self.start_sec_hint_lbl = Label(w/50, 5*h/20+offset_y+y, text="-1 indicates from beginning")
        self.end_sec_inp = InputBox(w/5+w/50, 5*h/20+offset_y, w/5-2*w/50, y, text='-1', func=self.__get_input, input="end_sec")
        self.end_sec_lbl = Label(w/5+w/50, 5*h/20, text="End Second:")
        self.end_sec_hint_lbl = Label(w/5+w/50, 5*h/20+offset_y+y, text="-1 indicates to the end")
        self.browse_batch_btn = Button("Select Batch Folder", w/5-2*w/50, y, (w/2, h/10+offset_y+offset_y), func=self.__select_batch)
        self.browse_batch_lbl = Label(w/2, h/10+offset_y+offset_y+y, w=w/2-w/50)
        self.browse_batch_hint_lbl = Label(w/2, h/10+offset_y+offset_y+y+offset_y, w=w/2-w/50)
        self.batch_processing_rb = RadioButton(screen, w/50, h/10+offset_y, w-2*w/50, y, ["Single Processing","Batch Processing"], default=0, func=self.__batch_processing)
        # common
        self.prediction_device_lbl = Label(w//20, offset_y, color=(20,50,200))
        self.browse_box_pose_weight_btn = Button("Load Box&Pose Weights", w/5-2*w/50, y, (w/50, 9*h/20+offset_y), func=self.__select_weight, type="box_pose")
        self.browse_box_pose_weight_lbl = Label(w/50, 9*h/20+y+offset_y, w=2*w/5-2*w/50)
        self.identification_ckb = Checkbox(screen, w/5+w/50+w/10, 9*h/20+offset_y, func=self.__enable_identification, caption="Identification")
        self.browse_identification_weight_btn = Button("Load Identification Weights", w/5-2*w/50, y, (2*w/5+w/50, 9*h/20+offset_y), clickable=False, func=self.__select_weight, type="identification")
        self.browse_identification_weight_lbl = Label(2*w/5+w/50, 9*h/20+y+offset_y, w=2*w/5-2*w/50)
        self.monkey_num_inp = InputBox(w/50, 6*h/10+offset_y, w/5-2*w/50, y, func=self.__get_input, input="monkey_num")
        self.monkey_num_lbl = Label(w/50, 6*h/10, text="Number of Monkeys:")
        self.box_conf_inp = InputBox(w/5+w/50, 6*h/10+offset_y, w/5-2*w/50, y, func=self.__get_input, input="box_conf")
        self.box_conf_lbl = Label(w/5+w/50, 6*h/10, text="Bounding Box Confidence:")
        self.box_conf_hint_tk = Toolkit(screen, self.box_conf_lbl.x+self.box_conf_lbl.get_width()+10, 6*h/10, text="Higher value, more boxes which falsely detect as monkeys; Lower value, miss more monkeys")
        self.iou_thresh_inp = InputBox(2*w/5+w/50, 6*h/10+offset_y, w/5-2*w/50, y, func=self.__get_input, input="iou_thresh")
        self.iou_thresh_lbl = Label(2*w/5+w/50, 6*h/10,text="IoU Threshold:")
        self.iou_thresh_hint_tk = Toolkit(screen, self.iou_thresh_lbl.x+self.iou_thresh_lbl.get_width()+10, 6*h/10, text="Higher value, tighter boxes but little monkeys will be missed; Lower value, looser boxes and not so accurate.")
        self.kpt_conf_inp = InputBox(3*w/5+w/50, 6*h/10+offset_y, w/5-2*w/50, y, func=self.__get_input, input="kpt_conf")
        self.kpt_conf_lbl = Label(3*w/5+w/50, 6*h/10, text="Keypoint Confidence:")
        self.kpt_conf_hint_tk = Toolkit(screen, self.kpt_conf_lbl.x+self.kpt_conf_lbl.get_width()+10, 6*h/10, text="Higher value, just more confident keypoints will be chosen; Lower value, more keypoints will be chosen but may be unaccurate.")
        self.show_video_ckb = Checkbox(screen, w/50, 15*h/20, func=self.__enable_option, caption="Show Video", option="show_video")
        self.save_video_ckb = Checkbox(screen, w/5+w/50, 15*h/20, func=self.__enable_option, caption="Save Video Output", option="save_video")
        self.save_txt_ckb = Checkbox(screen, 2*w/5+w/50, 15*h/20, func=self.__enable_option, caption="Save Text Results", option="save_text")
        self.output_path_btn = Button("Select Output Path", w/5-2*w/50, y, (3*w/5+w/50, 15*h/20), func=self.__select_output_path)
        self.output_path_lbl = Label(3*w/5+w/50, 15*h/20+y, w=2*w/5-2*w/50)
        self.processing_lbl = Label(0, 0, w, pos='center')
        self.predict_2d_btn = Button("Predict", x, y, (w-x-w//20, h-h//10), clickable=False, func=self.__wait_for_process)
 
        ## 3D transformation
        self.tetha_x = -20
        self.tetha_y = -30
        self.tetha_z = 0
        self.project_name_lbl = Label(offset_x, h//10-offset_y,text="Camera Pair Name:")
        self.project_name_inp = InputBox(offset_x, h//10, x, y, func=self.__set_project_name)
        self.cube_width = 2700
        self.cube_height = 1200
        self.cube_depth = 4500
        self.width_lbl = Label(offset_x, h//5, text="Room Width (mm):")
        self.width_inp = InputBox(offset_x, h//5+offset_y, x//3, y, text=str(self.cube_width) ,func=self.__change_width)
        self.height_lbl = Label(offset_x+11*w//100, h//5, text="Room Height (mm):")
        self.height_inp = InputBox(offset_x+11*w//100, h//5+offset_y, x//3, y, text=str(self.cube_height) ,func=self.__change_height)
        self.depth_lbl = Label(offset_x+2*11*w//100, h//5, text="Room Depth (mm):")
        self.depth_inp = InputBox(offset_x+2*11*w//100, h//5+offset_y, x//3, y, text=str(self.cube_depth) ,func=self.__change_depth)
        self.primary_lbl = Label(offset_x, 4*h//10, text="Primary Camera:")
        self.name_lbl_1 = Label(offset_x, 4*h//10+offset_y,text="Name:")
        self.name_inp_1 = InputBox(offset_x, 4*h//10+2*offset_y, x//3, y,func=self.__set_camera_name, cam=1)
        self.wall_lbl_1 = Label(offset_x+12*w//125, 4*h//10+offset_y, text="Select Wall:")
        wall_icon = np.full((30,30,4), (255,255,0,50), np.uint8)
        wall_selected_icon = np.full((30,30,4), (50,50,50,200), np.uint8)
        wall_selected_icon[5:26,5:26] = (255,255,0,200)
        self.wall1_btn_1 = ImageButton(wall_icon, wall_selected_icon, 30, 30, (offset_x+12*w//125, 4*h//10+2*offset_y), func=self.__select_wall1, cam=1)
        wall_icon = np.full((30,30,4), (0,255,0,50), np.uint8)
        wall_selected_icon[5:26,5:26] = (0,255,0,200)
        self.wall2_btn_1 = ImageButton(wall_icon, wall_selected_icon, 30,30,(offset_x+12*w//125+offset_y,4*h//10+2*offset_y), func=self.__select_wall2, cam=1)
        wall_icon = np.full((30,30,4), (0,0,255,50), np.uint8)
        wall_selected_icon[5:26,5:26] = (0,0,255,200)
        self.wall3_btn_1 = ImageButton(wall_icon, wall_selected_icon, 30,30,(offset_x+12*w//125,4*h//10+2*offset_y+offset_y), func=self.__select_wall3, cam=1)
        wall_icon = np.full((30,30,4), (255,0,0,50), np.uint8)
        wall_selected_icon[5:26,5:26] = (255,0,0,200)
        self.wall4_btn_1 = ImageButton(wall_icon, wall_selected_icon, 30,30,(offset_x+12*w//125+offset_y,4*h//10+2*offset_y+offset_y), func=self.__select_wall4, cam=1)
        self.camera_height_lbl_1 = Label(offset_x+3*12*w//125, 4*h//10+offset_y, text="Y (mm):")
        self.camera_height_inp_1 = InputBox(offset_x+3*12*w//125, 4*h//10+2*offset_y, x//3, y, func=self.__set_camera_height, enable=False, cam=1)
        self.camera_width_lbl_1 = Label(offset_x+2*12*w//125, 4*h//10+offset_y, text="X (mm):")
        self.camera_width_inp_1 = InputBox(offset_x+2*12*w//125, 4*h//10+2*offset_y, x//3, y, func=self.__set_camera_width, enable=False, cam=1)
        self.secondary_lbl = Label(offset_x, 6*h//10, text="Secondary Camera:")
        self.name_lbl_2 = Label(offset_x, 6*h//10+offset_y, text="Name:")
        self.name_inp_2 = InputBox(offset_x, 6*h//10+2*offset_y, x//3, y,func=self.__set_camera_name, cam=2)
        self.wall_lbl_2 = Label(offset_x+12*w//125, 6*h//10+offset_y,text="Select Wall:")
        wall_icon = np.full((30,30,4), (255,255,0,50), np.uint8)
        wall_selected_icon[5:26,5:26] = (255,255,0,200)
        self.wall1_btn_2 = ImageButton(wall_icon, wall_selected_icon, 30,30,(offset_x+12*w//125,6*h//10+2*offset_y),func=self.__select_wall1, cam=2)
        wall_icon = np.full((30,30,4), (0,255,0,50), np.uint8)
        wall_selected_icon[5:26,5:26] = (0,255,0,200)
        self.wall2_btn_2 = ImageButton(wall_icon, wall_selected_icon, 30,30,(offset_x+12*w//125+offset_y,6*h//10+2*offset_y),func=self.__select_wall2, cam=2)
        wall_icon = np.full((30,30,4), (0,0,255,50), np.uint8)
        wall_selected_icon[5:26,5:26] = (0,0,255,200)
        self.wall3_btn_2 = ImageButton(wall_icon, wall_selected_icon, 30,30,(offset_x+12*w//125,6*h//10+2*offset_y+offset_y),func=self.__select_wall3,  cam=2)
        wall_icon = np.full((30,30,4), (255,0,0,50), np.uint8)
        wall_selected_icon[5:26,5:26] = (255,0,0,200)
        self.wall4_btn_2 = ImageButton(wall_icon, wall_selected_icon, 30,30,(offset_x+12*w//125+offset_y,6*h//10+2*offset_y+offset_y),func=self.__select_wall4,  cam=2)
        self.camera_height_lbl_2 = Label(offset_x+3*12*w//125, 6*h//10+offset_y,  text="Y (mm):")
        self.camera_height_inp_2 = InputBox(offset_x+3*12*w//125, 6*h//10+2*offset_y, x//3, y, func=self.__set_camera_height, enable=False, cam=2)
        self.camera_width_lbl_2 = Label(offset_x+2*12*w//125, 6*h//10+offset_y, text="X (mm):")
        self.camera_width_inp_2 = InputBox(offset_x+2*12*w//125, 6*h//10+2*offset_y, x//3, y, func=self.__set_camera_width, enable=False, cam=2)
        self.primary_browse_lbl = Label(offset_x+4*12*w//125, 4*h//10+offset_y, text="Camera Parameters")
        self.primary_browse_btn = Button("Browse", x//3, y, (offset_x+4*12*w//125, 4*h//10+2*offset_y), func=self.__get_primary_projection_matrix)
        self.primary_yaml_lbl = Label(offset_x+4*12*w//125, 4*h//10+2*offset_y+y, text='')
        self.secondary_browse_lbl = Label(offset_x+4*12*w//125, 6*h//10+offset_y,text="Camera Parameters")
        self.secondary_browse_btn = Button("Browse", x//3, y, (offset_x+4*12*w//125, 6*h//10+2*offset_y), func=self.__get_secondary_projection_matrix)
        self.secondary_yaml_lbl = Label(offset_x+4*12*w//125, 6*h//10+2*offset_y+y,text='')
        self.next_step_btn = Button("Next Step", x, y, (w-x-w//20,h-h//10), func=self.__next_step)

        self.video1_btn = Button("Load Calibration Video 1", w/4, y, (w//8, y), func=self.__load_video1)
        self.video2_btn = Button("Load Calibration Video 2", w/4, y, (w-w//8-w/4, y), func=self.__load_video2)
        self.frame_num_lbl = Label(w-2*x-w//20-w//40, h-h//10-y, w = x, pos="center",text="Current Frame: Frame #1")
        self.next_frame_btn = Button("Next Frame", x, y, (w-2*x-w//20-w//40, h-h//10), func=self.__next_frame)
        self.select_btn = Button("Select Frame", x, y, (w-x-w//20, h-h//10), func=self.__next_step)
        
        self.img1_lbl = Label((w-400)//3, 0)
        self.img2_lbl = Label(w-(w-400)//3-200, 0)
        self.pos_w_lbl = Label(2*w//5,2*h//3+offset_y+y, text="X (mm)")
        self.pos_w_inp = InputBox(2*w//5,2*h//3+offset_y+offset_y+y, x//3, y, text="", func=self.__set_pos_w)
        self.pos_h_lbl = Label(2*w//5+x//3+w//40,2*h//3+offset_y+y, text="Y (mm)")
        self.pos_h_inp = InputBox(2*w//5+x//3+w//40,2*h//3+offset_y+offset_y+y, x//3, y, text="", func=self.__set_pos_h)
        self.pos_d_lbl = Label(2*w//5+2*x//3+2*w//40,2*h//3+offset_y+y,text="Z (mm)")
        self.pos_d_inp = InputBox(2*w//5+2*x//3+2*w//40,2*h//3+offset_y+offset_y+y,x//3, y, text="", func=self.__set_pos_d)
        self.get_point_lbl = Label(2*w//5, 2*h//3)
        self.next_btn = Button("NEXT", x//3,y, (7*w//10, 2*h//3+offset_y+offset_y+y), func=self.__next)
        self.reset_btn = Button("RESET", x//3,y, (7*w//10+x//3+w//40, 2*h//3+offset_y+offset_y+y), func=self.__reset_positions)
        self.done_btn = Button("DONE", x, y, (w-x-w//20,h-h//10),func=self.__stop_getting_point)
        self.done_lbl = Label(w//3, 7*h//10)
        self.save_trans_path_btn = Button("Select Output Path", x, y, (w-x-w//20,h-h//10),func=self.__done)

        ## 3D localization
        self.browse_2d_kp_btn1 = Button("Add 2D Keypoint File 1", (2*w//3-270)//3, y, (100, h//10), func=self.__add_kp_file)
        self.browse_2d_kp_btn2 = Button("Add 2D Keypoint File 2", (2*w//3-270)//3, y, (120+(2*w//3-270)//3, h//10), func=self.__add_kp_file)
        self.browse_video_btn1 = Button("Add Corresponding Video 1", (2*w//3-270)//3, y, (100, h//10), func=self.__add_debug_video)
        self.browse_vider_btn2 = Button("Add Corresponding Video 2", (2*w//3-270)//3, y, (120+(2*w//3-270)//3, h//10), func=self.__add_debug_video)
        self.kp_files_list_lbl = Label(100+50, h//10+y+2*offset_y)
        self.debug_video_list_lbl = Label(120+(2*w//3-270)//3+50, h//10+y+2*offset_y)
        self.add_calibration_btn = Button("Add Pickle File", (2*w//3-270)//3, y, (170+2*(2*w//3-270)//3, h//10), func=self.__add_calibration_file)
        self.calibraion_list_lbl = Label(170+2*(2*w//3-270)//3+50, h//10+y+2*offset_y)
        self.add_pair_btn = Button("Add Another Pair?", x, y, (2*w//3+100, h//10), func=self.__add_pair)
        self.browse_identifier_model_btn = Button("Select Identifier Model", x, y, (100, 3*h//5), func=self.__load_identifier_model)
        self.browse_identifier_model_hink_tk = Toolkit(screen, 100+x, 3*h//5, text="Select an identifier model to infer the list of monkey names. Then select each monkey you want to be processed.")
        self.save_path_btn = Button("Browse Directory to Save", x, y, (3*w/5, 3*h//5), func=self.__save_path)
        self.save_path_lbl = Label(3*w/5, 3*h//5+50, w=2*w/5-2*w//50)
        self.save_3D_video_ckb = Checkbox(screen, 170+2*(2*w//3-270)//3, 3*h//5, func=self.__save_video, caption="Show & Save Video")
        self.output_fps_inp = InputBox(170+2*(2*w//3-270)//3, 3*h//5+y+offset_y, x/3, y, enable=False, text='50')
        self.output_fps_lbl = Label(170+2*(2*w//3-270)//3, 3*h//5+y, x, text="FPS of output video:")
        self.save_seperately_ckb = Checkbox(screen, 170+2*(2*w//3-270)//3, 3*h//5+2*y+2*offset_y, caption="Save 3D Results per Each Pair")
        self.reconstruct_3d_btn = Button("Reconstruct 3D", x, y, (w-x-w//20,h-h//10), func=self.__wait_for_process)

        ## 3D visualization
        self.width_vis_lbl = Label(2*offset_x, y, text="Room Width (mm):")
        self.width_vis_inp = InputBox(2*offset_x, y+offset_y, x//3, y, text=str(self.cube_width) ,func=self.__change_width)
        self.height_vis_lbl = Label(2*offset_x+11*w//100, y, text="Room Height (mm):")
        self.height_vis_inp = InputBox(2*offset_x+11*w//100, y+offset_y, x//3, y, text=str(self.cube_height) ,func=self.__change_height)
        self.depth_vis_lbl = Label(2*offset_x+2*11*w//100, y, text="Room Depth (mm):")
        self.depth_vis_inp = InputBox(2*offset_x+2*11*w//100, y+offset_y, x//3, y, text=str(self.cube_depth) ,func=self.__change_depth)
        self.browse_identifier_model_vis_btn = Button("Select Identifier Model", x, y, (2*offset_x, 3*y+offset_y), func=self.__load_identifier_model)
        self.browse_identifier_model_vis_hink_tk = Toolkit(screen, 2*offset_x+x, 3*y+offset_y, text="Select an identifier model to infer the list of monkey names. Then select each monkey you want to be processed.")
        self.save_3d_vis_ckb = Checkbox(screen, 2*offset_x, 5*y+offset_y, caption="Save Output Video", func=self.__save_video)
        self.output_fps_vis_inp = InputBox(2*offset_x, 6*y+2*offset_y, x/3, y, enable=False, text='50')
        self.output_fps_vis_lbl = Label(2*offset_x, 6*y+offset_y, x, text="FPS of output video:")
        self.save_path_vis_btn = Button("Browse Directory to Save", x, y, (2*offset_x, 8*y+offset_y), clickable=False, func=self.__save_path)
        self.save_path_vis_lbl = Label(2*offset_x, 9*y+offset_y, x)
        self.add_guide_video_btn = Button("Add Guide Video", x, y, (2*offset_x, 10*y+3*offset_y), func=self.__add_debug_video)
        self.add_guide_video_lbl = Label(2*offset_x, 10*y+2*offset_y, w = x, text="Add at most 2 videos (optional)")
        self.debug_video_list_vis_lbl = Label(2*offset_x, 11*y+3*offset_y, w = x)
        self.vis_3d_btn = Button("Visualize 3D", x, y, (w-x-w//20,h-h//10), func=self.__vis_3d_kpt)


        ## interact with jetson
        self.show_recieved_frames_ckb = Checkbox(screen, w-w/3, y, caption="Show Recieved Frames")
        self.save_recieved_video_path_btn = Button("Save Path", x, y, (w-w/3, 2*y), func=self.__recieved_video_output)
        self.save_recieved_video_path_lbl = Label(w-w/3, 2*y+offset_y, w/3)
        self.server_ip_inp = InputBox(w/50, y, w/5, y, func=self.__check_for_valid_ip)
        self.server_ip_lbl = Label(w/50, y-offset_y, w/3-2*w/50, text="Server IP:")
        self.initiate_server_btn = Button("Start Listening...", x, y, (w//3, y), clickable=False, func=self.__wait_for_process)
        self.initiate_server_lbl = Label(w//3, 2*y, w= x, color=(200, 50, 50))

        self.get_client_addresses_btn = Button("Load Client's Info File", w/5, y, (w/50,4*y), clickable=False, func=self.__get_client_info)
        self.get_client_addresses_lbl = Label(w/50,5*y, w = w/5)
        self.recieve_from_jetsons_btn = Button("Next", x, y, (w-x-w//20, h-h//10), clickable=True, func=self.__get_selected_jetsons)
        self.save_server_config_btn = Button("Save Connection Config", x, y, (w-2*x-w//20, h-h//10), clickable=True, func=self.__save_server_config)
        self.load_server_config_btn = Button("Load Connection Config", x, y, (w-3*x-w//20, h-h//10), clickable=True, func=self.__load_server_config)

        self.select_video_csv_btn = Button("Choose CSV file", w/5, y, (w/50, y), func=self.__select_processing_videos)
        self.select_video_csv_lbl = Label(w/50, 2*y, w = w/5)
        self.select_video_csv_hint_lbl = Label(w/50, 2*y, w = w/5, color=(200,50,50))
        self.video_num_to_process_lbl = Label(w-x-w//20, h-h//10-offset_y, w = w/5, color=(50,50,200))
        self.send_to_jetson_btn = Button("Send Parameters", x, y, (w-x-w//20, h-h//10), func=self.__wait_for_process)
        self.save_param_config_btn = Button("Save Processing  Config", x, y, (w-2*x-w//20, h-h//10), func=self.__save_param_config)
        self.load_param_config_btn = Button("Load Processing Config", x, y, (w-3*x-w//20, h-h//10), func=self.__load_param_config)

        self.every_day_jetson_ckb = Checkbox(screen, w-w//10-w//20, h-h//10-5*y-offset_y, caption="Evryday", enable=False)
        self.send_frame_to_jetson_btn = Button("Send Frames", x, y, (w-x-w//20, h-h//10-y-offset_y), func=self.__handle_jetson_process_pre)
        self.stop_send_frame_to_jetson_btn = Button("Stop Sending Frames", x, y, (w-x-w//20, h-h//10), clickable=False, func=self.__wait_for_process)
        self.scheduled_jetson_module_ckb = Checkbox(screen, w-w//10-w//20, h-h//10-9*y-offset_y, caption="Scheduled Processing", func=self.__enable_scheduling)


        self.__init()

    def __init(self):
        gc.collect()
        torch.cuda.empty_cache()

        ## camera
        self.camera_stream = None
        self.obj_cam_operation = []
        self.camera_result_lst.select_all(False)
        self.camera_result_lst.options = []
        self.camera_result_lst.enable = False
        self.selected_list = []
        self.vertical_flip_ckb.enable = False
        self.horizental_flip_ckb.enable = False
        self.apply_flip_lst.options = ["None", "All"]
        self.apply_flip_lst.enable = False
        self.exposure_inp.text = ''
        self.exposure_inp.enable = False
        self.auto_exposure_ckb.enable = False
        self.gain_inp.text = ''
        self.gain_inp.enable = False
        self.auto_gain_ckb.enable = False
        self.frame_rate_inp.text = ''
        self.frame_rate_inp.enable = False
        self.binning_rb.set_enable(False)
        self.action_device_key_inp.text = ''
        self.action_device_key_inp.enable = False
        self.action_group_key_inp.text = ''
        self.action_group_key_inp.enable = False
        self.action_group_mask_inp.text = ''
        self.action_group_mask_inp.enable = False
        self.save_param_project_name_inp.text = ''
        self.save_param_session_inp.text = ''
        self.get_param_btn.clickable = False
        self.set_param_btn.clickable = False
        self.load_param_btn.clickable = False
        self.save_param_btn.clickable = False
        self.thread_ActionCommand = None
        self.thread_frame_update = None
        self.display_thread = None
        self.b_is_run = False
        self.grabbingRunning = False
        self.is_running_timing = False
        self.in_time_range = False
        self.scheduled_ckb.enable = True
        self.back_to_params_btn.clickable = True
        self.frame_interval = 1.0  # Initialize shared frame interval

        ## reset all checkboxes
        for var in list(filter(lambda x: x.endswith('_ckb'), vars(self).keys())):
            getattr(self, var).reset()

        ## reset all labels
        for var in list(filter(lambda x: x.endswith('_lbl'), vars(self).keys())):
            getattr(self, var).reset()         
        
        ## reset all radio buttons
        for var in list(filter(lambda x: x.endswith('_rb'), vars(self).keys())):
            getattr(self, var).reset()
        
        ## reset all dropdowns and lists
        for var in list(filter(lambda x: x.endswith('_lst'), vars(self).keys())):
            getattr(self, var).reset()

        if torch.cuda.is_available(): # gpu is available
            self.cuda = True
            self.cuda_status_lbl.text = "CUDA is detected! feel free to process."
            self.cuda_status_lbl.color = (50,200,10)
            self.prediction_device_lbl.text = "CUDA is available. Prediction will be done on GPU."
        else:
            self.cuda = False
            self.cuda_status_lbl.text = "CUDA isn't detected! process cannot be done."
            self.cuda_status_lbl.color = (200,10,50)
            self.prediction_device_lbl.text = "CUDA isn't available! Prediction will be done on CPU."

        self.task_lst.enable = self.cuda
        self.browse_pt_model_btn.clickable = self.cuda
        self.pt_model = None
        self.model_name_inp.enable = self.cuda
        self.browse_dataset_btn.clickable = self.cuda
        self.backbone_size_rb.set_enable(self.cuda)
        self.fine_tune_ckb.enable = self.cuda
        self.epoch_inp.enable = self.cuda
        self.batch_inp.enable = self.cuda
        self.worker_inp.enable = self.cuda
        
        self.model_path = ''
        self.dataset_path = ''
        self.fine_tune = False
        self.pretrained_weight = ''

        self.input_video = []
        self.box_weight = ''
        self.identification = False
        self.identification_weight = ''
        self.browse_identification_weight_btn.clickable = False
        self.output_path = ''
        self.input = {
            "monkey_num":1, "box_conf":0.2, "iou_thresh":0.75, "kpt_conf":0.4,
            "start_sec": -1, "end_sec":-1,
            "show_video": False, "save_video":False, "save_text": False}
        self.monkey_num_inp.text = self.input["monkey_num"]
        self.box_conf_inp.text = self.input["box_conf"]
        self.iou_thresh_inp.text = self.input["iou_thresh"]
        self.kpt_conf_inp.text = self.input["kpt_conf"]
        self.start_sec_inp.text = self.input["start_sec"]
        self.end_sec_inp.text = self.input["end_sec"]

        self.stream_process = None
        self.process = None
        self.database = None

        ## ----- 3D
        self.project_name = ''
        self.project_name_inp.text = self.project_name
        self.name_inp_1.text = ''
        self.camera_height_inp_1.text = ''
        self.camera_width_inp_1.text = ''
        self.name_inp_2.text = ''
        self.camera_height_inp_2.text = ''
        self.camera_width_inp_2.text = ''
        self.camera_config_list = {1:{}, 2:{}}
        self.camera_pos = dict()
        self.selected_wall = 0
        for i in range(1,5):
            getattr(self,f"wall{i}_btn_1").isClicked = False
            getattr(self,f"wall{i}_btn_2").isClicked = False
        self.primary_P_mat = None
        self.secondary_P_mat = None
        self.frame1 = None
        self.frame2 = None
        self.positions1 = []
        self.positions2 = []
        self.pos_w = None
        self.pos_h = None
        self.pos_d = None
        self.real_pos = []
        self.pos_i = 0
        self.no_further_point = False
        self.done = False

        self.frame_generator = None
        self.kp_files_list = []
        self.debug_video_list = []
        self.reconstruction_files = []
        self.monkey_names = []
        self.monkey_names_ckb_list = []
        self.monkey_names_ckb_to_select_list = []
        self.monkey_3d_kpt_files = {}
        self.save_path = ''
        self.save_3d_video = False
        try:
            self.video_writer.release()
        except:
            pass
        try:
            self.video_out.release()
        except:
            pass
        self.video_writer = None
        self.video_debug1 = None
        self.video_debug2 = None
        self.add_pair = True
        self.identifier_loading = False
        self.wait = False
        self.convert = False
        self.wait_to_back = False
        self.back_to_menu_btn.clickable = True

        ## ------ Jetson
        self.server_ip_inp.text = ""
        self.initiate_server_btn.clickable = False
        self.get_client_addresses_btn.clickable = False
        self.server_socket = None
        self.server_socket_backward = None
        self.credentials_dict = {'ip': [], 'user': [], 'password': [], 'status': []}
        self.connected_ips = set()
        self.clients = {}
        self.is_sending_frame_to_jetson = False
        self.received_frame_idx = {}
        self.current_processing_video = {}
        self.show_received_frames = {}
        self.ip_ckbs = []
        self.jetson_check_btn = []
        self.thread_run_time_check = None
        self.jetson_send_process = []
        self.processingRunning = False
        self.communication_thread =  None
        self.check_connection_process = None
        self.is_connection_established = False
        self.update_jetson_status_process = None


    def __quit(self):
        # cap.release()
        for ch in active_children():
            ch.kill()
        if self.b_is_run:
            self.__close_camera()
        
        self.is_connection_established = False
        if self.update_jetson_status_process is not None:
            self.update_jetson_status_process.join()
        for cl in self.clients.values():
            send_data(cl[0], 'close')
            cl[0].close()
            cl[1].close()
        if self.server_socket is not None:
            self.server_socket.close()
        if self.server_socket_backward is not None:
            self.server_socket_backward.close()
        if self.communication_thread is not None:
            self.communication_thread.join()
        print("quit")

    def __go_menu(self):
        self.back_to_menu_btn.clickable = False
        
        if self.camera_stream is not None:
            for stream in self.camera_stream:
                stream.frame_queue.put(None)
            if self.process is not None:
                self.process.join()
        elif (self.process is not None) and (not self.done):
            self.input_video.clear()
            self.is_process_running.value = False
            self.is_stream_running.value = False
            return
        if self.b_is_run:
            self.__close_camera()
        self.is_connection_established = False
        if self.update_jetson_status_process is not None:
            self.update_jetson_status_process.join()
        for cl in self.clients.values():
            send_data(cl[0], 'close')
            cl[0].close()
            cl[1].close()
        if self.server_socket is not None:
            self.server_socket.close()
        if self.server_socket_backward is not None:
            self.server_socket_backward.close()
        if self.communication_thread is not None:
            self.communication_thread.join()

        self.__init()
        self.step = 0

    def __go_page(self, step):
        self.step = step

    def __go_back(self):
        if self.process is not None:
            for stream in self.camera_stream:
                stream.frame_queue.put(None)
            self.process.join()
        self.step -= 1

    def __next_step(self):
        if self.step == 50:
            self.__add_camera()
        self.step += 1

    def __select_option(self, option):
        self.step = option

    def __prompt_file(self, mode="file", filetype=("all files", "*.*")):
        """Create a Tk file dialog and cleanup when finished"""
        top = tkinter.Tk()
        top.withdraw()  # hide window
        if mode=='file':
            file_name = tkinter.filedialog.askopenfilename(parent=top, filetypes = (filetype,))
        elif mode=='save':
            file_name = tkinter.filedialog.asksaveasfilename(parent=top, filetypes = (filetype,))
        else:
            file_name = tkinter.filedialog.askdirectory(parent=top)
        top.destroy()
        if isinstance(file_name, tuple):
            raise Exception("cancel selection")
        
        return file_name
    
    def __wait_for_back(self):
        self.back_to_menu_btn.clickable = False
        if not self.wait_to_back:
            self.wait_to_back = True
            return
        if not self.convert:
            self.convert = True
            return
        
        self.__go_menu()
        self.back_to_menu_btn.clickable = True
        self.wait_to_back = False
        self.convert = False


    def __wait_for_process(self, process=None):
        if not self.wait:
            self.wait = True
            return
        
        if not self.convert:
            self.convert = True
            return
        
        process()
        self.back_to_menu_btn.clickable = True
        self.wait = False
        self.convert = False
  
    ## --------------------------- 2D - quantization ----------------------------
    def __select_pt_model(self):
        try:
            file = self.__prompt_file(filetype=("pt models", "*.pt"))
            self.pt_model = file
            self.pt_model_lbl.text = self.pt_model
        except:
            self.pt_model = None
            self.pt_model_lbl.text = ''

    def __quantize(self):
        self.process = Process(target=qt_run, args=(self.pt_model, self.task_lst.get_active_option()))
        self.process.start()
        self.back_to_menu_btn.clickable = False
        self.process.join()
        self.done = True
        self.quantize_done_lbl.text = f"Congratulations!\nOptimized model {self.pt_model.split('.')[0]}.engine was saved."

    ## --------------------------- 2D - train ------------------------------
    def __model_path(self):
        try:
            folder = self.__prompt_file("folder")
            self.model_path = folder
        except:
            self.model_path = ''
        self.model_path_lbl.text = self.model_path
            
    def __browse_dataset(self):
        try:
            folder = self.__prompt_file(mode="directory")
            subdirs = [f.lower() for f in os.listdir(folder) if os.path.isdir(os.path.join(folder,f)) ]
            if ('train' in subdirs) and (('test' in subdirs) or ('val' in subdirs)):
                self.dataset_path = folder
                self.browse_dataset_lbl.text = self.dataset_path
                self.browse_dataset_lbl.color = (0,0,0)
            else:
                self.browse_dataset_lbl.text = "Dataset should contains 'train' and 'test' subfolders."
                self.browse_dataset_lbl.color = (50,10,100)
        except Exception as e:
            print(e)

    def __check_model_exist(self, selected='medium'):
        if selected=='medium':
            if os.path.exists(os.getcwd() + '/models/training_weights/medium.pt'):
                self.backbone_size_res_lbl.text = ''
                self.backbone_size_res_lbl.color = (0,0,0)
            else:
                self.backbone_size_res_lbl.text = "File './models/training_weights/medium.pt' does not exist."
                self.backbone_size_res_lbl.color = (100,20,50)
        elif selected=='small':
            if os.path.exists(os.getcwd() + '/models/training_weights/small.pt'):
                self.backbone_size_res_lbl.text = ''
                self.backbone_size_res_lbl.color = (0,0,0)
            else:
                self.backbone_size_res_lbl.text = "File './models/training_weights/small.pt' does not exist."
                self.backbone_size_res_lbl.color = (100,20,50)
        elif selected=='large':
            if os.path.exists(os.getcwd() + '/models/training_weights/large.pt'):
                self.backbone_size_res_lbl.text = ''
                self.backbone_size_res_lbl.color = (0,0,0)
            else:
                self.backbone_size_res_lbl.text = "File './models/training_weights/large.pt' does not exist."
                self.backbone_size_res_lbl.color = (100,20,50)

    def __enable_finetune(self, state=True):
        self.fine_tune = state
        self.browse_pretrained_weight_btn.clickable = state
        self.backbone_size_rb.set_enable((not state))

    def __load_weight(self):
        try:
            file = self.__prompt_file(filetype=("pt models", "*.pt"))
            self.pretrained_weight = file
            self.browse_pretrained_weight_lbl.text = file
        except:
            self.browse_pretrained_weight_lbl.text = ''

    def __train(self):        
        self.step+=1
        self.losses = {"train":[], "val":[]}
        self.accuracy = {"train":[], "val":[]}
        self.is_process_running = Value("b", True)  # "b" indicates a boolean, "i" int
        self.loss_data = Queue()
        self.process = Process(target=train_run, args=(self.model_path,self.model_name_inp.text, self.dataset_path, self.fine_tune, self.pretrained_weight, self.backbone_size_rb.selected_option, int(self.epoch_inp.text), int(self.batch_inp.text), int(self.worker_inp.text), self.loss_data, self.is_process_running))
        # process.daemon = True
        self.process.start()
        # train_run(self.model_name_inp.text, self.dataset_path, self.fine_tune, self.pretrained_weight, self.backbone_size_rb.selected_option, int(self.epoch_inp.text), int(self.batch_inp.text), None, None)
        self.back_to_menu_btn.clickable = False

    ## -------------------------- 2D - predict ----------------------------
    def __search_camera(self):
        deviceList_ = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE # | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList_)
        if ret != 0:
            print('show error','enum devices fail! ret = '+ str(ret))
            return

        devicesText = ''
        if deviceList_.nDeviceNum == 0:
            print('show info','Find no GIGE device!')
            self.no_camera_lbl.text = 'Find no GIGE device!'
            return
        else:
            self.no_camera_lbl.text = ''
            print("Find %d devices!" % deviceList_.nDeviceNum)
            # tkinter.messagebox.showinfo('show info', "Find %d GIGE devices!" % deviceList.nDeviceNum + devicesText)

        self.deviceList = []
        self.devList = []
        for i in range(0, deviceList_.nDeviceNum):
            self.deviceList.append(deviceList_.pDeviceInfo[i])

        for i in range(0, len(self.deviceList)):
            mvcc_dev_info = cast(self.deviceList[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print ("\ngige device: [%d]" % i)
                devicesText += "\ngige device: [%d]" % i
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print ("Model: %s" % strModeName)
                # devicesText += "\nModel: %s" % strModeName

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print ("ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                devicesText += "\nip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4)
                self.devList.append("Gige["+str(i)+"]:"+str(nip1)+"."+str(nip2)+"."+str(nip3)+"."+str(nip4))

        self.camera_result_lst.options = ["All"] + self.devList
        self.camera_result_lst.enable = True
        self.camera_result_lst.reset()
        self.selected_list = []
        self.trigger_option_lst.enable = True

    def __check_master_slave(self):
        MASTER_MODE = 5
        SLAVE_MODE = 8

        # a loop to continuously check the status of ptp CameraInfo
        while True:
            camera_info_list = []
            for i in range(len(self.obj_cam_operation)):
                # Retrieve IEEE1588 status
                stGevIEEE1588Status = MVCC_ENUMVALUE()
                memset(byref(stGevIEEE1588Status), 0, sizeof(MVCC_ENUMVALUE))
                ret = self.obj_cam_operation[i].obj_cam.MV_CC_GetEnumValue("GevIEEE1588Status", stGevIEEE1588Status)
                if ret != 0:
                    print("get GevIEEE1588Status fail! ret[0x%x]" % ret)
                    return False
                else:
                    print(f"Default GevIEEE1588Status: {stGevIEEE1588Status.nCurValue}")
                # Store camera information
                camera_info_list.append({'index': i, 'status_value': stGevIEEE1588Status.nCurValue})

            # Check for master and slave cameras
            masterCameraIndex = -1
            slaveCameraCount = 0
            for info in camera_info_list:
                if info['status_value'] == MASTER_MODE:
                    if masterCameraIndex == -1:
                        # camera in master mode
                        masterCameraIndex = info['index']
                    else:
                        print("Camera at index %d is in Master mode, We have two masters!" % info['index'])
                        return False
                elif info['status_value'] == SLAVE_MODE:
                    # Camera in slave mode
                    slaveCameraCount = slaveCameraCount + 1
                else:
                    print("Camera at index %d is neither master nor slave..." % info['index'])


            # If master is not found or all cameras are not in the correct state, sleep and try again
            if masterCameraIndex != -1 and slaveCameraCount == (len(self.obj_cam_operation) - 1):
                break # Exit the loop as the desired configuration is found
            else:
                print("Sleeping for 2 seconds before retrying...")
                time.sleep(2)
                continue # Retry the loop
        return True

    # Thread to update frame_interval
    def update_frame_interval(self):
        while not self.stop_trigger:  # Check if the trigger process is running
            fps = float('inf')
            for iopen in range(0, len(self.obj_cam_operation)):
                ret = self.obj_cam_operation[iopen].Get_parameter(
                    self.trigger_option_lst.selected_option,
                    self.auto_exposure_ckb.checked,
                    self.auto_gain_ckb.checked
                )
                if ret == 0 and self.obj_cam_operation[iopen].frame_rate > 0:
                    fps = min(fps, float(self.obj_cam_operation[iopen].frame_rate))

            if fps == float('inf'):
                print("Error: No valid camera frame rates found!")
                fps = 1  # Fallback to default value

            self.frame_interval = 1.0 / fps
            print(f"**** Updated frame_interval: {self.frame_interval}")
            time.sleep(1)  # Update every second

    def ActionCommandWorkThread(self, g_DeviceKey, g_GroupKey, g_GroupMask):     # Define a thread for sending action commands
        stActionCmdInfo = MV_ACTION_CMD_INFO()
        stActionCmdResults = MV_ACTION_CMD_RESULT_LIST()
        stActionCmdInfo.nDeviceKey = g_DeviceKey
        stActionCmdInfo.nGroupKey = g_GroupKey
        stActionCmdInfo.nGroupMask = g_GroupMask
        stActionCmdInfo.pBroadcastAddress = b"255.255.255.255"
        stActionCmdInfo.nTimeOut = 0  # nTimeOutMilliseconds, ms ACK timeout, 0 indicates no need for acknowledgment
        stActionCmdInfo.bActionTimeEnable = 0  # Enable scheduled time or not: 1-enable
        # stActionCmdInfo.nActionTime = nTimeOutMilliseconds * 1000000  # ns Scheduled time, it is valid only when bActionTimeEnable values "1", it is related to the clock rate.

        while not self.stop_trigger:
            frame_interval = self.frame_interval  # Fetch the updated frame interval
            nTimeOutMilliseconds = int(frame_interval * 1000)
            print(f"frame_interval: {frame_interval}, nTimeOutMilliseconds: {nTimeOutMilliseconds}")

            startTime = time.time()
            # Send the PTP clock photo command
            nRet = MvCamera.MV_GIGE_IssueActionCommand(stActionCmdInfo, stActionCmdResults)
            if nRet != 0:
                print(f"Issue Action Command fail! nRet [0x{nRet:x}]")
                continue
            elapsedTime = ((time.time() - startTime) * 1000)
            sleepTime = nTimeOutMilliseconds - elapsedTime
            if sleepTime > 0:
                time.sleep(sleepTime / 1000)

    def SoftwareTriggerWorkThread(self):     # Define a thread for sending software triggers
        while not self.stop_trigger:
            frame_interval = self.frame_interval  # Fetch the updated frame interval
            nTimeOutMilliseconds = int(frame_interval * 1000)
            print(f"frame_interval: {frame_interval}, nTimeOutMilliseconds: {nTimeOutMilliseconds}")

            startTime = time.time()
            # Send the software Trigger
            for i in range(0, len(self.obj_cam_operation)):
                ret = self.obj_cam_operation[i].Trigger_once()
                if 0 != ret:
                    print('show error', 'camera:' + str(i) + 'doing software triggering fail!')

            elapsedTime = ((time.time() - startTime) * 1000)
            sleepTime = nTimeOutMilliseconds - elapsedTime
            if sleepTime > 0:
                time.sleep(sleepTime / 1000)
    
    def display_frames_from_queues(self, frame_queues, output_queue, save_mode, path, ip, fps):
        
        writerThreadList = []
        writerQueueList = []
        # Get the current date and time
        now = datetime.now()
        exp_time = now.strftime("%Y%m%d%H%M%S")
        if save_mode == "Save Frames Seperately":  # save as seperate frames
            for i in range(0, len(frame_queues)):
                queue_frame = queue.Queue()
                ip_cam = ip[i].split(":")[1].strip()
                folder_name = str(self.save_param_project_name_inp.text+ '_' + self.save_param_session_inp.text + '_' + ip_cam.split('.')[-1] + '_' + exp_time)
                folder_path = f"{path}/{folder_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                writerQueueList.append(queue_frame)
                write_thread = threading.Thread(target=write_frames, args=(queue_frame, folder_path, self.saved_sync_frames, fps, self.done_save_sync))
                write_thread.start()
                writerThreadList.append(write_thread)
        elif save_mode == "Save Video and CSV":    # save as a video
            self.database = pd.DataFrame(columns=['video_name','video_path', 'should_process', 'record_date','record_start_time','record_end_time','record_num_frames','sync_method','binning','resolution','flip_vertically','flip_horizontally','exposure','gain','fps'])
            for i in range(0, len(frame_queues)):
                queue_frame = queue.Queue()
                writerQueueList.append(queue_frame)
                file_name = str(self.save_param_project_name_inp.text+ '_' + self.save_param_session_inp.text  + "_" + ip[i].split(":")[1].strip().split('.')[-1] + '_'+ exp_time+'.mp4')
                write_thread = threading.Thread(target=write_video2, args=(queue_frame, os.path.join(path, file_name), fps, self.saved_sync_frames, self.done_save_sync))
                write_thread.start()
                writerThreadList.append(write_thread)
                self.database = pd.concat([self.database, pd.DataFrame({
                    'video_name':[file_name],'video_path':[f"{path}/"],'record_date':[now.strftime("%d/%m/%Y")],'record_start_time':[now.strftime("%H:%M:%S")],
                    'flip_horizontally': [self.camera_specific_parameters[ip[i].split(":")[1].strip()]['flipH']], 'flip_vertically': [self.camera_specific_parameters[ip[i].split(":")[1].strip()]['flipV']],
                    'exposure': [self.camera_specific_parameters[ip[i].split(":")[1].strip()]['exposure']], 'gain':[self.camera_specific_parameters[ip[i].split(":")[1].strip()]['gain']],
                    'binning': [int(self.binning_rb.selected_option)], 'sync_method':["PTP" if self.trigger_option_lst.selected_option==1 else "Software Triggering"],
                    'fps': [fps], 'resolution':["2048x1500" if int(self.binning_rb.selected_option)==2 else ("4096x3000" if int(self.binning_rb.selected_option)==1 else "1024x750")],
                    'should_process': [1]})], ignore_index=True)
            print(self.database)
        
        binning_size = int(self.binning_rb.selected_option)
        threshold_drop_frames = int(200 * binning_size)
        print('threshold_drop_frames: ', threshold_drop_frames)

        n = np.ceil(np.sqrt(len(frame_queues)))
        h = .9*screen.get_height()/n - (n-1)*.01*screen.get_height()
        w = (4/3)*h
        x = (screen.get_width()-(n*w)-((n-1)*.01*screen.get_height()))/2                                
        self.multi_plane_screen_info = (n,x,w,h)

        flag = False
        while self.grabbingRunning:
            synchronized_frames = []
            total_images_frame_queues = sum(q.qsize() for q in frame_queues)
            # print('total_images in frame_queues: ', total_images_frame_queues)
            for i in range(len(frame_queues)):
                frame_queue = frame_queues[i]
                while True:
                    queue_s = frame_queue.qsize()
                    # print("Camera[{}]: display_frames_from_queues Queue size: {}".format(i, queue_s))
                    if queue_s > 0:
                        numArray = frame_queue.get()
                        if numArray is not None:
                            if numArray is not 'NO_IMAGE':
                                self.recieved_sync_frames.value+=1
                            if total_images_frame_queues < threshold_drop_frames:  # otherwise drop frame
                                synchronized_frames.append(numArray)
                            else:
                                self.drop_sync_frames.value+=1
                        else:
                            flag = True  # Set the flag to False to break out of both loops
                        break
                    else:
                        if self.obj_cam_operation[i].not_responding == True:
                            if total_images_frame_queues < threshold_drop_frames:  # otherwise drop frame
                                synchronized_frames.append('NO_IMAGE')
                            break
                        else:
                            time.sleep(0.01)
                if flag:
                    break
            if flag:
                break
            if len(synchronized_frames) > 0:
                if len(writerQueueList)>0:   # 1: images bmp --- 2: video file
                    total_images_writerQueueList = sum(q.qsize() for q in writerQueueList) # check for drop synchronized frames if greater than a threshold
                    print('total_images in writerQueueList: ', total_images_writerQueueList)
                for i in range(len(frame_queues)):
                    current_image = synchronized_frames[i]
                    if current_image is not 'NO_IMAGE':
                        if len(writerQueueList)>0: 
                            if total_images_writerQueueList < threshold_drop_frames: # otherwise drop frame
                                writerQueueList[i].put(current_image)
                            else:
                                self.drop_sync_frames.value += 1
                total_images_queue_display = total_length_of_lists_in_queue(self.sync_frames_queue)  # check for drop synchronized frames if greater than a threshold
                # print('total_images in queue_display: ', total_images_queue_display)
                if total_images_queue_display < int(threshold_drop_frames / 4):  # otherwise drop frame
                    self.sync_frames_queue.put(synchronized_frames)

        if len(writerQueueList)>0:
            for i in range(0, len(frame_queues)):
                writerQueueList[i].put(None)
                writerThreadList[i].join()
        self.sync_frames_queue.put(None)

    def __select_camera(self, selected, status):
        if selected=="All":
            self.camera_result_lst.select_all(status)
        else:
            if len(self.camera_result_lst.selected) == len(self.camera_result_lst.options)-1:
                if 0 in self.camera_result_lst.selected:
                    self.camera_result_lst.selected.remove(0)
                else:
                    self.camera_result_lst.selected.add(0)
        self.selected_list = list(self.camera_result_lst.selected)
        try:
            self.selected_list.remove(0)
        except:
            pass
        self.open_cameras_btn.clickable = len(self.selected_list) > 0

    def __open_camera(self):
        if self.b_is_run:
            self.__close_camera()

        ## no camera should be run
        self.obj_cam_operation = []
        nOpenDevSuccess = 0
        self.devList = [self.devList[i-1] for i in self.selected_list]
        self.deviceList = [self.deviceList[i-1] for i in self.selected_list]
        for i in range(0, len(self.deviceList)):
            camObj = MvCamera()
            self.obj_cam_operation.append(CameraOperation(camObj,self.deviceList[nOpenDevSuccess],nOpenDevSuccess))
            ret = self.obj_cam_operation[nOpenDevSuccess].Open_device(self.trigger_option_lst.selected_option)
            if  0!= ret:
                self.obj_cam_operation.pop()
                self.devList.pop(nOpenDevSuccess)
                self.deviceList.pop(nOpenDevSuccess)
                print("open cam %d fail ret[0x%x]" % (i, ret))
                continue
            else:
                nOpenDevSuccess = nOpenDevSuccess + 1
                print("nOpenDevSuccess = ", nOpenDevSuccess)
                self.b_is_run = True

        if self.b_is_run:
            if self.trigger_option_lst.selected_option == 1:
                print("-Initializing- It takes some time! Please wait...")
                time.sleep(35)
                ret = self.__check_master_slave()
                if ret == False:
                    print("Initializing PTP fail! checkMasterSlave failed...")
            self.camera_specific_parameters = {}
            for dev in self.devList:
                self.camera_specific_parameters.update({dev.split(':')[-1]: {"flipV": False, "flipH": False, "exposure": -1, "gain": -1}})
            self.__get_camera_parameter()
            self.open_cameras_btn.clickable = False
            self.camera_result_lst.enable = False
            self.trigger_option_lst.enable = False
            self.apply_flip_lst.options = ["None", "All"]+self.devList
            self.vertical_flip_ckb.enable = True
            self.horizental_flip_ckb.enable = True
            self.apply_flip_lst.enable = True
            self.auto_exposure_ckb.enable = True
            self.auto_gain_ckb.enable = True
            self.frame_rate_inp.enable = True
            self.binning_rb.set_enable(True)
            if self.trigger_option_lst.selected_option == 1:
                self.action_device_key_inp.enable = True
                self.action_group_key_inp.enable = True
                self.action_group_mask_inp.enable = True
            self.get_param_btn.clickable = True
            self.load_param_btn.clickable = True
            self.save_param_btn.clickable = True

    def __auto_exposure(self, state=False):
        self.exposure_inp.enable = not state
        if not state:
            for iopen in range(0, len(self.obj_cam_operation)):
                ret = self.obj_cam_operation[iopen].Get_parameter(self.trigger_option_lst.selected_option, False, self.camera_specific_parameters[self.devList[iopen].split(":")[-1]]['gain']==-1)
                if 0 != ret or (self.obj_cam_operation[iopen].frame_rate == 0 and self.obj_cam_operation[iopen].exposure_time == 0 and self.obj_cam_operation[iopen].gain == 0):
                    print('show error','camera'+ str(iopen) +' get parameter fail!')
                    continue

                self.exposure_inp.text = self.obj_cam_operation[iopen].exposure_time
                break

    def __auto_gain(self, state=False):
        self.gain_inp.enable = not state
        if not state:
            for iopen in range(0, len(self.obj_cam_operation)):
                ret = self.obj_cam_operation[iopen].Get_parameter(self.trigger_option_lst.selected_option,  self.camera_specific_parameters[self.devList[iopen].split(":")[-1]]['exposure']==-1, False)
                if 0 != ret or (self.obj_cam_operation[iopen].frame_rate == 0 and self.obj_cam_operation[iopen].exposure_time == 0 and self.obj_cam_operation[iopen].gain == 0):
                    print('show error','camera'+ str(iopen) +' get parameter fail!')
                    continue

                self.gain_inp.text = self.obj_cam_operation[iopen].gain
                break

    def __reset_list(self, state=None):
        self.apply_flip_lst.set_active_option(0)
        if ((not self.auto_exposure_ckb.checked) and self.exposure_inp.text=='') or ((not self.auto_gain_ckb.checked) and self.gain_inp.text==''):
            self.apply_flip_lst.enable = False
        else:
            self.apply_flip_lst.enable = True

    def __set_camera_specific_parameters(self, option):
        if option == 0: # None
            return
        try:
            exposure = -1 if self.auto_exposure_ckb.checked else float(self.exposure_inp.text)
            gain = -1 if self.auto_gain_ckb.checked else float(self.gain_inp.text)
            if option == 1: # All
                for dev in self.devList:
                    self.camera_specific_parameters[dev.split(":")[-1]]['exposure'] = exposure
                    self.camera_specific_parameters[dev.split(":")[-1]]['gain'] = gain
                    self.camera_specific_parameters[dev.split(":")[-1]]['flipH'] = self.horizental_flip_ckb.checked
                    self.camera_specific_parameters[dev.split(":")[-1]]['flipV'] = self.vertical_flip_ckb.checked
            else:
                value = self.apply_flip_lst.get_active_option()
                self.camera_specific_parameters[value.split(":")[-1]]['exposure'] = exposure
                self.camera_specific_parameters[value.split(":")[-1]]['gain'] = gain
                self.camera_specific_parameters[value.split(":")[-1]]['flipH'] = self.horizental_flip_ckb.checked
                self.camera_specific_parameters[value.split(":")[-1]]['flipV'] = self.vertical_flip_ckb.checked
        except:
            print('Exposure time, gain, and frame rate must be valid numeric values.')
            self.__reset_list()

    def __device_select_for_flip(self):
        # This function will be called when a value is selected from the dropdown
        for i in range(0, len(self.obj_cam_operation)):
            ret = self.obj_cam_operation[i].Set_ReverseX(self.camera_specific_parameters[self.devList[i].split(":")[-1]]['flipH'])
            if 0 != ret:
                print('show error ', 'camera ' + str(i) + ' Set_ReverseX fail!')
            ret = self.obj_cam_operation[i].Set_ReverseY(self.camera_specific_parameters[self.devList[i].split(":")[-1]]['flipV'])
            if 0 != ret:
                print('show error ', 'camera ' + str(i) + ' Set_ReverseY fail!')

    def __close_camera(self):
        if self.thread_ActionCommand is not None:
            self.thread_ActionCommand.join()
        if self.thread_frame_update is not None:
            self.thread_frame_update.join()
        for i in range(0,len(self.obj_cam_operation)):
            ret = self.obj_cam_operation[i].Close_device(self.trigger_option_lst.selected_option)
            if 0 != ret:
                print('show error','camera:'+ str(i) + 'close deivce fail!')
                return
        self.b_is_run = False
        print("cam close ok ")

    def __get_camera_parameter(self):
        if self.apply_flip_lst.selected_option > 1:
            iopen = self.apply_flip_lst.selected_option-2
            ret = self.obj_cam_operation[iopen].Get_parameter(self.trigger_option_lst.selected_option, self.camera_specific_parameters[self.devList[iopen].split(":")[-1]]['exposure']==-1, self.camera_specific_parameters[self.devList[iopen].split(":")[-1]]['gain']==-1)
            if 0 != ret or (self.obj_cam_operation[iopen].frame_rate == 0 and self.obj_cam_operation[iopen].exposure_time == 0 and self.obj_cam_operation[iopen].gain == 0):
                print('show error','camera'+ str(iopen) +' get parameter fail!')
            self.frame_rate_inp.text = self.obj_cam_operation[iopen].frame_rate

            if self.camera_specific_parameters[self.devList[iopen].split(":")[-1]]['exposure'] == -1:
                self.exposure_inp.text = ''
                self.auto_exposure_ckb.checked = True
                self.exposure_inp.enable = False
            else:
                self.exposure_inp.text = self.obj_cam_operation[iopen].exposure_time
                self.auto_exposure_ckb.checked = False
                self.exposure_inp.enable = True
            if self.camera_specific_parameters[self.devList[iopen].split(":")[-1]]['gain'] == -1:
                self.gain_inp.text = ''
                self.auto_gain_ckb.checked = True
                self.gain_inp.enable = False
            else:
                self.gain_inp.text = self.obj_cam_operation[iopen].gain
                self.auto_gain_ckb.checked = False
                self.gain_inp.enable = True
            self.binning_rb.select(self.binning_rb.options.index(str(self.obj_cam_operation[iopen].binning)))
            if self.trigger_option_lst.selected_option == 1:
                self.action_device_key_inp.text = self.obj_cam_operation[iopen].ActionDeviceKey
                self.action_group_key_inp.text = self.obj_cam_operation[iopen].ActionGroupKey
                self.action_group_mask_inp.text = self.obj_cam_operation[iopen].ActionGroupMask
        else:
            for iopen in range(0, len(self.obj_cam_operation)):
                ret = self.obj_cam_operation[iopen].Get_parameter(self.trigger_option_lst.selected_option, self.camera_specific_parameters[self.devList[iopen].split(":")[-1]]['exposure']==-1, self.camera_specific_parameters[self.devList[iopen].split(":")[-1]]['gain']==-1)
                if 0 != ret or (self.obj_cam_operation[iopen].frame_rate == 0 and self.obj_cam_operation[iopen].exposure_time == 0 and self.obj_cam_operation[iopen].gain == 0):
                    print('show error','camera'+ str(iopen) +' get parameter fail!')
                    continue
                self.frame_rate_inp.text = self.obj_cam_operation[iopen].frame_rate

                if self.camera_specific_parameters[self.devList[iopen].split(":")[-1]]['exposure'] == -1:
                    self.exposure_inp.text = ''
                    self.auto_exposure_ckb.checked = True
                    self.exposure_inp.enable = False
                else:
                    self.exposure_inp.text = self.obj_cam_operation[iopen].exposure_time
                    self.auto_exposure_ckb.checked = False
                    self.exposure_inp.enable = True
                if self.camera_specific_parameters[self.devList[iopen].split(":")[-1]]['gain'] == -1:
                    self.gain_inp.text = ''
                    self.auto_gain_ckb.checked = True
                    self.gain_inp.enable = False
                else:
                    self.gain_inp.text = self.obj_cam_operation[iopen].gain
                    self.auto_gain_ckb.checked = False
                    self.gain_inp.enable = True
                self.binning_rb.select(self.binning_rb.options.index(str(self.obj_cam_operation[iopen].binning)))
                if self.trigger_option_lst.selected_option == 1:
                    self.action_device_key_inp.text = self.obj_cam_operation[iopen].ActionDeviceKey
                    self.action_group_key_inp.text = self.obj_cam_operation[iopen].ActionGroupKey
                    self.action_group_mask_inp.text = self.obj_cam_operation[iopen].ActionGroupMask
                break

    def __set_camera_parameter(self):
        for i in range(0,len(self.obj_cam_operation)):
            auto_exposure = False
            auto_gain = False
            try:
                self.obj_cam_operation[i].exposure_time = self.camera_specific_parameters[self.devList[i].split(":")[-1]]['exposure']
                if self.camera_specific_parameters[self.devList[i].split(":")[-1]]['exposure'] == -1:
                    auto_exposure = True
                self.obj_cam_operation[i].gain = self.camera_specific_parameters[self.devList[i].split(":")[-1]]['gain']
                if self.camera_specific_parameters[self.devList[i].split(":")[-1]]['gain'] == -1:
                    auto_gain = True
            except Exception as e:
                print(e)
            try:
                self.obj_cam_operation[i].frame_rate = float(self.frame_rate_inp.text)
            except:
                print('Exposure time, gain, and frame rate must be valid numeric values.')
            self.obj_cam_operation[i].binning = int(self.binning_rb.selected_option)
            if self.trigger_option_lst.selected_option == 1:
                try:
                    self.obj_cam_operation[i].ActionDeviceKey = int(self.action_device_key_inp.text)
                    self.obj_cam_operation[i].ActionGroupKey = int(self.action_group_key_inp.text)
                    self.obj_cam_operation[i].ActionGroupMask = int(self.action_group_mask_inp.text)
                except:
                    print('ActionDeviceKey, ActionGroupKey, and ActionGroupMask must be valid integer values.')
                ret = self.obj_cam_operation[i].Set_parameter(self.obj_cam_operation[i].frame_rate,self.obj_cam_operation[i].exposure_time,self.obj_cam_operation[i].gain,self.obj_cam_operation[i].binning,
                                                            self.obj_cam_operation[i].ActionDeviceKey, self.obj_cam_operation[i].ActionGroupKey, self.obj_cam_operation[i].ActionGroupMask, self.trigger_option_lst.selected_option,
                                                            auto_exposure, auto_gain)
            else:
                ret = self.obj_cam_operation[i].Set_parameter(self.obj_cam_operation[i].frame_rate,self.obj_cam_operation[i].exposure_time,self.obj_cam_operation[i].gain,self.obj_cam_operation[i].binning,
                                                            0, 0, 0, self.trigger_option_lst.selected_option, auto_exposure, auto_gain)
            if 0 != ret:
                print('show error','camera'+ str(i) + ' set parameter fail!')
        self.__get_camera_parameter()

    def __load_camera_parameter(self):
        try:
            file_path = self.__prompt_file(filetype=("JSON files", "*.json"))
            with open(file_path, 'r') as file:
                parameters = json.load(file)
            print("Loaded parameters:", parameters)
            self.frame_rate_inp.text = parameters.get('frame_rate')
            self.binning_rb.select(self.binning_rb.options.index(str(parameters.get('binning'))))
            if self.trigger_option_lst.selected_option == 1:
                self.action_device_key_inp.text = parameters.get('ActionDeviceKey')
                self.action_group_key_inp.text = parameters.get('ActionGroupKey')
                self.action_group_mask_inp.text = parameters.get('ActionGroupMask')
            self.save_option_rb.select(parameters.get('save_option'))
            self.save_param_project_name_inp.text = parameters.get('project')
            self.save_param_session_inp.text = parameters.get('session')
            self.save_path = parameters.get('save_path')
            self.save_output_lbl.text = self.save_path
            self.camera_specific_parameters.update(parameters.get('ip_specific'))
            self.__set_camera_parameter()
        except Exception as e:
            print(f"Error loading parameters: {e}")

    def __save_camera_parameter(self):
        try:
            file_path = self.__prompt_file(filetype=("JSON files", "*.json"), mode='save')
            self.__get_camera_parameter()
            parameters = {}
            for iopen in range(0, len(self.obj_cam_operation)):
                ret = self.obj_cam_operation[iopen].Get_parameter(self.trigger_option_lst.selected_option, self.auto_exposure_ckb.checked, self.auto_gain_ckb.checked)
                if 0 != ret or (self.obj_cam_operation[iopen].frame_rate == 0 and self.obj_cam_operation[iopen].exposure_time == 0 and self.obj_cam_operation[iopen].gain == 0):
                    print('show error','camera'+ str(iopen) +' get parameter fail!')
                    continue
                parameters.update({'frame_rate': self.obj_cam_operation[iopen].frame_rate, 'binning': self.obj_cam_operation[iopen].binning})
                if self.trigger_option_lst.selected_option == 1:
                    parameters.update({'ActionDeviceKey': self.obj_cam_operation[iopen].ActionDeviceKey, 'ActionGroupKey': self.obj_cam_operation[iopen].ActionGroupKey,
                                'ActionGroupMask': self.obj_cam_operation[iopen].ActionGroupMask})
                else:
                    parameters.update({'ActionDeviceKey': 1, 'ActionGroupKey': 1, 'ActionGroupMask': 1})
                break
            parameters.update({'save_option': self.save_option_rb.selected_choice, 'save_path':self.save_path, 'project':self.save_param_project_name_inp.text, 'session':self.save_param_session_inp.text})
            parameters.update({'ip_specific': self.camera_specific_parameters})
            with open(file_path, 'w') as file:
                json.dump(parameters, file, indent=2)
            print("Parameters saved successfully!")
        except Exception as e:
            print(f"Error saving parameters: {e}")

    def __initiate_camera(self):
        print("param", self.camera_specific_parameters)
        self.__device_select_for_flip()
        self.camera_stream = []
        for i in range(len(self.obj_cam_operation)):
            stream = OnlineStream()
            stream.from_camera(self.obj_cam_operation[i])
            self.camera_stream.append(stream)
        # self.camera_stream = OnlineStream()
        # self.camera_stream.from_camera(self.obj_cam_operation[0])
        # (frame_width, frame_height) = self.camera_stream.get_frame_size()
        # fps_video = self.obj_cam_operation[0].frame_rate
        # (start_frame, end_frame) = self.camera_stream.get_length()
        
        # self.out_queue = Queue()
        # self.saved_frames = Value("i", 0)
        # self.save_video_process = Process(target=write_video, args=(self.out_queue, os.path.join(f"{self.output_path}",f"{self.devList[self.camera_result_lst.selected_option]}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.mp4"), fps_video, frame_width, frame_height, self.saved_frames))
        # self.save_video_process.start()
        self.done = False
        self.start_grabbing_btn.clickable = True
        self.step+=1
        
    def __initiate_2d_process(self):
        self.processed_frame_num = Value("i", 0)
        self.is_process_running = Value("b", True)  # "b" indicates a boolean, "i" int
        self.is_process_waiting = Value("b", True)
        is_ready = Value("b", False)
        self.output_show_queue = Queue()
        self.camera_stream = OnlineStream()
        self.camera_stream.from_camera(self.obj_cam_operation[0])

        (frame_width, frame_height) = self.camera_stream.get_frame_size()
        fps_video = self.obj_cam_operation[0].frame_rate
        frame_width = int(frame_width * int(self.binning_rb.selected_option)/4)
        frame_height = int(frame_height * int(self.binning_rb.selected_option)/4)
        if self.input["save_video"]:
            self.out_queue = Queue()
            self.saved_frames = Value("i", 0)
            self.save_video_process = Process(target=write_video, args=(self.out_queue, os.path.join(f"{self.output_path}",f"{self.devList[self.camera_result_lst.selected_option]}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_2D_result.mp4"), fps_video, frame_width, frame_height, self.saved_frames))
            self.save_video_process.start()
        self.process = Process(target=run_2d, args=(self.box_weight, self.identification, self.identification_weight, int(self.input["monkey_num"]), self.camera_stream.frame_queue, 
                                frame_width, frame_height,fps_video, os.path.join(f"{self.output_path}",f"{self.devList[self.camera_result_lst.selected_option]}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_2D_result"),
                                self.input["show_video"], self.input["save_video"], self.input["save_text"], float(self.input["box_conf"]), float(self.input["iou_thresh"]),
                                float(self.input["kpt_conf"]), self.processed_frame_num, self.output_show_queue, self.is_process_running, self.is_process_waiting, is_ready))
        self.process.start()
        self.done = False
        self.start_grabbing_btn.clickable = True
        while not is_ready.value:
            time.sleep(0.01)
        self.step+=1
    
    def __enable_scheduling(self, state):
        self.start_time_inp.enable = state
        self.end_time_inp.enable = state
        self.every_day_ckb.enable = state
        self.every_day_jetson_ckb.enable = state

    def __run_time_check(self, start_time, end_time):
        while self.is_running_timing:
            print('run_time_check')
            now = datetime.now().time()
            if start_time <= end_time:
                if (start_time <= now <= end_time) and not self.in_time_range:
                    self.in_time_range = True
                    self.__start_grabbing()
                    print('started at: ', datetime.now().strftime("%H:%M:%S"))
                elif not self.every_day_ckb.checked and now > end_time and self.in_time_range:
                    print('ended at:   ', datetime.now().strftime("%H:%M:%S"))
                    self.__stop_grabbing_pre()
                    break
                elif self.every_day_ckb.checked and now > end_time and self.in_time_range:
                    print('ended at (every_day on):   ', datetime.now().strftime("%H:%M:%S"))
                    self.in_time_range = False
                    self.__stop_grabbing()
            else:
                if (now >= start_time or now <= end_time) and not self.in_time_range:
                    self.in_time_range = True
                    self.__start_grabbing()
                    print('started at: ', datetime.now().strftime("%H:%M:%S"))
                elif not self.every_day_ckb.checked and (now > end_time and now < start_time) and self.in_time_range:
                    print('ended at:   ', datetime.now().strftime("%H:%M:%S"))
                    self.__stop_grabbing_pre()
                    break
                elif self.every_day_ckb.checked and (now > end_time and now < start_time) and self.in_time_range:
                    print('ended at (every_day on):   ', datetime.now().strftime("%H:%M:%S"))
                    self.in_time_range = False
                    self.__stop_grabbing()
            time.sleep(1)  # Wait for 1 second before checking again
        print('stop clicked...')

    def __start_grabbing_pre(self):
        if not self.scheduled_ckb.checked:
            self.__start_grabbing()
        else:
            if not validate_time(self.start_time_inp.text):
                print("Invalid start time format. Please enter the time in HH:MM:SS format.")
                self.start_time_lbl.text = 'Invalid format.'
                return
            if not validate_time(self.end_time_inp.text):
                print("Invalid period time format. Please enter the time in HH:MM:SS format.")
                self.end_time_lbl.text = 'Invalid format.'
                return
            try:
                start_time = datetime.strptime(self.start_time_inp.text, "%H:%M:%S").time()
                period_parts = list(map(int, self.end_time_inp.text.split(':')))
                period_time = timedelta(hours = period_parts[0], minutes = period_parts[1],seconds = period_parts[2])
                end_time = (datetime.combine(datetime.today(), start_time) + period_time).time()
                print('start_time: ', start_time)
                print('period_time: ', period_time)
                print('end_time: ', end_time)
                self.in_time_range = False
                self.is_running_timing = True
                self.done_save_sync = Value("b", True)
                
                thread_run_time_check = threading.Thread(target = self.__run_time_check, args=(start_time, end_time))
                thread_run_time_check.start()
            except ValueError:
                print("Invalid time format. Please enter the time in HH:MM:SS format.")
                return
        
        self.scheduled_ckb.enable = False
        self.start_time_inp.enable = False
        self.end_time_inp.enable = False
        self.every_day_ckb.enable = False
        self.back_to_menu_btn.clickable = False
        self.back_to_params_btn.clickable = False
        self.start_grabbing_btn.clickable = False
        self.stop_grabbing_btn.clickable = True
        self.done = False
            
    def __start_grabbing(self):
        self.start_time_for_database = datetime.now().strftime('%Y%m%d_%H%M%S')
        frame_queues = []
        for idx, stream in enumerate(self.camera_stream):
            try:
                frame_queues.append(stream.frame_queue)
                stream.start(idx)
            except Exception as e:
                print(e)

        video_fps = float('inf')  # Initialize video_fps with a very large value
        for iopen in range(0, len(self.obj_cam_operation)):
            ret = self.obj_cam_operation[iopen].Get_parameter(
                self.trigger_option_lst.selected_option,
                self.auto_exposure_ckb.checked,
                self.auto_gain_ckb.checked
            )
            if 0 != ret or (
                    self.obj_cam_operation[iopen].frame_rate == 0 and
                    self.obj_cam_operation[iopen].exposure_time == 0 and
                    self.obj_cam_operation[iopen].gain == 0
            ):
                print(f'show error: camera {iopen} get parameter fail!')
            elif self.obj_cam_operation[iopen].frame_rate > 0:
                video_fps = min(video_fps, float(self.obj_cam_operation[iopen].frame_rate)) # Update video_fps to the minimum value encountered

        if video_fps == float('inf'): # Ensure video_fps is not infinity in case no valid cameras were found
            print("Error: No valid camera frame rates found!")
            video_fps = 1  # Default to a safe fallback value

        self.frame_interval = 1.0 / video_fps

        if self.trigger_option_lst.selected_option == 1:
            # Action command PTP
            A_DeviceKey = 1
            A_GroupKey = 1
            A_GroupMask = 1
            for iopen in range(0, len(self.obj_cam_operation)):
                ret = self.obj_cam_operation[iopen].Get_parameter(self.trigger_option_lst.selected_option, self.auto_exposure_ckb.checked, self.auto_gain_ckb.checked)
                if 0 != ret:
                    print('show error ', ' camera' + str(0) + ' get parameter fail!')
                else:
                    # Action command PTP
                    A_DeviceKey = int(self.obj_cam_operation[iopen].ActionDeviceKey)
                    A_GroupKey = int(self.obj_cam_operation[iopen].ActionGroupKey)
                    A_GroupMask = int(self.obj_cam_operation[iopen].ActionGroupMask)
                    break
            self.stop_trigger = False
            self.thread_ActionCommand = threading.Thread(target=self.ActionCommandWorkThread, args=(A_DeviceKey,A_GroupKey,A_GroupMask))
            self.thread_ActionCommand.start()
        else:
            self.stop_trigger = False
            self.thread_ActionCommand = threading.Thread(target=self.SoftwareTriggerWorkThread)
            self.thread_ActionCommand.start()

        self.thread_frame_update = threading.Thread(target = self.update_frame_interval)
        self.thread_frame_update.start()

        self.grabbingRunning = True
        self.sync_frames_queue = queue.Queue()
        self.recieved_sync_frames = Value("i", 0)
        self.saved_sync_frames = Value("i", 0)
        self.drop_sync_frames = Value("i", 0)
        self.show_sync_frames = 0
        self.done_save_sync = Value("b", False)
        self.display_thread = threading.Thread(target=self.display_frames_from_queues, args=(frame_queues, self.sync_frames_queue, self.save_option_rb.selected_option, self.save_path, self.devList, video_fps))
        self.display_thread.start()

    def __stop_grabbing_pre(self):
        if not self.scheduled_ckb.checked:
            self.__stop_grabbing()
        else:
            if self.is_running_timing:
                if self.in_time_range:
                    self.__stop_grabbing()
                else:
                    self.done = True
            self.is_running_timing = False
            self.in_time_range = False
        self.stop_grabbing_btn.clickable = False
  
    def __stop_grabbing(self):
        self.stop_trigger = True
        if self.thread_ActionCommand is not None:
            self.thread_ActionCommand.join()
            print("thread action command ", self.thread_ActionCommand.is_alive())
        if self.thread_frame_update is not None:
            self.thread_frame_update.join()
            print("thread frame_update. ", self.thread_frame_update.is_alive())
        for stream in self.camera_stream:
            try:
                stream.release()
            except Exception as e:
                print(e)
                traceback.print_exc()
        self.stop_grabbing_btn.clickable = False
        

    def __batch_processing(self, selected="Single Processing"):
        if selected == "Single Processing":
            batch = False
        elif selected == "Batch Processing":
            batch = True
        self.batch_processing = batch
        self.browse_batch_btn.clickable = batch
        self.browse_video_btn.clickable = not batch
        self.start_sec_inp.enable = not batch
        self.end_sec_inp.enable = not batch

    def __select_video(self):
        try:
            file = self.__prompt_file(filetype=("videos", "*.mp4 *.avi *.mkv"))
            cap = cv2.VideoCapture(file)
            cap.read()
            cap.release()
            self.input_video = [file]
            self.browse_video_lbl.text = file
            self.total_batch_video_num = len(self.input_video)
        except:
            self.input_video = []

    def __select_batch(self):
        try:
            folder = self.__prompt_file(mode="folder")
            self.input_video = glob.glob(f'{folder}/*.mp4')+glob.glob(f'{folder}/*.avi')+glob.glob(f'{folder}/*.mkv')
            self.total_batch_video_num = len(self.input_video)
            self.browse_batch_lbl.text = folder
            if self.total_batch_video_num > 0:
                self.browse_batch_hint_lbl.text = f"{self.total_batch_video_num} video(s) found."
                self.browse_batch_hint_lbl.color = (50,20,150)
            else:
                self.browse_batch_hint_lbl.text = f"There is no video in the selected path!"
                self.browse_batch_hint_lbl.color = (150,20,50)
        except:
            self.input_video = []

    def __select_weight(self, type):
        try:
            file = self.__prompt_file(filetype=("model weights", "*.engine *.pt") if self.cuda else ("PyTorch models", "*.pt"))
            if type=='box_pose':
                self.box_weight = file
            elif type=='identification':
                self.identification_weight = file
            getattr(self,f"browse_{type}_weight_lbl").text = file
        except:
            pass

    def __get_input(self, input):
        self.input[input] = getattr(self,f"{input}_inp").text
    
    def __enable_identification(self, state=True):
        self.identification = state
        self.browse_identification_weight_btn.clickable = state

    def __enable_option(self, option, state=True):
        self.input[option] = state

    def __select_output_path(self):
        try:
            folder = self.__prompt_file(mode="directory")
            self.output_path = folder
            self.output_path_lbl.text = self.output_path
            self.save_path_lbl.text = self.output_path
        except:
            pass

    def __predict_batch(self):
        try:
            video = self.input_video.pop(0)
            self.processing_lbl.text = f"Processing {video} ... (video {self.total_batch_video_num - len(self.input_video)}/{self.total_batch_video_num})"
            self.processed_frame_num = Value("i", 0)
            self.is_process_running = Value("b", True)  # "b" indicates a boolean, "i" int
            self.is_process_waiting = Value("b", True)  # "b" indicates a boolean, "i" int
            is_ready = Value("b", False)
            self.output_show_queue = Queue()

            self.video_stream = OfflineStream()
            self.video_stream.from_video(video)

            (frame_width, frame_height) = self.video_stream.get_frame_size()
            fps_video = self.video_stream.get_fps()
            # (frame_width, frame_height) = (1920, 1280)
            # fps_video = 50
            # start_frame, end_frame = -1, -1
            if self.input["save_video"]:
                self.out_queue = Queue()
                self.saved_frames = Value("i",0)
                self.save_video_process = Process(target=write_video, args=(self.out_queue, os.path.join(f"{self.output_path}",f"{video.split('/')[-1].split('.')[-2]}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_2D_result.mp4"), fps_video, frame_width, frame_height, self.saved_frames))
                self.save_video_process.start()
            self.input_queue = Queue()
            self.is_stream_running = Value("b", True)
            self.stream_process = Process(target=streaming, args=(video, float(self.input["start_sec"]), float(self.input["end_sec"]), self.input_queue, self.is_stream_running))
            self.process = Process(target=run_2d, args=(self.box_weight, self.identification, self.identification_weight, int(self.input["monkey_num"]), self.input_queue, 
                                frame_width, frame_height,fps_video, os.path.join(f"{self.output_path}",f"{video.split('/')[-1].split('.')[-2]}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_2D_result"),
                                self.input["show_video"], self.input["save_video"], self.input["save_text"], float(self.input["box_conf"]), float(self.input["iou_thresh"]),
                                float(self.input["kpt_conf"]), self.processed_frame_num, self.output_show_queue, self.is_process_running, self.is_process_waiting, is_ready))
            self.process.start()
            while not is_ready.value:
                time.sleep(0.01)
            self.stream_process.start()

        except Exception as e:
            print(e)
            self.done = True

    def __predict_2d(self):
        self.step += 1
        self.__predict_batch()


    ## ----------------------------- 3D -----------------------------
    def __validation(self) -> bool:
        if self.step == 50:
            if self.project_name == '':
                return False
            # primary camera info
            if "name" not in self.camera_config_list[1]:
                return False
            if self.camera_width_inp_1.text == '' or self.camera_height_inp_1.text == '':
                return False
            # secondary camera info
            if "name" not in self.camera_config_list[2]:
                return False
            if self.camera_width_inp_2.text == '' or self.camera_height_inp_2.text == '':
                return False
            
        elif self.step ==51:
            if self.primary_P_mat is None:
                return False
            if self.secondary_P_mat is None:
                return False
            
        elif self.step == 52:
            if self.frame1 is None:
                return False
            if self.frame2 is None:
                return False
            
        elif self.step == 53:
            if self.pos_w_inp.text =='' or self.pos_h_inp.text == '' or self.pos_d_inp.text == '':
                return False
            
        return True

    def __render_cube(self):
        w = self.cube_width
        h = self.cube_height
        d = self.cube_depth
        # cube = [[0,-h,d],[w,-h,d],[w,0,d],[0,0,d],[0,-h,0],[w,-h,0],[w,0,0],[0,0,0]]
        # cube = [[0,0,0],(w,0,0),[w,h,0],[0,h,0],[0,0,d],[w,0,d],[w,h,d],[0,h,d]]
        cube = [[-w/2,-h/2,d/2],[w/2,-h/2,d/2],[w/2,h/2,d/2],[-w/2,h/2,d/2],[-w/2,-h/2,-d/2],[w/2,-h/2,-d/2],[w/2,h/2,-d/2],[-w/2,h/2,-d/2]]
        cube.extend([[cube[6][0]+1000, cube[6][1], cube[6][2]], [cube[6][0], cube[6][1]+1000, cube[6][2]], [cube[6][0], cube[6][1], cube[6][2]+1000]])
        projection_matrix = [[1,0],[0,1],[0,0]]

        rotation_matrix = rotation_x(self.tetha_x)
        rotation_matrix = np.dot(rotation_matrix, rotation_y(self.tetha_y))
        rotation_matrix = np.dot(rotation_matrix, rotation_z(self.tetha_z))
        rotated_cube = np.dot(cube, rotation_matrix)
        projected_2d = np.dot(rotated_cube, projection_matrix)

        return projected_2d, rotation_matrix

    ## ---------------------------- 3D - calibration -------------------------
    def __open_jarvis(self):
        res = os.system("AnnotationTool")
        if res != 0:
            self.menu_jarvis_res_lbl.text = "JARVIS is not installed!"
        else:
            self.menu_jarvis_res_lbl.text = ''

    ## ---------------------------- 3D - transformation -----------------------------
    def __set_project_name(self):
        self.project_name = self.project_name_inp.text

    def __change_width(self):
        self.cube_width = int(self.width_inp.text)

    def __change_height(self):
        self.cube_height = int(self.height_inp.text)

    def __change_depth(self):
        self.cube_depth = int(self.depth_inp.text)

    def __set_camera_name(self, cam):
        self.camera_config_list[cam]["name"] = getattr(self,f"name_inp_{cam}").text

    def __select_wall1(self, cam):
        for i in range(1,5):
            getattr(self,f"wall{i}_btn_{cam}").isClicked = False
        self.selected_wall=1
        self.tetha_y = -30
        self.tetha_x = -20
        self.camera_pos[cam] = (0, 0, -self.cube_depth//2)
        self.camera_config_list[cam]["wall"] = 1
        getattr(self, f"camera_width_inp_{cam}").enable = True
        getattr(self, f"camera_height_inp_{cam}").enable = True

    def __select_wall2(self, cam):
        for i in range(1,5):
            getattr(self,f"wall{i}_btn_{cam}").isClicked = False
        self.selected_wall=2
        self.tetha_y = - 60
        self.tetha_x = -20
        self.camera_pos[cam] = (self.cube_width//2, 0, 0)
        self.camera_config_list[cam]["wall"] = 2
        getattr(self, f"camera_width_inp_{cam}").enable = True
        getattr(self, f"camera_height_inp_{cam}").enable = True

    def __select_wall3(self, cam):
        for i in range(1,5):
            getattr(self,f"wall{i}_btn_{cam}").isClicked = False
        self.selected_wall=3
        self.tetha_y = 90+ 60
        self.tetha_x = 20
        self.camera_pos[cam] = (-self.cube_width//2, 0, 0)
        self.camera_config_list[cam]["wall"] = 3
        getattr(self, f"camera_width_inp_{cam}").enable = True
        getattr(self, f"camera_height_inp_{cam}").enable = True

    def __select_wall4(self, cam):
        for i in range(1,5):
            getattr(self,f"wall{i}_btn_{cam}").isClicked = False
        self.selected_wall=4
        self.tetha_y = 180 - 30
        self.tetha_x = 20
        self.camera_pos[cam] = (0, 0, self.cube_depth//2)
        self.camera_config_list[cam]["wall"] = 4
        getattr(self, f"camera_width_inp_{cam}").enable = True
        getattr(self, f"camera_height_inp_{cam}").enable = True

    def __set_camera_width(self, cam):
        try:
            camera_width = int( getattr(self,f"camera_width_inp_{cam}").text)
            if self.selected_wall ==1 :
                self.camera_pos[cam] = (camera_width-(self.cube_width//2), self.camera_pos[cam][1], self.camera_pos[cam][2])
            elif self.selected_wall==2:
                self.camera_pos[cam] = (self.camera_pos[cam][0], self.camera_pos[cam][1], camera_width-(self.cube_depth//2))
            elif self.selected_wall==3:
                self.camera_pos[cam] = (self.camera_pos[cam][0], self.camera_pos[cam][1], (self.cube_depth//2)-camera_width)
            elif self.selected_wall==4:
                self.camera_pos[cam] = ((self.cube_width//2)-camera_width, self.camera_pos[cam][1], self.camera_pos[cam][2])
        except:
            getattr(self,f"camera_width_inp_{cam}").text = getattr(self,f"camera_width_inp_{cam}").text[:-1]
        
    def __set_camera_height(self, cam):
        try:
            camera_height = int( getattr(self,f"camera_height_inp_{cam}").text)
            self.camera_pos[cam] = (self.camera_pos[cam][0], (self.cube_height//2) - camera_height, self.camera_pos[cam][2])
        except:
            getattr(self,f"camera_height_inp_{cam}").text = getattr(self,f"camera_height_inp_{cam}").text[:-1]

    def __add_camera(self):
        for c in [1, 2]:
            position = None
            if self.selected_wall==1:
                flip = (-1,-1,1)
                position = (self.camera_pos[c][0]-self.cube_width//2, self.camera_pos[c][1]-self.cube_height//2, 0)
            elif self.selected_wall==2:
                flip = (-1,-1,-1)
                position = (0, self.camera_pos[c][1]-self.cube_height//2, self.cube_depth//2+self.camera_pos[c][2])    
            elif self.selected_wall==3:
                flip = (1,-1,1)
                position = (-self.cube_width, self.camera_pos[c][1]-self.cube_height//2, self.cube_depth//2+self.camera_pos[c][0])
            
            elif self.selected_wall==4:
                # flip = (1, -1, -1)
                flip = (-1,-1,1)
                position = (self.camera_pos[c][0] - self.cube_width//2, self.camera_pos[c][1]-self.cube_height//2, self.cube_depth//2+self.camera_pos[c][2])
            self.camera_config_list[c]["pos"] = self.camera_pos[c]
            self.camera_config_list[c]["world_pos"]=position
            self.camera_config_list[c]['flip']= flip
        
    def __get_primary_projection_matrix(self):
        try:
            file = self.__prompt_file(filetype=('yaml files', '*.yaml'))
        except:
            return
        try:
            P = get_projection_matrix(file)
            self.primary_P_mat = P
            self.primary_yaml_lbl.color = (0,0,0)
            self.primary_yaml_lbl.text = file.split('/')[-1]
        except:
            self.primary_yaml_lbl.color = color=(200,50,100)
            self.primary_yaml_lbl.text = "bad file, cannot generate projection matrix"

    def __get_secondary_projection_matrix(self):
        try:
            file = self.__prompt_file(filetype=('yaml files', '*.yaml'))
        except:
            return
        try:
            P = get_projection_matrix(file)
            self.secondary_P_mat = P
            self.secondary_yaml_lbl.color = (0,0,0)
            self.secondary_yaml_lbl.text = file.split('/')[-1]
        except:
            self.secondary_yaml_lbl.color = color=(200,50,100)
            self.secondary_yaml_lbl.text = "bad file, cannot generate projection matrix"
    
    def __load_video1(self):
        try:
            file = self.__prompt_file(filetype=("videos", "*.mp4 *.avi *.mkv"))
            self.cap1 = cv2.VideoCapture(file)
            _, self.frame1 = self.cap1.read()
            self.frame1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
        except:
            pass

    def __load_video2(self):
        try:
            file = self.__prompt_file(filetype=("videos", "*.mp4 *.avi *.mkv"))
            self.cap2 = cv2.VideoCapture(file)
            _, self.frame2 = self.cap2.read()
            self.frame2 = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)
        except:
            pass

    def __next_frame(self):
        frame_num = int(self.cap1.get(cv2.CAP_PROP_POS_FRAMES)+1)
        self.frame_num_lbl.text = f"Current Frame: Frame #{frame_num}"
        _, self.frame1 = self.cap1.read()
        self.frame1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
        _, self.frame2 = self.cap2.read()
        self.frame2 = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)

    def __set_pos_w(self):
        self.pos_w = self.pos_w_inp.text
    
    def __set_pos_h(self):
        self.pos_h = self.pos_h_inp.text

    def __set_pos_d(self):
        self.pos_d = self.pos_d_inp.text

    def __next(self):
        self.pos_i += 1
        self.real_pos.append((int(self.pos_w)-self.camera_config_list[1]['world_pos'][0], int(self.pos_h)-self.camera_config_list[1]['world_pos'][1], int(self.pos_d)-self.camera_config_list[1]['world_pos'][2]))
        self.pos_w_inp.text = ''
        self.pos_h_inp.text = ''
        self.pos_d_inp.text = ''
    
    def __reset_positions(self):
        self.positions1 = []
        self.positions2 = []
        self.real_pos = []
        self.pos_i = 0

    def __stop_getting_point(self):
        self.no_further_point = True

    def __done(self):
        try:
            folder = self.__prompt_file("directory")
            self.save_pickle_path = folder
            self.__process()
            self.done_lbl.text= f"Congratulations! {self.project_name}.pickle was saved."
            self.done_lbl.color = (0,0,0)
            self.done = True
        except Exception as e:
            self.done_lbl.text= f"An error occured.\n{e}"
            self.done_lbl.color = (200,50,100)

    def __process(self):
        transformation_matrix = find_transformation_matrix(self.primary_P_mat, self.secondary_P_mat, self.positions1, self.positions2, self.real_pos)
                        
        data = {
            "primary_mat": self.primary_P_mat,
            "secondary_mat": self.secondary_P_mat,
            "transformation_mat": transformation_matrix,
            "primary_cam": self.camera_config_list[1],
            "room_dim": [self.cube_width, self.cube_height, self.cube_depth]
        }
        with open(f"{self.save_pickle_path}/{self.project_name}.pickle", 'wb') as f:
            pickle.dump(data, f)

    ## ------------------------- 3D - vaisualization -----------------------------
    def __add_kp_file(self):
        try:
            file = self.__prompt_file(filetype=("2D keypoints", "*.txt"))
            self.kp_files_list.append(file)
            self.kp_files_list_lbl.text = '\n'.join([f.split('/')[-1] for f in self.kp_files_list])
        except:
            pass
    
    def __add_debug_video(self):
        try:
            file = self.__prompt_file(filetype=("Video", "*.mp4 *.avi *.mkv"))
            self.debug_video_list.append(file)
            self.debug_video_list_lbl.text = '\n'.join([f.split('/')[-1] for f in self.debug_video_list])
            self.debug_video_list_vis_lbl.text = '\n'.join([f.split('/')[-1] for f in self.debug_video_list])
        except:
            pass

    def __add_calibration_file(self):
        try:
            file = self.__prompt_file(filetype=("transformation files", "*.pickle"))
            self.reconstruction_files.append((self.kp_files_list[-2:], file))
            self.calibraion_list_lbl.text = '\n'.join([a[1].split('/')[-1] for a in self.reconstruction_files])
            self.add_pair = False
        except:
            pass

    def __add_pair(self):
        self.add_pair = True

    def __add_monkey(self, name, state=True):
        if state:
            self.monkey_names.append(name)
        else:
            self.monkey_names.remove(name)
    
    def __add_monkey2(self, name, i, state):
        if state:
            self.monkey_names.append(name)
        else:
            self.monkey_names.remove(name)
        self.monkey_names_ckb_to_select_list[i][1].clickable = state

    def __load_identifier_model(self):
        try:
            self.identifier_model_file = self.__prompt_file(filetype=("pytorch model", "*.pt *.engine" if self.cuda else "*.pt"))
            self.identifier_loading=True
            self.__wait_for_process()
        except:
            pass

    def __get_classes_name(self):
        classes_name = Queue()
        process = Process(target=inference, args=(self.identifier_model_file, classes_name, 0 if self.cuda else 'cpu'))
        process.start()
        self.monkey_names = list()
        while True:
            c = classes_name.get()
            if c is None:
                break
            self.monkey_names.append(c)
        process.join()
        print('classifier class names: ', self.monkey_names)

    def __parse_identifier_model(self):
        try:
            self.__get_classes_name()
            for i, cls in enumerate(self.monkey_names):
                self.monkey_names_ckb_list.append(
                    Checkbox(screen, self.browse_identifier_model_btn.x, self.browse_identifier_model_btn.y+70+i*50, caption=cls, func=self.__add_monkey, name=cls)
                )
                self.monkey_names_ckb_list[i].checked = True
                
        except Exception as e:
            print("an error occured", e)
            traceback.print_exc()
        self.identifier_loading = False

    def __parse_identifier_model2(self):
        try:
            self.__get_classes_name()
            for i, cls in enumerate(self.monkey_names):
                self.monkey_names_ckb_to_select_list.append((
                    Checkbox(screen, 2*screen.get_width()/5, 70+i*70, caption=cls, func=self.__add_monkey2, name=cls, i=i),
                    Button("Browse 3D file", 200, 50, (2*screen.get_width()/5 + 150, 70+i*70), func=self.__browse_3d_kpt, cls=cls, i=i),
                    Label(2*screen.get_width()/5 + 150+200+10, 70+i*70, w = screen.get_width()-(2*screen.get_width()/5 + 150+200+10)-50)
                ))
                self.monkey_names_ckb_to_select_list[i][0].checked = True
                
        except Exception as e:
            print("an error occured", e)
            traceback.print_exc()
        self.identifier_loading = False

    def __save_path(self):
        try:
            file = self.__prompt_file(mode='directory')
            self.save_path = file
        except:
            self.save_path = ''
        self.save_path_lbl.text = self.save_path
        self.save_path_vis_lbl.text = self.save_path
        self.save_output_lbl.text = self.save_path

    def __save_video(self, state=True):
        self.save_3d_video = state
        self.output_fps_inp.enable = state
        self.output_fps_vis_inp.enable = state
        self.save_path_vis_btn.clickable = state

    def __convert_2d_3d(self):
        import time
        s = time.time()

        tag = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        os.makedirs(f'{self.save_path}/3d_keypoints_{tag}', exist_ok=True)

        key_points_3d = []
        for files in self.reconstruction_files:
            kpts_3d = convert_2d_3d(files[0][0], files[0][1], files[1], classes=self.monkey_names)
            key_points_3d.append(kpts_3d)
            if self.save_seperately_ckb.checked:
                os.makedirs(f'{self.save_path}/3d_keypoints_{tag}/pair_{os.path.basename(files[1])[:-7]}', exist_ok=True)
                for cls_i, cls in enumerate(self.monkey_names):
                    with open(f'{self.save_path}/3d_keypoints_{tag}/pair_{os.path.basename(files[1])[:-7]}/{cls}.txt', 'a') as f: # self.f[mnk] = open(f'3d_keypoints_{mnk}.txt', 'a')
                        f.write(('%s' + '\n') % ('frame_num, nose_x, nose_y, nose_z, left_eye_x, left_eye_y, left_eye_z, right_eye_x, right_eye_y, right_eye_z, left_ear_x, left_ear_y, left_ear_z, right_ear_x, right_ear_y, right_ear_z, left_shoulder_x, left_shoulder_y, left_shoulder_z, right_shoulder_x, right_shoulder_y, right_shoulder_z, left_elbow_x, left_elbow_y, left_elbow_z, right_elbow_x, right_elbow_y, right_elbow_z, left_wrist_x, left_wrist_y, left_wrist_z, right_wrist_x, right_wrist_y, right_wrist_z, left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z, left_knee_x, left_knee_y, left_knee_z, right_knee_x, right_knee_y, right_knee_z, left_ankle_x, left_ankle_y, left_ankle_z, right_ankle_x, right_ankle_y, right_ankle_z'))
                        f.writelines("\n".join(f"{f_i+1} "+" ".join(f"{i:.2f}" for i in x) for f_i, x in enumerate(np.array(kpts_3d[cls_i]).reshape(-1,17*3))))

        num_frames = 0
        for x in key_points_3d:
            num_frames = max(num_frames, len(x[0]))
        for i, x in enumerate(key_points_3d):
            for j in range(len(self.monkey_names)):
                key_points_3d[i][j].extend([np.array([[-1, -1, -1]]*17)]*(num_frames-len(x[j])))
        print('calculate 3d', time.time()-s)
        with open(files[1], 'rb') as f:
            calibration_data = pickle.load(f)
        self.cube_width, self.cube_height, self.cube_depth = calibration_data["room_dim"]

        # ## for processing stream
        # s = time.time()
        # reconst_3d = combine(key_points_3d, monkeys=self.monkey_names)
        # print('combine', time.time()-s)
        # self.colors = [tuple((int(np.random.randint(0,255)), int(np.random.randint(0,255)), int(np.random.randint(0,255)))) for _ in range(len(self.monkey_names))]
        # s = time.time()
        # self.frame_generator = run_3d_reconstruction_stream(reconst_3d, num_frames, classes=self.monkey_names, colors=self.colors)
        # print('reconstruct', time.time()-s)

        ## for processing file
        s = time.time()
        reconst_3d = combine_all(key_points_3d, monkeys=self.monkey_names)
        print("combine", time.time()-s, np.array(reconst_3d[self.monkey_names[0]]).shape)
        np.random.seed(123)
        self.colors = [tuple((int(np.random.randint(0,255)), int(np.random.randint(0,255)), int(np.random.randint(0,255)))) for _ in range(len(self.monkey_names))]
        s = time.time()
        self.frame_generator = run_3d_reconstruction(reconst_3d, num_frames, classes=self.monkey_names, colors=self.colors)
        print("reconstruct", time.time()-s)

        s = time.time()
        for mnk in self.monkey_names:
            with open(f'{self.save_path}/3d_keypoints_{tag}/3d_keypoints_{mnk}_{tag}.txt', 'a') as f: # self.f[mnk] = open(f'3d_keypoints_{mnk}.txt', 'a')
                f.write(('%s' + '\n') % ('frame_num, nose_x, nose_y, nose_z, left_eye_x, left_eye_y, left_eye_z, right_eye_x, right_eye_y, right_eye_z, left_ear_x, left_ear_y, left_ear_z, right_ear_x, right_ear_y, right_ear_z, left_shoulder_x, left_shoulder_y, left_shoulder_z, right_shoulder_x, right_shoulder_y, right_shoulder_z, left_elbow_x, left_elbow_y, left_elbow_z, right_elbow_x, right_elbow_y, right_elbow_z, left_wrist_x, left_wrist_y, left_wrist_z, right_wrist_x, right_wrist_y, right_wrist_z, left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z, left_knee_x, left_knee_y, left_knee_z, right_knee_x, right_knee_y, right_knee_z, left_ankle_x, left_ankle_y, left_ankle_z, right_ankle_x, right_ankle_y, right_ankle_z'))
                f.writelines("\n".join(f"{f_i+1} "+" ".join(f"{i:.2f}" for i in x) for f_i, x in enumerate(np.array(reconst_3d[mnk]).reshape(-1,17*3))))
        print("write", time.time()-s)
        if self.save_3d_video:
            self.video_writer = cv2.VideoWriter(os.path.join(self.save_path, f'3D_vis_output_{tag}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), int(self.output_fps_inp.text), (screen.get_width(), screen.get_height()))

        try:
            self.video_debug1 = Stream()
            self.video_debug1.from_video(self.debug_video_list[0])
            self.video_debug2 = Stream()
            self.video_debug2.from_video(self.debug_video_list[1])
        except:
            print("couldn't load debug videos")

        self.step+=1

    def __browse_3d_kpt(self, cls, i):
        try:
            file = self.__prompt_file(filetype=("text file", "*.txt"))
            self.monkey_3d_kpt_files[cls] = file
            self.monkey_names_ckb_to_select_list[i][2].text = file
        except Exception as e:
            print(e)

    def __vis_3d_kpt(self):
        import time
        tag = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        reconst_3d = {}
        num_frames = 0
        for mnk in self.monkey_names:
            with open(self.monkey_3d_kpt_files[mnk], 'r') as f:
                tmp = [list(map(float, row.split(' ')[1:52])) for row in f.readlines()[1:]]
                print(np.array(tmp).shape)
                reconst_3d[mnk] = np.expand_dims(tmp, 1)
                num_frames = max(len(reconst_3d[mnk]), num_frames)
        np.random.seed(123)
        self.colors = [tuple((int(np.random.randint(0,255)), int(np.random.randint(0,255)), int(np.random.randint(0,255)))) for _ in range(len(self.monkey_names))]
        self.frame_generator = run_3d_reconstruction(reconst_3d, num_frames, classes=self.monkey_names, colors=self.colors)

        if self.save_3d_video:
            self.video_writer = cv2.VideoWriter(os.path.join(self.save_path, f'3D_vis_output_{tag}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), int(self.output_fps_inp.text), (screen.get_width(), screen.get_height()))

        try:
            self.video_debug1 = Stream()
            self.video_debug1.from_video(self.debug_video_list[0])
        except:
            print("couldn't load debug videos")
            self.video_debug1 = None
        try:
            self.video_debug2 = Stream()
            self.video_debug2.from_video(self.debug_video_list[1])
        except:
            self.video_debug2 = None

        self.step +=1


    ######## ------------------- frame manegment --------------------
    def __recieved_video_output(self):
        try:
            folder = self.__prompt_file(mode="directory")
            self.recieved_video_output = folder
            self.save_recieved_video_path_lbl.text = folder
        except:
            pass

    def __check_for_valid_ip(self):
        items = self.server_ip_inp.text.split('.')
        if len(items)==4:
            try:
                int(items[0])
                int(items[1])
                int(items[2])
                int(items[3])
                self.initiate_server_btn.clickable = True
                return
            except:
                pass
        self.initiate_server_btn.clickable = False
            
    def __init_server(self):
        self.initiate_server_lbl.text = ""
        server_port = 65432
        buffer_size = 30 * 1024 * 1024 * 2 # 30 MB buffer size for each jetson
        self.server_socket = bind_server(self.server_ip_inp.text, server_port, buffer_size, 2)        
        self.server_socket_backward = bind_server(self.server_ip_inp.text, 65431, buffer_size, 2)
        if self.server_socket is None or self.server_socket_backward is None:
            print("close server socket 3")
            if self.server_socket is not None:
                self.server_socket.close()
            if self.server_socket_backward is not None:
                self.server_socket_backward.close()
            self.initiate_server_lbl.text ="Connection Failed."
            self.initiate_server_lbl.color = (200,50,50)
            return

        self.is_connection_established = True
        self.get_client_addresses_btn.clickable = True
        self.initiate_server_btn.clickable = False
        self.initiate_server_lbl.text ="Connected."
        self.initiate_server_lbl.color = (50,200,50)

    def __check_jetson_connection(self):

        self.connected_ips = set()
        for idx in list(self.clients.keys()):
            if self.clients[idx][0].fileno() != -1:
                self.clients[idx][0].close()
            if self.clients[idx][1].fileno() != -1:
                self.clients[idx][1].close()

        while self.is_connection_established:
            try:
                print("check connection", self.clients.keys())
                # Accept a connection
                if self.server_socket is not None and self.server_socket_backward is not None:
                    client_socket, client_address = self.server_socket.accept()
                    client_socket2, client_address2 = self.server_socket_backward.accept()
                    if str(client_address[0]) in self.credentials_dict['ip']:
                        idx = self.credentials_dict['ip'].index(str(client_address[0]))
                        self.connected_ips.add(str(client_address[0]))
                        self.clients[idx] = (client_socket,client_socket2)
                        if self.connected_ips == set(self.credentials_dict['ip']):  #set([self.credentials_dict['ip'][i] for i, ch in enumerate(self.ip_ckbs) if ch.checked]): #
                            print("+++ All Jetsons have connected. +++")
                            # break
                    else:
                        print(f"Unknown IP {client_address[0]} tried to connect.")
                        client_socket.close()
            except Exception as e:
                try:
                    print(e)
                    self.server_socket.close()
                    self.server_socket_backward.close()
                    client_socket.close()
                    client_socket2.close()
                except:
                    pass


    def __check_jetsons_status(self):
        while self.is_connection_established:
            for crend_id in range(len(self.credentials_dict['status'])):
                self.credentials_dict['status'][crend_id] = [is_jetson_on(self.credentials_dict['ip'][crend_id], self.credentials_dict['user'][crend_id], self.credentials_dict['password'][crend_id]), is_gui_running(self.credentials_dict['ip'][crend_id], self.credentials_dict['user'][crend_id], self.credentials_dict['password'][crend_id], "jetson_client.py"), self.credentials_dict['ip'][crend_id] in self.connected_ips]
                self.jetson_check_btn[crend_id][0].clickable=np.array(self.credentials_dict['status'][crend_id]).all()
            time.sleep(10)

    def __get_client_info(self):

        self.credentials_dict = {'ip': [], 'user': [], 'password': [], 'status': []}
        try:
            file = self.__prompt_file(filetype=("text files", ".txt"))
            self.get_client_addresses_lbl.text = file

            with open(file, 'r') as file:
                header = file.readline().strip()  # Read and discard the header
                lines = file.readlines()

            for i,line in enumerate(lines):
                line = line.strip()
                parts = line.split('/')
                if len(parts) == 3:
                    self.credentials_dict['ip'].append(parts[0])
                    self.credentials_dict['user'].append(parts[1])
                    self.credentials_dict['password'].append(parts[2])
                self.credentials_dict['status'].append([False, False, False])
        except:
            self.get_client_addresses_lbl.text = ""

        self.__get_jetson_ips()

    def __get_jetson_ips(self):
        self.ip_ckbs = []
        try:
            for i in range(len(self.credentials_dict['ip'])):
                self.ip_ckbs.append(Checkbox(screen, screen.get_width()/50, (6+i*2)*(screen.get_height()/20), enable=True, caption=self.credentials_dict['ip'][i], default=True, func=self.enable_jetson, jetson_id = i))
                self.jetson_check_btn.append((Button("Check", screen.get_width()/10, screen.get_height()/20, (4*screen.get_width()/5, (6+i*2)*(screen.get_height()/20)), clickable = False, func=self.check_jetson, jetson_id=i), Label(4*screen.get_width()/5, (7+i*2)*(screen.get_height()/20))))

            self.update_jetson_status_process = threading.Thread(target=self.__check_jetsons_status, args=())
            self.update_jetson_status_process.start()
            self.check_connection_process = threading.Thread(target=self.__check_jetson_connection, args=())
            self.check_connection_process.start()
        except Exception as e:
            pass
    
    def enable_jetson(self,state, jetson_id):
        self.ip_ckbs[jetson_id].checked = state
        self.jetson_check_btn[jetson_id][0].clickable = np.array(self.credentials_dict['status'][jetson_id]).all() and state

    def check_jetson(self, jetson_id):
        frame = cv2.imread('utils/test_frame.png')
        _, encoded_frame = cv2.imencode('.jpg', frame)
        frame_data = encoded_frame.tobytes()
        data={'phase':'test', 'content': frame_data}
        ack = send_data(self.clients[jetson_id][0], data)
        if ack == "successful":
            self.jetson_check_btn[jetson_id][1].text = "Check passed successfully."
        else:
            self.jetson_check_btn[jetson_id][1].text = "Check failed."

    def __select_processing_videos(self):
        try:
            video_list_file = self.__prompt_file(filetype=("CSV files", "*.csv"))
            df = pd.read_csv(video_list_file)
            if "video_name" in df.columns and "video_path" in df.columns and "should_process" in df.columns:
                self.select_video_csv_lbl.text = video_list_file
                self.select_video_csv_hint_lbl.text = ""
            else:
                self.select_video_csv_lbl.text = ""
                self.select_video_csv_hint_lbl.text = "Necessary columns are missing."
        except:
            self.select_video_csv_lbl.text = ""
            self.select_video_csv_hint_lbl.text = ""

    def __send_parameters_jetson(self):
        print("sending ...")
        classifier = None
        classifier_type = ""
        if self.identification_weight!='':
            if self.identification_weight.split('.')[-1] == "engine":
                with open(self.identification_weight, 'rb') as f:
                    classifier = f.read()
                classifier_type="engine"
            else:
                classifier = YOLO(self.identification_weight, task='classify')
                classifier = pickle.dumps(classifier)
                classifier_type = "pt"
        if self.box_weight.split('.')[-1] == "engine":
            with open(self.box_weight, 'rb') as f:
                detector = f.read()
            detector_type = "engine"
        else:
            detector = YOLO(self.box_weight, task='pose')
            detector = pickle.dumps(detector)
            detector_type = "pt"
        data = {"monkey_num": self.monkey_num_inp.text, "box_conf":self.box_conf_inp.text, "iou_thresh": self.iou_thresh_inp.text, "kpt_conf": self.kpt_conf_inp.text, "detector": detector, "detector_type":detector_type, "identification": self.identification,"classifier": classifier, "classifier_type":classifier_type}
        for _, cli in self.clients.items():
            ack = send_data(cli[0], {'phase':'parameter', "content":data})
            if ack != 'ack':
                print("one client is failed!")
                return
        print("all clients get parameters successfully.")
        self.step+=1

    def __send_frame_jetson(self, jetson_id):
        try:
            while self.is_sending_frame_to_jetson:
                if not self.jetson_ready[jetson_id]:
                    continue
                self.jetson_ready[jetson_id] = False
                with self.lock:
                    if not self.shared_video_list:
                        print("No more videos to process.")
                        self.is_sending_frame_to_jetson = False
                        # self.send_frame_to_jetson_btn.clickable = True
                        # self.stop_send_frame_to_jetson_btn.clickable = False
                        break
                    video_address = self.shared_video_list.pop(0)  # Fetch and remove the first video address
                

                ## save log in database
                # self.database["jetson_ip"] = self.database["jetson_ip"].astype(str)
                self.database.loc[(self.database['video_path']+self.database['video_name'])==video_address, 'jetson_ip'] = self.clients[jetson_id][0].getpeername()[0]
                
                recieve_process = threading.Thread(target=self.__receive_result_jetson, args=(jetson_id,video_address))
                recieve_process.daemon = True
                recieve_process.start()

                print(f"Process {jetson_id} processing video: {video_address}")
                self.current_processing_video[jetson_id] = video_address
                cap = cv2.VideoCapture(video_address)
                frame_number = 1
                new_frame = True

                while self.processingRunning:
                    if frame_number - 50 > self.received_frame_idx[jetson_id]:
                        time.sleep(0.01)
                        continue

                    if new_frame:
                        ret, frame = cap.read()
                        if not ret:
                            break

                    # Encode frame as JPEG image
                    _, encoded_frame = cv2.imencode('.jpg', frame)
                    frame_data = encoded_frame.tobytes()

                    ack = send_data(self.clients[jetson_id][0], (frame_data, frame_number))
                    
                    # Get acknowledgement
                    if ack == str(frame_number):
                        print(f"Frame {frame_number} sent successfully.")
                        frame_number += 1
                        new_frame = True
                    else:
                        new_frame = False
                        continue

                    time.sleep(.1)
                # Send end-of-video trigger
                send_data(self.clients[jetson_id][0], (None, 'END'))
                print("sending done")
                
                cap.release()

                recieve_process.join()
                print(f"Process {jetson_id}: Finished video {video_address}")
        finally:
            print(f"Process {jetson_id} exiting.")


    def __draw_box_and_kpt(self, im0, c1, c2, x2, classlabel, classifier_conf, box_conf, colorlabel):
        # if classifier_conf < 0.5:
        #     classifier_conf = 1 - classifier_conf
        # classifier_conf = str(round(classifier_conf, 2))
        box_conf_str = str(round(box_conf, 2))

        Scale = round(max(1, min(im0.shape[0], im0.shape[1]) / 500))
        fontScale = round(max(1, min(c2[0]-c1[0], c2[1]-c1[1]) / 100))

        cv2.rectangle(im0, c1, c2, color=colorlabel, thickness=Scale, lineType=cv2.LINE_AA)

        plot_skeleton_kpts(im0, x2, 3, [int(c1[0]), int(c1[1]), int(c2[0]), int(c2[1])], self.kpt_conf_inp.text, fontScale)
        cv2.putText(im0, classlabel + ' ' + box_conf_str, c1, 0, fontScale, color=colorlabel, thickness=Scale, lineType=cv2.LINE_AA) #'id:' + classifier_conf +  +' '+ str(round(box_conf, 2)) +' ' classlabel + ' ' + classifier_conf
        return im0

    def __receive_result_jetson(self, jetson_id, file):
        print("receiving ...", jetson_id)
        self.clients[jetson_id][1].setblocking(False)  # Set non-blocking mode
        try:
            while True:
                data = self.clients[jetson_id][1].recv(1024)
                if not data:
                    break
        except BlockingIOError:
            pass  # No more data to read
        self.clients[jetson_id][1].setblocking(True)  # Restore blocking mode if needed
        self.received_frame_idx[jetson_id] = 0
        cap = cv2.VideoCapture(file)
        now = datetime.now()
        file_name = f"result_2D_{os.path.basename(file).split('.')[0]}_{now.strftime('%Y-%m-%d_%H:%M:%S')}.txt"
        result_file_name = f"{self.recieved_video_output}/{file_name}"
        with open(result_file_name, 'w') as f:
            f.write(('%s' + '\n') % ('frame_number, monkey_ID, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_confidence, nose_x, nose_y, nose_confidence, left_eye_x, left_eye_y, left_eye_confidence, right_eye_x, right_eye_y, right_eye_confidence, left_ear_x, left_ear_y, left_ear_confidence, right_ear_x, right_ear_y, right_ear_confidence, left_shoulder_x, left_shoulder_y, left_shoulder_confidence, right_shoulder_x, right_shoulder_y, right_shoulder_confidence, left_elbow_x, left_elbow_y, left_elbow_confidence, right_elbow_x, right_elbow_y, right_elbow_confidence, left_wrist_x, left_wrist_y, left_wrist_confidence, right_wrist_x, right_wrist_y, right_wrist_confidence, left_hip_x, left_hip_y, left_hip_confidence, right_hip_x, right_hip_y, right_hip_confidence, left_knee_x, left_knee_y, left_knee_confidence, right_knee_x, right_knee_y, right_knee_confidence, left_ankle_x, left_ankle_y, left_ankle_confidence, right_ankle_x, right_ankle_y, right_ankle_confidence'))
        
        ## save log to database
        # self.database["process_date"] = self.database["process_date"].astype(str)
        # self.database["process_start_time"] = self.database["process_start_time"].astype(str)
        # self.database["2d_result_path"] = self.database["2d_result_path"].astype(str)
        # self.database["2d_result_file"] = self.database["2d_result_file"].astype(str)
        self.database.loc[(self.database['video_path']+self.database['video_name'])==file, 'process_date'] = now.strftime('%d/%m/%Y')
        self.database.loc[(self.database['video_path']+self.database['video_name'])==file, 'process_start_time'] = now.strftime('%H:%M:%S')
        self.database.loc[(self.database['video_path']+self.database['video_name'])==file, '2d_result_path'] = self.recieved_video_output
        self.database.loc[(self.database['video_path']+self.database['video_name'])==file, '2d_result_file'] = f"{file_name}"

        n = np.ceil(np.sqrt(len(self.clients.keys())))
        h = .8*screen.get_height()/n - (n-1)*.01*screen.get_height()
        w = (4/3)*h
        x = (screen.get_width()-(n*w)-((n-1)*.01*screen.get_height()))/2 
        while True:
            try:
                data = receive_data(self.clients[jetson_id][1])
                if data is None:
                    print("data is none")
                    break
                # print(data)
                data = pickle.loads(data)
                if data is None:
                    print("pickle is none")
                    self.show_received_frames[jetson_id]=None
                    break

                # print(data[-2], self.received_frame_idx[jetson_id], data)
                if data[-2] > self.received_frame_idx[jetson_id]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, data[-2]-1)
                    # print('recieved frame', data[-2])
                    res, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = self.show_received_frames[jetson_id]

                if not(data[0][0]==-1 and data[0][1]==-1 and data[1][0]==-1 and data[1][1] ==-1):
                    with open(result_file_name, 'a') as f:
                        f.write(('%s ' * (2) + '%.2f ' * (17 * 3 + 5) + '\n') % (data[7], data[3], data[0][0], data[0][1], data[1][0], data[1][1], data[4], data[2][0], data[2][1], data[2][2], data[2][3], data[2][4], data[2][5], data[2][6]
                                                    , data[2][7], data[2][8], data[2][9], data[2][10], data[2][11], data[2][12], data[2][13], data[2][14], data[2][15], data[2][16], data[2][17], data[2][18], data[2][19], data[2][20]
                                                    , data[2][21], data[2][22], data[2][23], data[2][24], data[2][25], data[2][26], data[2][27], data[2][28], data[2][29], data[2][30], data[2][31], data[2][32], data[2][33], data[2][34]
                                                    , data[2][35], data[2][36], data[2][37], data[2][38], data[2][39], data[2][40], data[2][41], data[2][42], data[2][43], data[2][44], data[2][45], data[2][46], data[2][47], data[2][48], data[2][49], data[2][50]))

                    frame = self.__draw_box_and_kpt(frame, data[0], data[1], data[2], data[3], data[4], data[5], data[6])

                # frame = cv2.resize(frame, (int(w), int(h)))
                self.show_received_frames[jetson_id]=frame
                self.received_frame_idx[jetson_id] = data[-2]
                self.clients[jetson_id][1].sendall(f"ack {self.received_frame_idx[jetson_id]}".encode())
            except Exception as e:
                print("error in receiving result", e)
                traceback.print_exc()
                self.show_received_frames[jetson_id]=None
                break
        # self.__handle_jetson_process()
        print("=============end recieving===========")
        self.jetson_ready[jetson_id] = True

        self.database.loc[(self.database['video_path']+self.database['video_name'])==file, 'process_end_time'] = datetime.now().strftime('%d/%m/%Y')
        self.database.loc[(self.database['video_path']+self.database['video_name'])==file, 'process_num_frames'] = self.received_frame_idx[jetson_id]
        self.database.loc[(self.database['video_path']+self.database['video_name'])==file, 'should_process'] = 0



    def __handle_jetson_communication(self):
        self.communication_thread = threading.Thread(target=self.__handle_jetson_process)
        self.communication_thread.start()

    def __handle_jetson_process(self):

        print(f"Number of open threads: {len(threading.enumerate())}")

        self.database = pd.read_csv(self.select_video_csv_lbl.text)

        self.database['valid'] = self.database.apply(lambda x: os.path.isfile(os.path.join(x['video_path'], x['video_name'])), axis=1)
        self.valid_indices_of_database = list(self.database[self.database['should_process']==1][self.database['valid']].index)
        print(self.database)
        self.input_video = [os.path.join(row['video_path'], row['video_name']) for _, row in self.database.iloc[self.valid_indices_of_database].iterrows()]
        self.video_num_to_process_lbl.text = f"{len(self.input_video)} video(s) found."

        ## save processing parameters on the csv file
        self.database.loc[self.valid_indices_of_database, 'box_pose_model'] = self.box_weight
        self.database.loc[self.valid_indices_of_database, 'identification_model'] = self.identification_weight
        self.database.loc[self.valid_indices_of_database, 'num_monkeys'] = self.monkey_num_inp.text
        self.database.loc[self.valid_indices_of_database, 'bbox_confidence'] = self.box_conf_inp.text
        self.database.loc[self.valid_indices_of_database, 'IoU_threshold'] = self.iou_thresh_inp.text
        self.database.loc[self.valid_indices_of_database, 'keypoint_confidence'] = self.kpt_conf_inp.text
        
        self.is_sending_frame_to_jetson = True
        self.jetson_send_process = []
        self.jetson_recieve_process = []
        self.show_received_frames = {}

        manager = Manager()
        self.shared_video_list = manager.list(self.input_video)  # Shared list of videos
        self.lock = Lock()
        self.jetson_ready = {k: True for k in self.clients.keys()}
        self.processingRunning = True
        for idx in self.clients.keys():
            print("start thread for jetson", idx)
            send_process = threading.Thread(target=self.__send_frame_jetson, args=(idx,))
            self.jetson_send_process.append(send_process)
            send_process.start()

        while self.is_sending_frame_to_jetson:
            time.sleep(1)

        for pr in self.jetson_send_process:
            pr.join()


        self.processingRunning = False
        if not self.every_day_jetson_ckb.checked:
            self.is_running_timing = False
            self.done = True
            self.send_frame_to_jetson_btn.clickable = True
            self.stop_send_frame_to_jetson_btn.clickable = False
            self.video_num_to_process_lbl.text = ""

        self.database.to_csv(self.select_video_csv_lbl.text, index=False)


    def __run_time_check2(self, start_time):
        while self.is_running_timing:
            print('run_time_check')
            now = datetime.now().time()
            print(now)
            if start_time <= now and not self.in_time_range:
                self.in_time_range = True
                self.__handle_jetson_communication()
                print('started at: ', datetime.now().strftime("%H:%M:%S"))
            elif start_time > now and not self.processingRunning:
                self.in_time_range = False
            time.sleep(1)  # Wait for 1 second before checking again
        print('stop clicked...')

    def __handle_jetson_process_pre(self):
        self.done = False
        self.send_frame_to_jetson_btn.clickable =False
        if not self.scheduled_jetson_module_ckb.checked:
            self.__handle_jetson_communication()
        else:
            if not validate_time(self.start_time_inp.text):
                print("Invalid start time format. Please enter the time in HH:MM:SS format.")
                self.start_time_lbl.text = 'Invalid format.'
                return
            try:
                start_time = datetime.strptime(self.start_time_inp.text, "%H:%M:%S").time()
                print('start_time: ', start_time)
                self.in_time_range = False
                self.is_running_timing = True
                self.done_save_sync = Value("b", True)
                
                self.thread_run_time_check = threading.Thread(target = self.__run_time_check2, args=(start_time,))
                self.thread_run_time_check.start()
            except ValueError:
                print("Invalid time format. Please enter the time in HH:MM:SS format.")
                return
        
        self.stop_send_frame_to_jetson_btn.clickable = True
    
    def __get_selected_jetsons(self):
        # print(self.clients)
        active_ips = [b.checked for b in self.ip_ckbs]
        for key, a in enumerate(active_ips):
            if not a and key in self.clients:
                del self.clients[key]
        # print(self.clients)
        self.credentials_dict['ip'] = list(np.array(self.credentials_dict['ip'])[active_ips])
        self.credentials_dict['user'] = list(np.array(self.credentials_dict['user'])[active_ips])
        self.credentials_dict['password'] = list(np.array(self.credentials_dict['password'])[active_ips])
        self.credentials_dict['status'] = list(np.array(self.credentials_dict['status'])[active_ips])
        self.__next_step()

    def __stop_sending_frame(self):
        self.done = True
        self.stop_send_frame_to_jetson_btn.clickable = False
        self.video_num_to_process_lbl.text = ""
        self.is_sending_frame_to_jetson = False
        self.processingRunning = False
        self.is_running_timing = False
        for idx in range(len(self.clients.keys())):
            self.jetson_send_process[idx].join()
            print(f"process {idx} joined.")
        # for process in self.jetson_send_process:
        #     process.join()
        # print("send processes finished")
        # for process in self.jetson_recieve_process:
        #     process.join()
        # print("recive processes finished")
        self.is_running_timing = False
        self.send_frame_to_jetson_btn.clickable = True

    def __save_server_config(self):
        try:
            file_path = self.__prompt_file(mode='save', filetype=("json file", "*.json"))
        except:
            return
        parameters = {}
        parameters['server_ip']= self.server_ip_inp.text
        parameters['clients'] = self.credentials_dict
        parameters['client_ckb']=[ch.checked for ch in self.ip_ckbs]
        parameters['show'] = self.show_recieved_frames_ckb.checked
        parameters['save_path'] = self.save_recieved_video_path_lbl.text
        with open(file_path, 'w') as file:
            json.dump(parameters, file, indent=2)

    def __load_server_config(self):
        try:
            file = self.__prompt_file(mode="file", filetype=("json files", "*.json"))
            with open(file, 'r') as f:
                parameters = json.load(f)
            print(parameters)
            try:
                if self.server_socket is not None:
                    self.server_socket.close()
                if self.server_socket_backward is not None:
                    self.server_socket_backward.close()
            except Exception as e:
                print(e)
            self.server_ip_inp.text = parameters['server_ip']
            self.__init_server()

            self.credentials_dict = parameters['clients']
            self.__get_jetson_ips()

            for i in range(len(self.credentials_dict['ip'])):
                self.credentials_dict['status'][i] = [False, False, False]
                self.ip_ckbs[i].click(parameters['client_ckb'][i])

            self.show_recieved_frames_ckb.click(parameters['show'])

            self.recieved_video_output = parameters['save_path']
            self.save_recieved_video_path_lbl.text = parameters['save_path']
        
        except:
            return

    def __save_param_config(self):
        try:
            file_path = self.__prompt_file(mode='save', filetype=("json file", "*.json"))
        except:
            return 
        parameters = {}
        parameters['csv_file'] = self.select_video_csv_lbl.text
        parameters['box_model'] = self.box_weight
        parameters['identification'] = self.identification
        parameters['identification_model'] = self.identification_weight
        parameters['num_monkey'] = self.monkey_num_inp.text
        parameters['box_confidence'] = self.box_conf_lbl.text
        parameters['iou_threshold'] = self.iou_thresh_inp.text
        parameters['keypoint_confidence'] = self.kpt_conf_inp.text
        with open(file_path, 'w') as file:
            json.dump(parameters, file, indent=2)

    def __load_param_config(self):
        try:
            file = self.__prompt_file(mode="file", filetype=("json files", "*.json"))
            with open(file, 'r') as f:
                parameters = json.load(f)
            print(parameters)
            self.select_video_csv_lbl.text = parameters['csv_file']
            self.box_weight = parameters['box_model']
            self.browse_box_pose_weight_lbl.text = self.box_weight
            self.identification_ckb.checked = parameters['identification']
            self.__enable_identification(parameters['identification'])
            if self.identification:
                self.identification_weight = parameters['identification_model']
                self.browse_identification_weight_lbl.text = self.identification_weight
            self.monkey_num_inp.text = parameters['num_monkey']
            self.box_conf_lbl.text = parameters['box_confidence']
            self.iou_thresh_inp.text = parameters['iou_threshold']
            self.kpt_conf_inp.text = parameters['keypoint_confidence']
        
        except:
            return

    def __recieve_data(self):
        pass

    def run(self):
        done = False

        while not done:
            try:
                ## get and handle events in this frame
                self.events = pygame.event.get()
                for event in self.events:
                    if event.type == QUIT:
                        self.__quit()
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            self.__quit()
                            pygame.quit()
                            sys.exit()
                    # if event.type == VIDEORESIZE:
                    #     screen = pygame.display.set_mode((b,a), RESIZABLE)

                screen.fill((255, 255, 255))

                if self.step == 0: ## menu
                    ### draw sepration boxes
                    pygame.draw.rect(screen, (0,0,0), self.rect_2d ,border_radius = 6, width=5)
                    pygame.draw.rect(screen, (0,0,0), self.rect_3d ,border_radius = 6, width=5)
                    pygame.draw.rect(screen, (0,0,0), self.rect_cam ,border_radius = 6, width=5)
                    
                    pygame.draw.rect(screen, (255,255,255), self.r_2d)
                    screen.blit(self.text_2d, (self.r_2d.x+10, self.r_2d.y))
                    
                    pygame.draw.rect(screen, (255,255,255), self.r_3d)
                    screen.blit(self.text_3d, (self.r_3d.x+10, self.r_3d.y))

                    pygame.draw.rect(screen, (255,255,255), self.r_cam)
                    screen.blit(self.text_cam, (self.r_cam.x+10, self.r_cam.y))
                    
                    self.menu_quantize_btn.draw(screen)
                    self.menu_train_btn.draw(screen)
                    self.menu_box_pose_id_btn.draw(screen)
                    self.menu_jarvis_btn.draw(screen)
                    self.menu_jarvis_res_lbl.draw(screen)
                    self.menu_calibration_btn.draw(screen)
                    self.menu_localization_btn.draw(screen)
                    self.menu_visualization_btn.draw(screen)
                    self.menu_camera_btn.draw(screen)
                    self.menu_frame_manager_btn.draw(screen)
                    self.menu_box_pos_id_hint_tk.draw()
                    self.menu_train_hint_tk.draw()
                    self.menu_quantize_hint_tk.draw()
                    self.menu_visualization_hint_tk.draw()
                    self.menu_localization_hint_tk.draw()                    
                    self.menu_calibration_hint_tk.draw()
                    self.menu_jarvis_hint_tk.draw()
                    self.menu_frame_manager_tk.draw()
                    self.menu_camera_hint_tk.draw()

                else:
                    self.back_to_menu_btn.draw(screen)

                    if self.step == 10: # optimization
                        if self.cuda and (self.pt_model is not None):
                            self.quantize_btn.clickable = True
                        else:
                            self.quantize_btn.clickable = False
                        self.cuda_status_lbl.draw(screen)
                        self.task_lbl.draw(screen)
                        self.task_lst.update(self.events)
                        self.task_lst.draw(screen)
                        self.browse_pt_model_btn.draw(screen)
                        self.pt_model_lbl.draw(screen)
                        self.quantize_btn.draw(screen)
                        self.quantize_done_lbl.draw(screen)

                        if self.wait:
                            s = pygame.Surface((screen.get_width(),screen.get_height()))  # the size of your rect
                            s.set_alpha(220)                # alpha level
                            s.fill((50,50,50))           # this fills the entire surface
                            screen.blit(s, (0,0)) 
                            font = pygame.font.Font(None, 92)
                            text = font.render("Please Wait ...", True, (255,255,255))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                            self.__wait_for_process(self.__quantize)

                    if self.step == 20: # train - getting parameters
                        if self.model_name_inp.text != '' and self.model_path!='' and self.dataset_path != '' and self.epoch_inp.text != '' and self.batch_inp.text != '' and self.worker_inp.text != '' and ((not self.fine_tune and self.backbone_size_res_lbl.text=='') or (self.fine_tune and self.pretrained_weight!='')):
                            self.train_btn.clickable = True
                        else:
                            self.train_btn.clickable = False
                        self.cuda_status_lbl.draw(screen)
                        self.model_name_inp.draw(screen, self.events)
                        self.model_name_lbl.draw(screen)
                        self.model_path_btn.draw(screen)
                        self.model_path_lbl.draw(screen)
                        self.browse_dataset_btn.draw(screen)
                        self.browse_dataset_lbl.draw(screen)
                        self.browse_dataset_hink_tk.draw()
                        self.backbone_size_rb.update()
                        self.backbone_size_rb.draw()
                        self.backbone_size_lbl.draw(screen)
                        self.backbone_size_hint_tk.draw()
                        self.backbone_size_res_lbl.draw(screen)
                        self.fine_tune_ckb.render_checkbox()
                        self.browse_pretrained_weight_btn.draw(screen)
                        self.browse_pretrained_weight_lbl.draw(screen)
                        self.epoch_inp.draw(screen, self.events)
                        self.epoch_lbl.draw(screen)
                        self.batch_inp.draw(screen, self.events)
                        self.batch_lbl.draw(screen)
                        self.worker_inp.draw(screen, self.events)
                        self.worker_lbl.draw(screen)
                        self.train_btn.draw(screen)
                    elif self.step == 21:   # train - show results
                        if not self.is_process_running.value:
                            self.process.join()
                            self.back_to_menu_btn.clickable = True
                            self.done = True
                        elif self.loss_data.qsize() > 0:
                            losses = self.loss_data.get()
                            self.losses["train"].append(losses[1])
                            self.losses["val"].append(losses[2])
                            self.accuracy["val"].append(losses[0])
                        fig, ax = plt.subplots()
                        ax.plot(self.losses["train"], color='blue', label="train")
                        ax.plot(self.losses["val"], color='orange', label="validation")
                        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
                        plt.title("LOSS")
                        fig.tight_layout()
                        fig.canvas.draw()
                        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        data = cv2.resize(data, (2*screen.get_width()//5, int((data.shape[0]/data.shape[1])*2*screen.get_width()//5)))
                        screen.blit(pygame.image.frombuffer(data.tobytes(), (data.shape[1],data.shape[0]), "RGB"), (screen.get_width()//20,100))

                        fig = plt.figure()
                        plt.plot(self.accuracy["val"])
                        plt.title('Top1 Validation ACCURACY')
                        fig.canvas.draw()
                        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        data = cv2.resize(data, (2*screen.get_width()//5, int((data.shape[0]/data.shape[1])*2*screen.get_width()//5)))
                        screen.blit(pygame.image.frombuffer(data.tobytes(), (data.shape[1],data.shape[0]), "RGB"), (11*screen.get_width()//20,100))
                        plt.close('all')
                        if self.done :
                            font = pygame.font.Font(None, 36)
                            text = font.render(f"Training is finished! Weights are saved in '{os.path.join(self.model_path,self.model_name_inp.text)}/weights/best.pt'", True, (50,50,100))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, data.shape[0]+100+150))
                        
                    if self.step == 30:     # prediction from video - getting parameters (including batch processing)
                        if len(self.input_video)>0 and self.box_weight!='' and ((not self.identification) or (self.identification and self.identification_weight!='')) and (self.input["show_video"] or self.input["save_video"] or self.input["save_text"]) and ((not self.input['save_video'] and not self.input['save_text']) or ((self.input["save_video"] or self.input["save_text"]) and self.output_path!='')):
                            self.predict_2d_btn.clickable = True
                        else:
                            self.predict_2d_btn.clickable = False
                        self.prediction_device_lbl.draw(screen)
                        self.batch_processing_rb.update()
                        self.batch_processing_rb.draw()
                        self.browse_video_btn.draw(screen)
                        self.browse_video_lbl.draw(screen)
                        self.start_sec_inp.draw(screen, self.events)
                        self.start_sec_lbl.draw(screen)
                        self.start_sec_hint_lbl.draw(screen)
                        self.end_sec_inp.draw(screen, self.events)
                        self.end_sec_lbl.draw(screen)
                        self.end_sec_hint_lbl.draw(screen)
                        self.browse_batch_btn.draw(screen)
                        self.browse_batch_lbl.draw(screen)
                        self.browse_batch_hint_lbl.draw(screen)
                        self.browse_box_pose_weight_btn.draw(screen)
                        self.browse_box_pose_weight_lbl.draw(screen)
                        self.identification_ckb.render_checkbox()
                        self.browse_identification_weight_btn.draw(screen)
                        self.browse_identification_weight_lbl.draw(screen)
                        self.monkey_num_inp.draw(screen, self.events)
                        self.monkey_num_lbl.draw(screen)
                        self.box_conf_inp.draw(screen, self.events)
                        self.box_conf_lbl.draw(screen)
                        self.iou_thresh_inp.draw(screen, self.events)
                        self.iou_thresh_lbl.draw(screen)
                        self.kpt_conf_inp.draw(screen, self.events)
                        self.kpt_conf_lbl.draw(screen)
                        self.show_video_ckb.render_checkbox()
                        self.save_video_ckb.render_checkbox()
                        self.save_txt_ckb.render_checkbox()
                        if self.input["save_text"] or self.input["save_video"]:
                            self.output_path_btn.draw(screen)
                            self.output_path_lbl.draw(screen)
                        self.predict_2d_btn.draw(screen)
                        self.kpt_conf_hint_tk.draw()
                        self.iou_thresh_hint_tk.draw()
                        self.box_conf_hint_tk.draw()

                        if self.wait:
                            s = pygame.Surface((screen.get_width(),screen.get_height()))  # the size of your rect
                            s.set_alpha(220)                # alpha level
                            s.fill((50,50,50))           # this fills the entire surface
                            screen.blit(s, (0,0)) 
                            font = pygame.font.Font(None, 92)
                            text = font.render("Please Wait ...", True, (255,255,255))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                            self.__wait_for_process(self.__predict_2d)
                    elif self.step == 31:   # prediction from video - showing video, and how many frames have been processed
                        if self.done:
                            font = pygame.font.Font(None, 108)
                            text = font.render("Process is done!", True, (50,50,100))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2-50))
                        else:
                            if self.input['show_video']:
                                img = self.output_show_queue.get()
                                if img is None:
                                    self.process.join()
                                    self.is_stream_running.value = False
                                    self.stream_process.join()
                                    if self.input['save_video']:
                                        self.out_queue.put(None)
                                        self.save_video_process.join()
                                    self.__predict_batch()
                                    continue
                                if isinstance(img, str) and img=='ERROR':
                                    self.process.join()
                                    self.is_stream_running.value = False
                                    self.stream_process.join()
                                    if self.input['save_video']:
                                        self.out_queue.put(None)
                                        self.save_video_process.join()
                                    self.done = True
                                    # self.__go_menu()
                                    continue
                                if self.input['save_video']:
                                    self.out_queue.put(img)
                                self.processed_frame_num.value+=1
                                self.processing_lbl.draw(screen)
                                r = img.shape[1]/img.shape[0]
                                h = .9*screen.get_height()
                                w = r*h
                                if w > 13*screen.get_width()/20:
                                    w = 13*screen.get_width()/20
                                    h = w/r
                                img = cv2.resize(img, (int(w), int(h)))
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                screen.blit(pygame.image.frombuffer(img.tobytes(), (img.shape[1],img.shape[0]), "RGB"), ((screen.get_width()-img.shape[1])//2,(screen.get_height()-img.shape[0])//2))

                            elif self.input['save_video']:
                                img = self.output_show_queue.get()
                                if img is None:
                                    self.process.join()
                                    self.is_stream_running.value = False
                                    self.stream_process.join()
                                    if self.input['save_video']:
                                        self.out_queue.put(None)
                                        self.save_video_process.join()
                                    self.__predict_batch()
                                    continue
                                if isinstance(img, str) and img=='ERROR':
                                    self.process.join()
                                    self.is_stream_running.value = False
                                    self.stream_process.join()
                                    if self.input['save_video']:
                                        self.out_queue.put(None)
                                        self.save_video_process.join()
                                    self.done = True
                                    # self.__go_menu()
                                    continue
                                self.out_queue.put(img)
                                self.processed_frame_num.value+=1
                                font = pygame.font.Font(None, 36)
                                text = font.render(self.processing_lbl.text, True, (50,50,50))
                                r = text.get_rect()
                                screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2-50))
                                font = pygame.font.Font(None, 92)
                                if self.is_process_running.value:
                                    text = font.render(f"{self.processed_frame_num.value} frames have been processed.", True, (50,50,50))
                                    r = text.get_rect()
                                    screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2+50))
                                else:
                                    text = font.render("Output video is saving ...", True, (50,50,50))
                                    r = text.get_rect()
                                    screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2+50))

                            
                            elif self.input['save_text']:
                                font = pygame.font.Font(None, 36)
                                text = font.render(self.processing_lbl.text, True, (50,50,50))
                                r = text.get_rect()
                                font = pygame.font.Font(None, 92)
                                screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2-100)) 
                                if self.is_process_running.value:
                                    text = font.render(f"{self.processed_frame_num.value} frames have been processed.", True, (50,50,50))
                                    r = text.get_rect()
                                    screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))   
                                else:
                                    self.process.join()
                                    self.is_stream_running.value = False
                                    self.stream_process.join()
                                    self.__predict_batch()
                                    continue
                            
                    if self.step == 40: # 3D calibration
                        pass

                    if self.step == 50 or self.step == 51:
                        scale = 0.1
                        position = (300,500)
                        canvas = np.ones((screen.get_height(),700,4), np.uint8)*250
                        projected_2d, mat_r = self.__render_cube()
                        projected_2d = np.int32(position+scale*projected_2d)
                        for point in projected_2d[:8]:
                            cv2.circle(canvas, point, 3, (0,0,0,255), -1)
                        
                        for i in range(4):
                            cv2.line(canvas, projected_2d[i], projected_2d[(i+1)%4], (0,0,0,255), 2)
                            cv2.line(canvas, projected_2d[4+i], projected_2d[4+(i+1)%4], (0,0,0,255), 2)
                            cv2.line(canvas, projected_2d[i], projected_2d[i+4], (0,0,0,255), 2)

                        # ### roof
                        # roof_points = np.array([projected_2d[i]-(0,200) for i in [0,1,5,4]], np.int32)
                        # for i, point in enumerate(roof_points):
                        #     cv2.circle(canvas, point, 3, (0,0,0,255), -1)
                        # cv2.fillPoly(canvas, [roof_points], (255,0,255,50))
                            
                        if self.selected_wall > 0:
                            if self.selected_wall==1:
                                cv2.fillPoly(canvas, [projected_2d[:4]], (255,0,0,50))
                                cv2.fillPoly(canvas, [np.array([projected_2d[0],projected_2d[3],projected_2d[7],projected_2d[4]])], (0,0,255,50))
                                cv2.fillPoly(canvas, [np.array([projected_2d[1],projected_2d[2],projected_2d[6],projected_2d[5]])], (0,255,0,50))
                                cv2.fillPoly(canvas, [projected_2d[4:8]], (255,255,0,200))
                                cv2.line(canvas, projected_2d[4], projected_2d[7], (0,0,0,255), 5)
                                cv2.line(canvas, projected_2d[6], projected_2d[7], (0,0,0,255), 5)
                                cv2.putText(canvas, 'X', (projected_2d[6]+projected_2d[7])//2+(0,20), 1, 1, (0,0,0,255))
                                cv2.putText(canvas, 'Y', (projected_2d[4]+projected_2d[7])//2-(20,0), 1, 1, (0,0,0,255))
                            elif self.selected_wall==2:
                                cv2.fillPoly(canvas, [np.array([projected_2d[0],projected_2d[3],projected_2d[7],projected_2d[4]])], (0,0,255,50))
                                cv2.fillPoly(canvas, [projected_2d[4:8]], (255,255,0,50))
                                cv2.fillPoly(canvas, [projected_2d[:4]], (255,0,0,50))
                                cv2.fillPoly(canvas, [np.array([projected_2d[1],projected_2d[2],projected_2d[6],projected_2d[5]])], (0,255,0,200))
                                cv2.line(canvas, projected_2d[5], projected_2d[6], (0,0,0,255), 5)
                                cv2.line(canvas, projected_2d[6], projected_2d[2], (0,0,0,255), 5)
                                cv2.putText(canvas, 'X', (projected_2d[6]+projected_2d[2])//2+(0,20), 1, 1, (0,0,0,255))
                                cv2.putText(canvas, 'Y', (projected_2d[5]+projected_2d[6])//2-(20,0), 1, 1, (0,0,0,255))
                            elif self.selected_wall==3:
                                cv2.fillPoly(canvas, [np.array([projected_2d[1],projected_2d[2],projected_2d[6],projected_2d[5]])], (0,255,0,50))
                                cv2.fillPoly(canvas, [projected_2d[:4]], (255,0,0,50))
                                cv2.fillPoly(canvas, [projected_2d[4:8]], (255,255,0,50))
                                cv2.fillPoly(canvas, [np.array([projected_2d[0],projected_2d[3],projected_2d[7],projected_2d[4]])], (0,0,255,200))
                                cv2.line(canvas, projected_2d[0], projected_2d[3], (0,0,0,255), 5)
                                cv2.line(canvas, projected_2d[3], projected_2d[7], (0,0,0,255), 5)
                                cv2.putText(canvas, 'X', (projected_2d[3]+projected_2d[7])//2+(0,20), 1, 1, (0,0,0,255))
                                cv2.putText(canvas, 'Y', (projected_2d[0]+projected_2d[3])//2-(20,0), 1, 1, (0,0,0,255))
                            elif self.selected_wall==4:
                                cv2.fillPoly(canvas, [projected_2d[4:8]], (255,255,0,50))
                                cv2.fillPoly(canvas, [np.array([projected_2d[1],projected_2d[2],projected_2d[6],projected_2d[5]])], (0,255,0,50))
                                cv2.fillPoly(canvas, [np.array([projected_2d[0],projected_2d[3],projected_2d[7],projected_2d[4]])], (0,0,255,50))
                                cv2.fillPoly(canvas, [projected_2d[:4]], (255,0,0,200))
                                cv2.line(canvas, projected_2d[1], projected_2d[2], (0,0,0,255), 5)
                                cv2.line(canvas, projected_2d[2], projected_2d[3], (0,0,0,255), 5)
                                cv2.putText(canvas, 'X', (projected_2d[2]+projected_2d[3])//2+(0,20), 1, 1, (0,0,0,255))
                                cv2.putText(canvas, 'Y', (projected_2d[1]+projected_2d[2])//2-(20,0), 1, 1, (0,0,0,255))


                            for k, v in self.camera_config_list.items():
                                if k in self.camera_pos.keys():
                                    # pos = np.dot(v['pos'], mat_r)[:2]
                                    pos = np.dot(self.camera_pos[k], mat_r)[:2]
                                    pos = np.int32(position+scale*pos)
                                    cv2.circle(canvas, pos, 10, (50,50,50,150), )    
                            # pos = np.dot(self.camera_pos, mat_r)[:2]
                            # pos = np.int32(position+scale*pos)
                            # cv2.circle(canvas, pos, 10, (0,0,0,255), 5)
                        else:
                            cv2.fillPoly(canvas, [projected_2d[:4]], (255,0,0,50))
                            cv2.fillPoly(canvas, [np.array([projected_2d[0],projected_2d[3],projected_2d[7],projected_2d[4]])], (0,0,255,50))
                            cv2.fillPoly(canvas, [np.array([projected_2d[1],projected_2d[2],projected_2d[6],projected_2d[5]])], (0,255,0,50))
                            cv2.fillPoly(canvas, [projected_2d[4:8]], (255,255,0,50))

                        x_ = np.min([x[0] for x in projected_2d[:8]]) - 10
                        w_ = np.max([x[0] for x in projected_2d[:8]]) - x_ +20
                        y_ = np.min([x[1] for x in projected_2d[:8]])-10
                        h_ = np.max([x[1] for x in projected_2d[:8]]) - y_+20
                        if (w_ > h_):
                            y1 = max(0, y_ - (w_-h_)//2)
                            y2 = y_+h_+(w_-h_)//2
                            x1 = max(0, x_)
                            x2 = x_+w_
                            canvas = canvas[y1:y2, x1:]
                        else:
                            y1 = max(0, y_)
                            y2 = y_+h_
                            x1 = max(0, x_ - (h_-w_)//2)
                            x2 = x_+w_+(h_-w_)//2
                            canvas = canvas[y1:y2, x1:x2]
                        canvas = cv2.resize(canvas, (int(3*screen.get_width()//10), int(3*screen.get_width()//10)))

                        screen.blit(pygame.image.frombuffer(canvas.tobytes(), (canvas.shape[1],canvas.shape[0]), "RGBA"), (screen.get_width()-canvas.shape[1]-50,100))

                        self.project_name_lbl.draw(screen)
                        self.project_name_inp.draw(screen, self.events)

                        self.width_lbl.draw(screen)
                        self.width_inp.draw(screen, self.events)
                        self.height_lbl.draw(screen)
                        self.height_inp.draw(screen, self.events)
                        self.depth_lbl.draw(screen)
                        self.depth_inp.draw(screen, self.events)

                        self.primary_lbl.draw(screen)
                        self.name_lbl_1.draw(screen)
                        self.name_inp_1.draw(screen, self.events)
                        self.wall_lbl_1.draw(screen)
                        self.wall1_btn_1.draw(screen)
                        self.wall2_btn_1.draw(screen)
                        self.wall3_btn_1.draw(screen)
                        self.wall4_btn_1.draw(screen)
                        self.camera_height_lbl_1.draw(screen)
                        self.camera_height_inp_1.draw(screen, self.events)
                        self.camera_width_lbl_1.draw(screen)
                        self.camera_width_inp_1.draw(screen, self.events)
                        self.secondary_lbl.draw(screen)
                        self.name_lbl_2.draw(screen)
                        self.name_inp_2.draw(screen, self.events)
                        self.wall_lbl_2.draw(screen)
                        self.wall1_btn_2.draw(screen)
                        self.wall2_btn_2.draw(screen)
                        self.wall3_btn_2.draw(screen)
                        self.wall4_btn_2.draw(screen)
                        self.camera_height_lbl_2.draw(screen)
                        self.camera_height_inp_2.draw(screen, self.events)
                        self.camera_width_lbl_2.draw(screen)
                        self.camera_width_inp_2.draw(screen, self.events)

                        if self.step==51:
                            self.primary_browse_lbl.draw(screen)
                            self.primary_browse_btn.draw(screen)
                            self.primary_yaml_lbl.draw(screen)
                            self.secondary_browse_lbl.draw(screen)
                            self.secondary_browse_btn.draw(screen)
                            self.secondary_yaml_lbl.draw(screen)

                        self.next_step_btn.clickable = self.__validation()
                        self.next_step_btn.draw(screen)
                    elif self.step == 52:
                        self.video1_btn.draw(screen)
                        self.video2_btn.draw(screen)
                        
                        if self.frame1 is not None:
                            r = self.frame1.shape[1]/self.frame1.shape[0]
                            w = 4*screen.get_width()//10
                            h = w/r
                            x = screen.get_width()/20
                            img1 = cv2.resize(self.frame1, (int(w),int(h)))
                            screen.blit(pygame.image.frombuffer(img1.tobytes(), (img1.shape[1],img1.shape[0]), "RGB"), (x,3*screen.get_height()//20))
                            self.frame_num_lbl.draw(screen)
                        if self.frame2 is not None:
                            r = self.frame2.shape[1]/self.frame2.shape[0]
                            w = 4*screen.get_width()//10
                            h = w/r
                            x = screen.get_width()/20
                            img2 = cv2.resize(self.frame2, (int(w),int(h)))
                            screen.blit(pygame.image.frombuffer(img2.tobytes(), (img2.shape[1],img2.shape[0]), "RGB"), (screen.get_width()-int(w)-x,3*screen.get_height()//20))
                            self.frame_num_lbl.draw(screen)

                        isValid = self.__validation()
                        self.next_frame_btn.clickable = isValid
                        self.select_btn.clickable = isValid
                        self.next_frame_btn.draw(screen)
                        self.select_btn.draw(screen)
                    elif self.step == 53:
                        img1_ = self.frame1.copy()
                        img2_ = self.frame2.copy()
                        for pos1 in self.positions1[:self.pos_i]:
                            cv2.circle(img1_, np.uint16(pos1), 10, (0,255,0), -1)
                        if len(self.positions1)==self.pos_i+1: cv2.circle(img1_, np.uint16(self.positions1[self.pos_i]), 10, (255,0,0), -1)
                        for pos2 in self.positions2[:self.pos_i]:
                            cv2.circle(img2_, np.uint16(pos2), 10, (0,255,0), -1)
                        if len(self.positions2)==self.pos_i+1: cv2.circle(img2_, np.uint16(self.positions2[self.pos_i]), 10, (255,0,0), -1)
                        r = img1_.shape[0]/img1_.shape[1]
                        w = 2*screen.get_width()//5
                        h = w*r
                        img1_ = cv2.resize(img1_, (int(w), int(h)))
                        img2_ = cv2.resize(img2_, (int(w), int(h)))

                        img1_rect = pygame.Rect(int(w/6),50,img1_.shape[1], img1_.shape[0])
                        img2_rect = pygame.Rect(screen.get_width()-w-int(w/6),50,img2_.shape[1], img2_.shape[0])
                        if (not self.no_further_point) and (len(self.positions1) == self.pos_i):
                            mouse_pos = pygame.mouse.get_pos()
                            if img1_rect.collidepoint(mouse_pos):
                                if pygame.mouse.get_pressed()[0]:
                                    self.positions1.append(((mouse_pos[0] - img1_rect.x)*self.frame1.shape[1]/w, (mouse_pos[1]-img1_rect.y)*self.frame1.shape[0]/h))
                        
                        if (not self.no_further_point) and (len(self.positions2) == self.pos_i):
                            mouse_pos = pygame.mouse.get_pos()
                            if img2_rect.collidepoint(mouse_pos):
                                if pygame.mouse.get_pressed()[0]:
                                    self.positions2.append(((mouse_pos[0] - img2_rect.x)*self.frame2.shape[1]/w, (mouse_pos[1]-img2_rect.y)*self.frame2.shape[0]/h))

                        self.img1_lbl.text = f"View of Camera {self.camera_config_list[1]['name']}"
                        self.img2_lbl.text = f"View of Camera {self.camera_config_list[2]['name']}"
                        self.img1_lbl.draw(screen)
                        self.img2_lbl.draw(screen)
                        screen.blit(pygame.image.frombuffer(img1_.tobytes(), (img1_.shape[1],img1_.shape[0]), "RGB"), (int(w/6),50))
                        screen.blit(pygame.image.frombuffer(img2_.tobytes(), (img2_.shape[1],img2_.shape[0]), "RGB"), (screen.get_width()-w-int(w/6),50))

                        if self.done:
                            self.done_lbl.draw(screen)
                        elif self.no_further_point:
                            self.save_trans_path_btn.draw(screen)
                            self.done_lbl.draw(screen)
                        else:
                            if len(self.positions1)==self.pos_i+1 and len(self.positions2)==self.pos_i+1:
                                self.get_point_lbl.text = f"enter world coordinate for point {self.pos_i+1} (at least 3 points, more better):"
                                # self.get_point_lbl.y = h + 100
                                self.get_point_lbl.draw(screen)
                                self.pos_w_lbl.draw(screen)
                                self.pos_w_inp.draw(screen,self.events)
                                self.pos_h_lbl.draw(screen)
                                self.pos_h_inp.draw(screen,self.events)
                                self.pos_d_lbl.draw(screen)
                                self.pos_d_inp.draw(screen,self.events)

                                scale = 0.05
                                position = (200,200)
                                canvas = np.ones((500,500,4), np.uint8)*250
                                canvas2 = np.ones((500,500,4), np.uint8)*0
                                if self.camera_config_list[1]["wall"] == 1:
                                    self.tetha_y = -30
                                    self.tetha_x = -20
                                elif self.camera_config_list[1]["wall"] == 2:
                                    self.tetha_y = - 60
                                    self.tetha_x = -20
                                elif self.camera_config_list[1]["wall"] == 3:
                                    self.tetha_y = 90+ 60
                                    self.tetha_x = 20
                                elif self.camera_config_list[1]["wall"] == 4:
                                    self.tetha_y = 180 - 30
                                    self.tetha_x = 20
                                projected_2d, mat_r = self.__render_cube()
                                projected_2d = np.int32(position+scale*projected_2d)
                                for point in projected_2d[:8]:
                                    cv2.circle(canvas, point, 3, (0,0,0,255), -1)
                                
                                for i in range(4):
                                    cv2.line(canvas, projected_2d[i], projected_2d[(i+1)%4], (0,0,0,255), 2)
                                    cv2.line(canvas, projected_2d[4+i], projected_2d[4+(i+1)%4], (0,0,0,255), 2)
                                    cv2.line(canvas, projected_2d[i], projected_2d[i+4], (0,0,0,255), 2)
                                
                                try:
                                    if (self.pos_w_inp.text != '') and (self.pos_h_inp.text != '') and (self.pos_d_inp.text != ''):
                                        if self.camera_config_list[1]["wall"] == 1:
                                            temp_pos = np.dot([int(self.pos_w)+self.cube_width//2, int(self.pos_h)+self.cube_height//2, int(self.pos_d)-self.cube_depth//2], mat_r)
                                        elif self.camera_config_list[1]["wall"] == 2:
                                            temp_pos = np.dot([int(self.pos_w)+self.cube_width//2, int(self.pos_h)+self.cube_height//2, int(self.pos_d)-self.cube_depth//2], mat_r)
                                        elif self.camera_config_list[1]["wall"] == 3:
                                            temp_pos = np.dot([int(self.pos_w)+self.cube_width//2, int(self.pos_h)+self.cube_height//2, int(self.pos_d)-self.cube_depth//2], mat_r)
                                        elif self.camera_config_list[1]["wall"] == 4:
                                            temp_pos = np.dot([int(self.pos_w)+self.cube_width//2, int(self.pos_h)+self.cube_height//2, int(self.pos_d)-self.cube_depth//2], mat_r)
                                        temp_pos = temp_pos[:2]
                                        temp_pos = np.int32(temp_pos*scale+position)
                                        cv2.circle(canvas, temp_pos, 5, (0,0,255,255), -1)
                                except:
                                    pass

                                if self.camera_config_list[1]["wall"] == 1:
                                    cv2.fillPoly(canvas2, [projected_2d[4:8]], (255,255,0,100))
                                    cv2.putText(canvas, 'X', (projected_2d[6]+projected_2d[7])//2+(0,20), 2, 0.75, (255,0,0,255))
                                    cv2.putText(canvas, 'Y', (projected_2d[4]+projected_2d[7])//2-(20,0), 2, 0.75, (0,150,50,255))
                                    cv2.putText(canvas, 'Z', (projected_2d[6]+projected_2d[2])//2+(0,20), 2, 0.75, (0,0,255,255))
                                elif self.camera_config_list[1]["wall"] == 2:
                                    cv2.fillPoly(canvas2, [np.array([projected_2d[1],projected_2d[2],projected_2d[6],projected_2d[5]])], (0,255,0,100))
                                    cv2.putText(canvas, 'X', (projected_2d[6]+projected_2d[7])//2+(0,20), 2, 1, (255,0,0,255))
                                    cv2.putText(canvas, 'Y', (projected_2d[7]+projected_2d[4])//2-(20,0), 2, 1, (0,150,50,255))
                                    cv2.putText(canvas, 'Z', (projected_2d[2]+projected_2d[6])//2+(0,20), 2, 1, (0,0,255,255))
                                elif self.camera_config_list[1]["wall"] == 3:
                                    cv2.fillPoly(canvas2, [np.array([projected_2d[0],projected_2d[3],projected_2d[7],projected_2d[4]])], (0,0,255,100))
                                    cv2.putText(canvas, 'X', (projected_2d[3]+projected_2d[2])//2+(0,20), 2, 1, (255,0,0,255))
                                    cv2.putText(canvas, 'Y', (projected_2d[2]+projected_2d[1])//2-(20,0), 2, 1, (0,150,50,255))
                                    cv2.putText(canvas, 'Z', (projected_2d[3]+projected_2d[7])//2+(0,20), 2, 1, (0,0,255,255))
                                elif self.camera_config_list[1]["wall"] == 4:
                                    cv2.fillPoly(canvas2, [projected_2d[:4]], (255,0,0,100))
                                    cv2.putText(canvas, 'X', (projected_2d[2]+projected_2d[3])//2+(0,20), 2, 1, (255,0,0,255))
                                    cv2.putText(canvas, 'Y', (projected_2d[1]+projected_2d[2])//2-(20,0), 2, 1, (0,150,50,255))
                                    cv2.putText(canvas, 'Z', (projected_2d[3]+projected_2d[7])//2+(0,20), 2, 1, (0,0,255,255))
                                cv2.circle(canvas, projected_2d[6], 5, (255,0,0,255))

                                cv2.line(canvas, projected_2d[6], projected_2d[8], (255,0,0,255), 3)
                                cv2.line(canvas, projected_2d[6], projected_2d[9], (0,255,0,255), 3)
                                cv2.line(canvas, projected_2d[6], projected_2d[10], (0,0,255,255), 3)

                                x_ = np.min([x[0] for x in projected_2d]) -30
                                w_ = np.max([x[0] for x in projected_2d]) - x_ +60
                                y_ = np.min([x[1] for x in projected_2d])
                                h_ = np.max([x[1] for x in projected_2d]) - y_+30
                                if (w_ > h_):
                                    canvas = canvas[y_ - (w_-h_)//2:y_+h_+(w_-h_)//2, x_:x_+w_]
                                    canvas2 = canvas2[y_ - (w_-h_)//2:y_+h_+(w_-h_)//2, x_:x_+w_]
                                else:
                                    canvas = canvas[y_:y_+h_, x_ - (h_-w_)//2: x_+w_+(h_-w_)//2]
                                    canvas2 = canvas2[y_:y_+h_, x_ - (h_-w_)//2: x_+w_+(h_-w_)//2]
                                canvas = cv2.resize(canvas, (int(w/3), int(w/3)))
                                canvas2 = cv2.resize(canvas2, (int(w/3), int(w/3)))

                                screen.blit(pygame.image.frombuffer(canvas.tobytes(), (canvas.shape[1],canvas.shape[0]), "RGBA"), (screen.get_width()//5,(h+100)+(screen.get_height()-h-100-w//3)//2))
                                screen.blit(pygame.image.frombuffer(canvas2.tobytes(), (canvas.shape[1],canvas.shape[0]), "RGBA"), (screen.get_width()//5,(h+100)+(screen.get_height()-h-100-w//3)//2))

                                self.next_btn.clickable = self.__validation()
                                self.next_btn.draw(screen)
                                self.reset_btn.draw(screen)
                            else:
                                self.get_point_lbl.text = f"Point number {self.pos_i+1} (select identity point on two frames):"
                                self.get_point_lbl.x = (screen.get_width()-self.get_point_lbl.get_width())//2
                                self.get_point_lbl.y = img1_.shape[0]+50+screen.get_height()//20
                                self.get_point_lbl.draw(screen)
                                if self.pos_i>=3:
                                    self.done_btn.draw(screen)
                    
                    if self.step == 60: # 3D visualization
                        self.browse_2d_kp_btn1.clickable = self.add_pair
                        self.browse_2d_kp_btn2.clickable = self.add_pair
                        self.add_calibration_btn.clickable = False
                        self.add_pair_btn.clickable = not self.add_pair and (len(self.debug_video_list)==2)
                        self.reconstruct_3d_btn.clickable = not self.add_pair and (len(self.debug_video_list)==2) and (len(self.monkey_names)>0) and (self.save_path!='')
                        if len(self.kp_files_list) > 0 and len(self.kp_files_list)%2==0:
                            self.browse_2d_kp_btn1.clickable = False
                            self.browse_2d_kp_btn2.clickable = False
                            self.add_calibration_btn.clickable = self.add_pair
                        if len(self.reconstruction_files) == len(self.kp_files_list)/2:
                            self.browse_2d_kp_btn1.clickable = self.add_pair
                            self.browse_2d_kp_btn2.clickable = self.add_pair
                            self.add_calibration_btn.clickable = False
                        if len(self.kp_files_list)==2 and len(self.debug_video_list)<2:
                            self.browse_video_btn1.draw(screen)
                            self.browse_vider_btn2.draw(screen)
                        else:
                            self.browse_2d_kp_btn1.draw(screen)
                            self.browse_2d_kp_btn2.draw(screen)
                        self.kp_files_list_lbl.draw(screen)
                        self.debug_video_list_lbl.draw(screen)
                        self.add_calibration_btn.draw(screen)
                        self.calibraion_list_lbl.draw(screen)
                        self.add_pair_btn.draw(screen)

                        self.browse_identifier_model_btn.draw(screen)
                        for ckb in self.monkey_names_ckb_list:
                            ckb.render_checkbox()
                        self.save_3D_video_ckb.render_checkbox()
                        self.output_fps_inp.draw(screen, self.events)
                        self.output_fps_lbl.draw(screen)
                        self.save_seperately_ckb.render_checkbox()
                        self.save_path_btn.draw(screen)
                        self.save_path_lbl.draw(screen)
                        
                        self.reconstruct_3d_btn.draw(screen)

                        self.back_to_menu_btn.draw(screen)
                        
                        self.browse_identifier_model_hink_tk.draw()
                        if self.wait:
                            s = pygame.Surface((screen.get_width(),screen.get_height()))  # the size of your rect
                            s.set_alpha(220)                # alpha level
                            s.fill((50,50,50))           # this fills the entire surface
                            screen.blit(s, (0,0)) 
                            font = pygame.font.Font(None, 92)
                            text = font.render("Please Wait ...", True, (255,255,255))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                            if self.identifier_loading:
                                self.__wait_for_process(process=self.__parse_identifier_model)
                            else:
                                self.__wait_for_process(process=self.__convert_2d_3d)
                    elif self.step == 61:
                        if self.save_3d_video:
                            try:
                                canvas, frame_i = next(self.frame_generator)
                                r = canvas.shape[1] / canvas.shape[0]
                                h = screen.get_height()
                                w = r * h
                                canvas = cv2.resize(canvas, (int(w),int(h)))
                                temp = np.ones((canvas.shape[0], canvas.shape[1]+100, 3), np.uint8)*255
                                temp[:canvas.shape[0], :canvas.shape[1]] = canvas
                                for mnk_i, mnk in enumerate(self.monkey_names):
                                    cv2.putText(temp, mnk, (canvas.shape[1]-100, 100+50*mnk_i), 2, 1, self.colors[mnk_i], 2)
                                screen.blit(pygame.image.frombuffer(temp.tobytes(), (temp.shape[1],temp.shape[0]), "BGR"), (screen.get_width()-temp.shape[1]-100,0))

                                font = pygame.font.Font(None, 36)
                                text = font.render(f"frame number {frame_i}", True, (50,50,50))
                                r = text.get_rect()
                                screen.blit(text, (50+150, 20))

                                try:
                                    f1 = self.video_debug1.get_frame()
                                    f2 = self.video_debug2.get_frame()
                                except:
                                    # print("failed to read frame")
                                    f1 = None
                                    f2 = None
                                if (f1 is not None) and (f2 is not None):
                                    h,w,_ = f1.shape
                                    h_ = screen.get_height()//3
                                    w_ = int((w/h)*h_)
                                    if w_ >  screen.get_width()-canvas.shape[1]-100:
                                        w_ = screen.get_width()-canvas.shape[1]-100-100
                                        h_ = int((h/w)*w_)
                                    f1 = cv2.resize(f1, (w_,h_))
                                    f2 = cv2.resize(f2, (w_,h_))
                                    screen.blit(pygame.image.frombuffer(f1.tobytes(), (w_,h_), "BGR"), (50, 100))
                                    screen.blit(pygame.image.frombuffer(f2.tobytes(), (w_,h_), "BGR"), (50, h_+50+100))

                                if self.video_writer is not None:
                                    tmp = pygame.surfarray.array3d(screen)
                                    tmp = tmp.swapaxes(0,1)
                                    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
                                    self.video_writer.write(tmp)

                                self.back_to_menu_btn.draw(screen)
                            except Exception as e:
                                font = pygame.font.Font(None, 44)
                                text = font.render("Congratulations! Process is done!", True, (50,50,100))
                                r = text.get_rect()
                                screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                                # text = font.render(f"3D keypoints are saved. {os.path.join(self.save_path, 'output.mp4')+' is saved.' if (self.save_3d_video) else ''}", True, (50,50,100))
                                text = font.render( f"3D keypoints are saved {self.save_path}", True, (50, 50, 100))
                                r = text.get_rect()
                                screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2+50))
                                if self.save_3d_video:
                                    self.video_writer.release()
                                    text = font.render(f"output.mp4 is saved {self.save_path}", True, (50,50,100))
                                    r = text.get_rect()
                                    screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2+100))
                        else:
                            font = pygame.font.Font(None, 44)
                            text = font.render("Process is done!", True, (50,50,100))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                            # text = font.render(f"3D keypoints are saved. {os.path.join(self.save_path, 'output.mp4')+' is saved.' if (self.save_3d_video) else ''}", True, (50,50,100))
                            text = font.render( f"3D keypoints are saved {self.save_path}", True, (50, 50, 100))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2+50))

                    if self.step == 70:
                        self.vis_3d_btn.clickable = (len(self.monkey_names)>0) and all(x in self.monkey_3d_kpt_files.keys() for x in self.monkey_names) and ((not self.save_3d_video) or (self.save_3d_video and self.save_path!=''))
                        self.width_vis_inp.draw(screen, self.events)
                        self.width_vis_lbl.draw(screen)
                        self.height_vis_inp.draw(screen, self.events)
                        self.height_vis_lbl.draw(screen)
                        self.depth_vis_inp.draw(screen, self.events)
                        self.depth_vis_lbl.draw(screen)
                        self.browse_identifier_model_vis_btn.draw(screen)
                        self.browse_identifier_model_vis_hink_tk.draw()
                        self.save_3d_vis_ckb.render_checkbox()
                        self.output_fps_vis_inp.draw(screen, self.events)
                        self.output_fps_vis_lbl.draw(screen)
                        self.save_path_vis_btn.draw(screen)
                        self.save_path_vis_lbl.draw(screen)
                        self.add_guide_video_btn.clickable = len(self.debug_video_list)<2
                        self.add_guide_video_btn.draw(screen)
                        self.add_guide_video_lbl.draw(screen)
                        self.debug_video_list_vis_lbl.draw(screen)
                        self.vis_3d_btn.draw(screen)

                        for row in self.monkey_names_ckb_to_select_list:
                            row[0].render_checkbox()
                            row[1].draw(screen)
                            row[2].draw(screen)

                        if self.wait:
                            s = pygame.Surface((screen.get_width(),screen.get_height()))  # the size of your rect
                            s.set_alpha(220)                # alpha level
                            s.fill((50,50,50))           # this fills the entire surface
                            screen.blit(s, (0,0)) 
                            font = pygame.font.Font(None, 92)
                            text = font.render("Please Wait ...", True, (255,255,255))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                            self.__wait_for_process(process=self.__parse_identifier_model2)
                    elif self.step == 71:
                        try:
                            canvas, frame_i = next(self.frame_generator)
                            r = canvas.shape[1] / canvas.shape[0]
                            h = screen.get_height()
                            w = r * h
                            canvas = cv2.resize(canvas, (int(w),int(h)))
                            temp = np.ones((canvas.shape[0], canvas.shape[1]+100, 3), np.uint8)*255
                            temp[:canvas.shape[0], :canvas.shape[1]] = canvas
                            for mnk_i, mnk in enumerate(self.monkey_names):
                                cv2.putText(temp, mnk, (canvas.shape[1]-100, 100+50*mnk_i), 2, 1, self.colors[mnk_i], 2)
                            screen.blit(pygame.image.frombuffer(temp.tobytes(), (temp.shape[1],temp.shape[0]), "BGR"), (screen.get_width()-temp.shape[1]-100,0))

                            font = pygame.font.Font(None, 36)
                            text = font.render(f"frame number {frame_i}", True, (50,50,50))
                            r = text.get_rect()
                            screen.blit(text, (50+150, 20))

                            try:
                                f1 = self.video_debug1.get_frame()
                                h,w,_ = f1.shape
                                h_ = screen.get_height()//3
                                w_ = int((w/h)*h_)
                                if w_ >  screen.get_width()-canvas.shape[1]-100:
                                    w_ = screen.get_width()-canvas.shape[1]-100-100
                                    h_ = int((h/w)*w_)
                                f1 = cv2.resize(f1, (w_,h_))
                                screen.blit(pygame.image.frombuffer(f1.tobytes(), (w_,h_), "BGR"), (50, 100))
                            except:
                                f1 = None
                            try:
                                f2 = self.video_debug2.get_frame()
                                h,w,_ = f2.shape
                                h_ = screen.get_height()//3
                                w_ = int((w/h)*h_)
                                if w_ >  screen.get_width()-canvas.shape[1]-100:
                                    w_ = screen.get_width()-canvas.shape[1]-100-100
                                    h_ = int((h/w)*w_)
                                f2 = cv2.resize(f2, (w_,h_))
                                screen.blit(pygame.image.frombuffer(f2.tobytes(), (w_,h_), "BGR"), (50, h_+50+100))
                            except:
                                f2 = None

                            if self.video_writer is not None:
                                tmp = pygame.surfarray.array3d(screen)
                                tmp = tmp.swapaxes(0,1)
                                tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
                                self.video_writer.write(tmp)

                            self.back_to_menu_btn.draw(screen)

                        except Exception as e:
                            print(e)
                            font = pygame.font.Font(None, 44)
                            text = font.render("Process is done!", True, (50,50,100))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                            if self.save_3d_video:
                                self.video_writer.release()
                                text = font.render(f"output.mp4 is saved {self.save_path}", True, (50,50,100))
                                r = text.get_rect()
                                screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2+100))
                        
                            self.__wait_for_process(self.__initiate_2d_process)
                    
                    if self.step ==80:    # save camera stream - getting camera parameters

                        # check whether all fields are field or not to enable set parameter button
                        if ((self.exposure_inp.text!='' or self.auto_exposure_ckb.checked) and (self.gain_inp.text!='' or self.auto_gain_ckb.checked) and self.frame_rate_inp.text!='') and (self.trigger_option_lst.selected_option==0 or (self.action_device_key_inp.text!='' and self.action_group_key_inp.text!='' and self.action_group_mask_inp.text!='')):
                            self.set_param_btn.clickable = True
                        else:
                            self.set_param_btn.clickable = False
        
                        if self.b_is_run and (self.save_option_rb.selected_option == "No Save" or self.save_path!=''):
                            self.initialize_save_btn.clickable = True
                        else:
                            self.initialize_save_btn.clickable = False
                        if self.save_option_rb.selected_option == "No Save":
                            self.save_param_project_name_inp.enable = False
                            self.save_param_session_inp.enable = False
                            self.save_output_btn.clickable = False
                        else:
                            self.save_param_project_name_inp.enable = True
                            self.save_param_session_inp.enable = True
                            self.save_output_btn.clickable = True
                        self.search_camera_btn.draw(screen)
                        self.no_camera_lbl.draw(screen)
                        self.select_camera_lbl.draw(screen)
                        self.trigger_option_lst.update(self.events)
                        self.trigger_option_lst.draw(screen)
                        self.trigger_option_lbl.draw(screen)
                        self.open_cameras_btn.draw(screen)
                        self.general_param_lbl.draw(screen)
                        self.frame_rate_inp.draw(screen, self.events)
                        self.frame_rate_lbl.draw(screen)
                        self.binning_rb.update()
                        self.binning_rb.draw()
                        self.binning_lbl.draw(screen)
                        self.action_device_key_inp.draw(screen, self.events)
                        self.action_device_key_lbl.draw(screen)
                        self.action_group_key_inp.draw(screen, self.events)
                        self.action_group_key_lbl.draw(screen)
                        self.action_group_mask_inp.draw(screen, self.events)
                        self.action_group_mask_lbl.draw(screen)
                        self.specific_param_lbl.draw(screen)
                        self.vertical_flip_ckb.render_checkbox()
                        self.horizental_flip_ckb.render_checkbox()
                        self.apply_flip_lbl.draw(screen)
                        self.apply_flip_lst.update(self.events)
                        self.apply_flip_lst.draw(screen)
                        self.exposure_inp.draw(screen, self.events)
                        self.auto_exposure_ckb.render_checkbox()
                        self.exposure_lbl.draw(screen)
                        self.gain_inp.draw(screen, self.events)
                        self.auto_gain_ckb.render_checkbox()
                        self.gain_lbl.draw(screen)
                        self.save_option_rb.update()
                        self.save_option_rb.draw()
                        self.save_option_lbl.draw(screen)
                        self.save_output_btn.draw(screen)
                        self.save_output_lbl.draw(screen)
                        self.save_param_project_name_lbl.draw(screen)
                        self.save_param_project_name_inp.draw(screen, self.events)
                        self.save_param_session_lbl.draw(screen)
                        self.save_param_session_inp.draw(screen, self.events)
                        self.get_param_btn.draw(screen)
                        self.set_param_btn.draw(screen)
                        self.load_param_btn.draw(screen)
                        self.save_param_btn.draw(screen)
                        self.camera_result_lst.update(self.events)
                        self.camera_result_lst.draw(screen)
                        # self.show_stream_ckb.render_checkbox()
                        # self.save_path_btn.draw(screen)
                        # self.save_path_lbl.draw(screen)
                        self.initialize_save_btn.draw(screen)
                        if self.wait:
                            s = pygame.Surface((screen.get_width(),screen.get_height()))  # the size of your rect
                            s.set_alpha(220)                # alpha level
                            s.fill((50,50,50))           # this fills the entire surface
                            screen.blit(s, (0,0)) 
                            font = pygame.font.Font(None, 92)
                            text = font.render("Please Wait ...", True, (255,255,255))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                            self.__wait_for_process(process=self.__open_camera)
                    elif self.step == 81:   # show camera stream (if selected), announced how many frames have been saved
                        self.back_to_params_btn.draw(screen)
                        self.start_grabbing_btn.draw(screen)
                        self.stop_grabbing_btn.draw(screen)
                        self.scheduled_ckb.render_checkbox()
                        self.start_time_inp.draw(screen, self.events)
                        self.start_time_title_lbl.draw(screen)
                        self.start_time_lbl.draw(screen)
                        self.end_time_inp.draw(screen, self.events)
                        self.end_time_title_lbl.draw(screen)
                        self.end_time_lbl.draw(screen)
                        self.every_day_ckb.render_checkbox()
                        # print("---------------------------------\n ", self.grabbingRunning, self.done)
                        if self.done:
                            if self.save_option_rb.selected_option == "No Save" or self.done_save_sync.value:
                                for stream in self.camera_stream:
                                    stream.reset()
                                if self.display_thread is not None:
                                    self.display_thread.join()
                                self.grabbingRunning = False
                                font = pygame.font.Font(None, 108)
                                text = font.render("Process is done!", True, (50,50,100))
                                r = text.get_rect()
                                screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2-50))
                                self.start_grabbing_btn.clickable = True
                                self.back_to_params_btn.clickable = True
                                self.back_to_menu_btn.clickable=True
                                self.scheduled_ckb.enable = True
                                if self.scheduled_ckb.checked:
                                    self.start_time_inp.enable = True
                                    self.end_time_inp.enable = True
                                    self.every_day_ckb.enable = True
                                if self.database is not None:
                                    self.database['record_num_frames'] = self.saved_sync_frames.value / len(self.camera_stream)
                                    self.database['record_end_time'] = datetime.now().strftime('%H:%M:%S')
                                    self.database.to_csv(os.path.join(self.save_path,f'database_{self.save_param_project_name_inp.text}_{self.save_param_session_inp.text}_{self.start_time_for_database}.csv'), index=False)
                            else:
                                self.processing_lbl.text = f"{self.recieved_sync_frames.value} frames recieved. {self.show_sync_frames} frames displayed, {self.drop_sync_frames.value} frames dropped, {self.saved_sync_frames.value} frames saved."
                                self.processing_lbl.draw(screen)
                        elif self.grabbingRunning:
                            new_frame = False
                            if self.sync_frames_queue.qsize() > 0:
                                frames = self.sync_frames_queue.get()
                                new_frame = True
                            try:
                                if frames is None:
                                    if not self.is_running_timing:
                                        self.done = True
                                    else:
                                        for stream in self.camera_stream:
                                            stream.reset()
                                        self.stop_grabbing_btn.clickable = True
                                    # self.recieved_sync_frames.value-=1
                                    self.grabbingRunning = False
                                    del frames
                                    continue
                                
                                n,x,w,h = self.multi_plane_screen_info
                                for i in range(len(frames)):
                                    font = pygame.font.Font(None, int(34-2*n))
                                    img = frames[i]

                                    if isinstance(img, str) and img=='NO_IMAGE':
                                        text = font.render(self.devList[i]+' - NO DATA', True, (200,50,50))
                                    else:
                                        if new_frame:
                                            self.show_sync_frames+=1
                                            img = cv2.resize(frames[i], (int(w),int(h)))
                                            frames[i] = img
                                        screen.blit(pygame.image.frombuffer(img.tobytes(), (img.shape[1],img.shape[0]), "RGB"), (x+(i%n)*(w+.01*screen.get_height()),.05*screen.get_height()+(i//n)*(h+.01*screen.get_height())))
                                        text = font.render(self.devList[i], True, (50,200,50))
                                    screen.blit(text, (x+(i%n)*(w+.01*screen.get_height())+10,.05*screen.get_height()+(i//n)*(h+.01*screen.get_height())+10))
                                self.processing_lbl.text = f"{self.recieved_sync_frames.value} frames recieved. {self.show_sync_frames} frames displayed, {self.drop_sync_frames.value} frames dropped, {self.saved_sync_frames.value} frames saved."
                                self.processing_lbl.draw(screen)
                            except Exception as e:
                                pass
                        elif self.is_running_timing and not self.in_time_range:
                            font = pygame.font.Font(None, 108)
                            text = font.render("Waiting ...", True, (50,50,100))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2-50))

                    if self.step == 90:
                        self.show_recieved_frames_ckb.update()
                        self.show_recieved_frames_ckb.render_checkbox()
                        self.save_recieved_video_path_btn.draw(screen)
                        self.save_recieved_video_path_lbl.draw(screen)
                        self.server_ip_inp.draw(screen, self.events)
                        self.server_ip_lbl.draw(screen)
                        self.initiate_server_btn.draw(screen)
                        self.initiate_server_lbl.draw(screen)
                        if self.wait:
                            s = pygame.Surface((screen.get_width(),screen.get_height()))  # the size of your rect
                            s.set_alpha(220)                # alpha level
                            s.fill((50,50,50))           # this fills the entire surface
                            screen.blit(s, (0,0)) 
                            font = pygame.font.Font(None, 92)
                            text = font.render("Please Wait ...", True, (255,255,255))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                            self.__wait_for_process(process=self.__init_server)
                        if self.server_socket is not None and len(self.credentials_dict['ip'])>0 and self.save_recieved_video_path_lbl.text!='':
                            self.save_server_config_btn.clickable = True
                        else:
                            self.save_server_config_btn.clickable = False

                        if self.server_socket is not None and len(self.ip_ckbs)>0 and np.array([ch.checked for ch in self.ip_ckbs]).any() and np.array([self.credentials_dict['status'][i] for i in range(len(self.credentials_dict['status'])) if self.ip_ckbs[i].checked]).all() and self.save_recieved_video_path_lbl.text!='':
                            self.recieve_from_jetsons_btn.clickable = True
                        else:
                            self.recieve_from_jetsons_btn.clickable = False
                        self.get_client_addresses_btn.draw(screen)
                        self.get_client_addresses_lbl.draw(screen)
                        self.recieve_from_jetsons_btn.draw(screen)
                        self.save_server_config_btn.draw(screen)
                        self.load_server_config_btn.draw(screen)
                        w = screen.get_width()
                        y = screen.get_height()/20
                        for i, ip in enumerate(self.credentials_dict['ip']):
                            self.ip_ckbs[i].update()
                            self.ip_ckbs[i].render_checkbox()
                            # ip_box = Label(w/50+100, 6*y+i*2*y, text=ip)
                            # ip_box.draw(screen)
                            on_status = Label(w/5, 6*y+i*2*y, text="Jetson is On" if self.credentials_dict['status'][i][0] else "Jetson is Off", color=(50,200, 50) if self.credentials_dict['status'][i][0] else (200, 50, 50))
                            run_status = Label(2*w/5, 6*y+i*2*y, text="Client script is running on Jeston" if self.credentials_dict['status'][i][1] else "Client script is not running on Jetson", color=(50,200, 50) if self.credentials_dict['status'][i][1] else (200, 50, 50))
                            connection_status = Label(3*w/5, 6*y+i*2*y, text="Jetson is Connected." if self.credentials_dict['status'][i][2] else "Jetson is Disconnected.", color=(50,200, 50) if self.credentials_dict['status'][i][2] else (200, 50, 50))
                            on_status.draw(screen)
                            run_status.draw(screen)
                            connection_status.draw(screen)
                            self.jetson_check_btn[i][0].draw(screen)
                            self.jetson_check_btn[i][1].draw(screen)

                    elif self.step==91:
                        if self.select_video_csv_lbl.text!="" and self.box_weight!='' and (not self.identification or (self.identification and self.identification_weight!='')):
                            self.send_to_jetson_btn.clickable = True
                            self.save_param_config_btn.clickable = True
                        else:
                            self.send_to_jetson_btn.clickable = False
                            self.save_param_config_btn.clickable = False
                        self.select_video_csv_btn.draw(screen)
                        self.select_video_csv_lbl.draw(screen)
                        self.select_video_csv_hint_lbl.draw(screen)
                        self.browse_box_pose_weight_btn.draw(screen)
                        self.browse_box_pose_weight_lbl.draw(screen)
                        self.identification_ckb.render_checkbox()
                        self.browse_identification_weight_btn.draw(screen)
                        self.browse_identification_weight_lbl.draw(screen)
                        self.monkey_num_inp.draw(screen, self.events)
                        self.monkey_num_lbl.draw(screen)
                        self.box_conf_inp.draw(screen, self.events)
                        self.box_conf_lbl.draw(screen)
                        self.iou_thresh_inp.draw(screen, self.events)
                        self.iou_thresh_lbl.draw(screen)
                        self.kpt_conf_inp.draw(screen, self.events)
                        self.kpt_conf_lbl.draw(screen)
                        
                        self.save_param_config_btn.draw(screen)
                        self.load_param_config_btn.draw(screen)
                        self.send_to_jetson_btn.draw(screen)
                        self.kpt_conf_hint_tk.draw()
                        self.iou_thresh_hint_tk.draw()
                        self.box_conf_hint_tk.draw()
                        if self.wait:
                            s = pygame.Surface((screen.get_width(),screen.get_height()))  # the size of your rect
                            s.set_alpha(220)                # alpha level
                            s.fill((50,50,50))           # this fills the entire surface
                            screen.blit(s, (0,0)) 
                            font = pygame.font.Font(None, 92)
                            text = font.render("Please Wait ...", True, (255,255,255))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                            self.__wait_for_process(process=self.__send_parameters_jetson)
                    elif self.step == 92:
                        self.send_frame_to_jetson_btn.draw(screen)
                        self.video_num_to_process_lbl.draw(screen)
                        self.stop_send_frame_to_jetson_btn.draw(screen)
                        self.scheduled_jetson_module_ckb.render_checkbox()
                        self.start_time_inp.draw(screen, self.events)
                        self.start_time_title_lbl.draw(screen)
                        self.start_time_lbl.draw(screen)
                        self.every_day_jetson_ckb.render_checkbox()

                        n = np.ceil(np.sqrt(len(self.clients.keys())))
                        h = .8*screen.get_height()/n - (n-1)*.01*screen.get_height()
                        w = (4/3)*h
                        x = (screen.get_width()-(n*w)-((n-1)*.01*screen.get_height()))/2  
                        try:
                            if self.done:
                                font = pygame.font.Font(None, 108)
                                text = font.render("Process has done.", True, (50,50,100))
                                r = text.get_rect()
                                screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2-50))
                            elif self.is_running_timing and not self.processingRunning:
                                font = pygame.font.Font(None, 108)
                                text = font.render("Waiting ...", True, (50,50,100))
                                r = text.get_rect()
                                screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2-50))
                            
                            else:
                                if self.show_recieved_frames_ckb.checked:
                                    try:  
                                        for i, idx in enumerate(self.clients.keys()):
                                            if idx in self.show_received_frames and self.show_received_frames[idx] is not None:
                                                f = self.show_received_frames[idx]
                                                f = cv2.resize(f, (int(w), int(h)))
                                                screen.blit(pygame.image.frombuffer(f.tobytes(), (f.shape[1],f.shape[0]), "RGB"), (x+(i%n)*(w+.01*screen.get_height()),.05*screen.get_height()+(i//n)*(h+.01*screen.get_height())))
                                    except Exception as e:
                                        print(e)
                                        traceback.print_exc()
                                        pass
                                font = pygame.font.Font(None, int(32/(np.ceil(np.sqrt(len(self.clients.keys()))))))
                                for i, j_i in enumerate(self.clients.keys()):
                                    text = font.render(f"Jetson {self.clients[j_i][1].getpeername()[0]} processed {self.received_frame_idx[j_i]} frames of {os.path.basename(self.current_processing_video[j_i])}.", True, (50,50,100))
                                    r = text.get_rect()
                                    screen.blit(text, (x+(i%n)*(w+.01*screen.get_height())+10,.05*screen.get_height()+(i//n)*(h+.01*screen.get_height())+10))
                            # else:
                            #     self.send_frame_to_jetson_btn.clickable = True
                            #     self.stop_send_frame_to_jetson_btn.clickable = False
                            #     self.select_video_csv_hint_lbl.text = ""
                        except Exception as e:
                            # print(e)
                            # traceback.print_exc()
                            pass

                        self.back_to_menu_btn.draw(screen)

                        if self.wait:
                            s = pygame.Surface((screen.get_width(),screen.get_height()))  # the size of your rect
                            s.set_alpha(220)                # alpha level
                            s.fill((50,50,50))           # this fills the entire surface
                            screen.blit(s, (0,0)) 
                            font = pygame.font.Font(None, 92)
                            text = font.render("Please Wait ...", True, (255,255,255))
                            r = text.get_rect()
                            screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                            self.__wait_for_process(process=self.__stop_sending_frame)

                    if self.wait_to_back:
                        s = pygame.Surface((screen.get_width(),screen.get_height()))  # the size of your rect
                        s.set_alpha(220)                # alpha level
                        s.fill((50,50,50))           # this fills the entire surface
                        screen.blit(s, (0,0)) 
                        font = pygame.font.Font(None, 92)
                        text = font.render("Please Wait ...", True, (255,255,255))
                        r = text.get_rect()
                        screen.blit(text, ((screen.get_width()-r.width)//2, (screen.get_height()-r.height)//2))
                        self.__wait_for_back()


            except Exception as e:
                traceback.print_exc()
                self.__go_menu()        

            ### update the screen and limit max framerate
            # real_screen.blit(pygame.transform.scale(screen, real_screen.get_rect().size), (0, 0))

            pygame.display.update()
            mainClock.tick(20 if self.step>40 else 60)
    

if __name__ == "__main__":
    import cv2
    import pygame
    from pygame.locals import *
    import os
    import glob
    import gc
    import sys
    import tkinter
    import tkinter.filedialog
    import torch
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
    from multiprocessing import Process, active_children, freeze_support, Value, Queue, Manager, Lock
    import queue
    import traceback
    from datetime import datetime, timedelta
    from utils.stream import Stream
    import json
    import pandas as pd
    from utils.frame_stream import OnlineStream, OfflineStream

    sys.path.append("./utils/MvImport/")
    from utils.CamOperation_class import *
    from MvCameraControl_class import *

    ## 2D modules
    from Modules_2D.pt_to_tensorrt import run as qt_run
    from Modules_2D.box_pose_identification_2d import run as run_2d
    from Modules_2D.train_identifier import run as train_run
    from Modules_2D.utils.plots import plot_skeleton_kpts

    ## 3D modules
    from Modules_3D.findCoordinateTransformationMatrix import *
    from Modules_3D.convert_2d_to_3d import *
    from Modules_3D.combine import combine, combine_all
    from Modules_3D.reconstruction import *

    ## camera
    from utils.computer_server import bind_server, jetson_handler, send_data, receive_data
    from utils.check_jetson import is_jetson_on, is_gui_running
    
    pygame.init()
    pygame.display.set_caption('ABT Software')
    screen = pygame.display.set_mode((0, 0), (pygame.RESIZABLE), vsync=1) # , pygame.FULLSCREEN
    # screen = real_screen.copy()
    a,b = screen.get_height(), screen.get_width()
    print(screen.get_height(), screen.get_width())
    mainClock = pygame.time.Clock()

    freeze_support()  # for multiprocessing on windows
    torch.multiprocessing.set_start_method('spawn')

    from utils.ui_utils import *
    app = App()
    app.run()
