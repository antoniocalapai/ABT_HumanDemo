import cv2
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from .utils.plots import plot_skeleton_kpts as plot_skeleton_kpts
from multiprocessing import Process, Queue, active_children, freeze_support, Value, Manager, Event
from .utils.iou import get_iou as get_iou
import os
from ultralytics import YOLO
import traceback
def read_video_file(video_file, start_second, end_second):
    source = video_file
    cap = cv2.VideoCapture(source)

    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
        sys.exit()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    print('video fps: ', fps_video)
    print('frame_width, frame_height: ', (frame_width, frame_height))

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_second == -1:
        start_frame = 0
    else:
        start_frame = min(max(0, fps_video * start_second), video_length - 2)
    if end_second == -1:
        end_frame = video_length
    else:
        end_frame = max(min(video_length, fps_video * end_second), start_frame + 1)

    return cap, frame_width, frame_height, fps_video, start_frame, end_frame

def append_to_tube(tube, val, tube_len):
    if len(tube) >= tube_len:
        tube.pop(0)
    return tube.append(val)

def similarity_to_tube(tube, val):
    mean = 0
    tubeNonelessLength = 0
    tubeWeight = [(count+1) if value is not None else 0 for count, value in enumerate(tube)]
    iouList = []

    if len(tube) > 0:
        for i in range(len(tube)):
            if tube[i] is not None:
                tubeNonelessLength += 1
                iouList.append(get_iou(tube[i], val))
            else:
                iouList.append(0)
        if tubeNonelessLength > 0:
            mean = sum([a*b for a, b in zip(iouList, tubeWeight)])/sum(tubeWeight)
    return mean


def draw_box_and_kpt(im0, c1, c2, x2, classlabel, classifier_conf, conf_thresh_kpt, box_conf, tube, colorlabel):
    # if classifier_conf < 0.5:
    #     classifier_conf = 1 - classifier_conf
    # classifier_conf = str(round(classifier_conf, 2))
    box_conf_str = str(round(box_conf, 2))

    Scale = round(max(1, min(im0.shape[0], im0.shape[1]) / 500))
    fontScale = round(max(1, min(c2[0]-c1[0], c2[1]-c1[1]) / 100))

    if tube is not None:
        for i in range(len(tube)):
            if tube[i] is not None:
                cv2.rectangle(im0, tube[i][0], tube[i][1], color=colorlabel, thickness=Scale, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(im0, c1, c2, color=colorlabel, thickness=Scale, lineType=cv2.LINE_AA)

    plot_skeleton_kpts(im0, x2, 3, [int(c1[0]), int(c1[1]), int(c2[0]), int(c2[1])], conf_thresh_kpt, fontScale)
    cv2.putText(im0, classlabel + ' ' + box_conf_str, c1, 0, fontScale, color=colorlabel, thickness=Scale, lineType=cv2.LINE_AA) #'id:' + classifier_conf +  +' '+ str(round(box_conf, 2)) +' ' classlabel + ' ' + classifier_conf

def save_txt_results(c1, c2, x2, classlabel, box_conf, frame_idx, name_info):
    # Write results to file
    with open(f"{name_info}" + '.txt', 'a') as f:
        f.write(('%s ' * (2) + '%.2f ' * (17 * 3 + 5) + '\n') % (frame_idx, classlabel, c1[0], c1[1], c2[0], c2[1], box_conf, x2[0], x2[1], x2[2], x2[3], x2[4], x2[5], x2[6]
                                                 , x2[7], x2[8], x2[9], x2[10], x2[11], x2[12], x2[13], x2[14], x2[15], x2[16], x2[17], x2[18], x2[19], x2[20]
                                                 , x2[21], x2[22], x2[23], x2[24], x2[25], x2[26], x2[27], x2[28], x2[29], x2[30], x2[31], x2[32], x2[33], x2[34]
                                                 , x2[35], x2[36], x2[37], x2[38], x2[39], x2[40], x2[41], x2[42], x2[43], x2[44], x2[45], x2[46], x2[47], x2[48], x2[49], x2[50]))

def imshow_imwrite_Process(outShowQueue, video_info, shared_show_, shared_write_):
    shared_show = shared_show_.value
    shared_write = shared_write_.value
    print('shared_show (imshow_imwrite process): ', shared_show)
    print('shared_write (imshow_imwrite process): ', shared_write)

    name_info = video_info.get()
    fps_info = video_info.get()
    size1_info = video_info.get()
    size2_info = video_info.get()

    if shared_write:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{name_info}.mp4",
                            fourcc, fps_info, (size1_info, size2_info)) # ,(cv2.VIDEOWRITER_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    frame_idx = 0

    while True:
        if outShowQueue.qsize() > 0:
            frame_idx += 1
            im0 = outShowQueue.get()
            if im0 is not None:
                cv2.putText(im0, 'frame: ' + str(frame_idx), (10, 30), 0, 1, [255, 0, 0], thickness=1, lineType=cv2.LINE_AA)

                if shared_show:
                    cv2.imshow('Video', im0)
                if shared_write:
                    out.write(im0)
                cv2.waitKey(1)
            else:
                break
        else:
            time.sleep(0.01)
    if shared_write:
        out.release()

def classifier_Process(outputQueue, imageQueue, video_info, classifer_info, shared_show_, shared_write_, shared_write_txt_results_, shared_processed_frame_, outShowQueue, running, ready):
    import heapq

    try:
        fps_info = video_info.get()
        name_info = video_info.get()
        classifier_path = classifer_info.get()
        kpt_thresh = classifer_info.get()
        tube_len = int(fps_info)

        shared_write_txt_results = shared_write_txt_results_.value
        shared_show = shared_show_.value
        shared_write = shared_write_.value
        print('shared_write_txt_results: ', shared_write_txt_results)
        print('shared_show (classifier process): ', shared_show)
        print('shared_write (classifier process): ', shared_write)

        framesQueue = []

        len_postprocess_frames = int(fps_info / 2)
        tube_len_seq = (len_postprocess_frames * 2) + 1

        # if (shared_show or shared_write):
        #     outShowQueue = Queue()
        #     process_imshow = Process(target=imshow_imwrite_Process, args=(outShowQueue, video_info, shared_show_, shared_write_))
        #     process_imshow.daemon = True
        #     process_imshow.start()

        classifier_ = YOLO(classifier_path, task='classify')
        imgsz_classifier = 224

        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        results = classifier_.predict(np.zeros((imgsz_classifier, imgsz_classifier, 3)), device=device, task='classify', imgsz=imgsz_classifier, verbose=False)
        for result in results:
            result = result.to('cpu')
            class_names = list(result.names.values())
            classes_size = len(class_names)
            print('classifier class names: ', class_names)
            print('classifier classes_size: ', classes_size)

        conf_thresh_kpt = kpt_thresh
        unknown_thresh = max(0.1, 0.2 - (classes_size / 100)) # 0.2
        unknown_thresh_2 = max(0.7, 0.8 - (classes_size / 100)) # 0.8
        similarity_box_thresh = 0.25
        similarity_box_thresh2 = 0.35 # 0.5
        similarity_box_thresh3 = 0.75

        outputList = []
        imageList = []
        lists_of_Tubes = [[] for _ in range(classes_size)] # tube per each monkey
        lists_of_Tubes_post = [[] for _ in range(classes_size)] # post processing tubes
        finished = 0

        np.random.seed(123)
        def generate_opencv_colors(num_colors):
            for i in range(num_colors):
                r = np.random.randint(0, 255)
                g = np.random.randint(0, 255)
                b = np.random.randint(0, 255)
                bgr_color = (r, g, b)
                yield tuple(map(int, bgr_color))

        # Create a generator for colors
        color_generator = generate_opencv_colors(classes_size)
        colorlabel = {name: next(color_generator) for name in class_names}
        classlabel = 'Unknown'
        frame_counter = 0
        frame_counter_filled = 0

        def post_process_box(Tube_seq, len_postprocess_frames):
            value_pred_ = None
            # previous frames
            prev = None
            for i3 in range(len_postprocess_frames):
                if Tube_seq[len_postprocess_frames - (i3 + 1)] == None:
                    continue
                else:
                    prev = Tube_seq[len_postprocess_frames - (i3 + 1)]
                    break
            # next frames
            next = None
            for i3 in range(len_postprocess_frames):
                if Tube_seq[len_postprocess_frames + (i3 + 1)] == None:
                    continue
                else:
                    next = Tube_seq[len_postprocess_frames + (i3 + 1)]
                    break
            if (prev is not None) and (next is not None):
                prev_next_iou = get_iou([prev[0], prev[1]], [next[0], next[1]])
                if prev_next_iou > 0.25:
                    c1_ = [int((g + h) / 2) for g, h in zip(prev[0], next[0])]
                    c1_ = tuple(x for x in c1_)
                    c2_ = [int((g + h) / 2) for g, h in zip(prev[1], next[1])]
                    c2_ = tuple(x for x in c2_)
                    x2_ = [(g + h) / 2 for g, h in zip(prev[2].tolist(), next[2].tolist())]
                    conf1 = (prev[3] + next[3]) / 2
                    conf2 = (prev[4] + next[4]) / 2
                    value_pred_ = [c1_, c2_, np.array(x2_), conf1, conf2]

            return value_pred_

        ready.value = True
        while True:
            ouputlistsize = len(outputList)
            if (ouputlistsize >= 1):
                cropimages = []

                im0 = imageList.pop(0)
                output = outputList.pop(0)
                originalim0 = im0.copy()

                if output.shape[0] > 0:
                    for idx in range(output.shape[0]):
                        x = output[idx, 0:4].T
                        c1 = (min(im0.shape[1], max(0, int(x[0] - x[2] / 2))), min(im0.shape[0], max(0, int(x[1] - x[3] / 2))))
                        c2 = (min(im0.shape[1], max(0, int(x[0] + x[2] / 2))), min(im0.shape[0], max(0, int(x[1] + x[3] / 2))))
                        cropimages.append(originalim0[int(c1[1]): int(c2[1]), int(c1[0]): int(c2[0])])

                predictions = []
                if len(cropimages) > 0:
                    t1_ = time.time()
                    for imgg in cropimages:
                        results = classifier_.predict(imgg, device=0, task='classify', imgsz=imgsz_classifier, verbose=False)
                        for result in results:
                            result = result.to('cpu')
                            predictions.append(result.probs.data.detach().numpy().tolist())
                    print('classifier time: ', time.time() - t1_)

                # check tube similarity
                appendedToTube = [0] * classes_size

                if output.shape[0] > 0:
                    for idx in range(output.shape[0]):
                        if 0 in appendedToTube:
                            confi = output[idx, 4] # box confidence
                            x = output[idx, 0:4].T
                            c1 = (min(im0.shape[1], max(0, int(x[0] - x[2] / 2))), min(im0.shape[0], max(0, int(x[1] - x[3] / 2)))) # box x1, y1
                            c2 = (min(im0.shape[1], max(0, int(x[0] + x[2] / 2))), min(im0.shape[0], max(0, int(x[1] + x[3] / 2))))  # box x2, y2
                            x2 = output[idx, 5:].T # keypoints
                            two_largest = heapq.nlargest(2, predictions[idx]) # for checking "If it is less than a certain limit, it is unknown"
                            max_ind = np.where(predictions[idx] == np.max(predictions[idx]))[0][0]

                            if np.abs(two_largest[0] - two_largest[1]) > unknown_thresh: # If it is less than a certain limit, it is unknown
                                sklet_confs = [x for x in range(len(x2)) if x % 3 == 2]
                                sklet_face_confs = [x for x in range(5 * 3) if x % 3 == 2]
                                count_face_sklet = len([i for i in x2[sklet_face_confs] if i > conf_thresh_kpt])
                                count_sklet = len([i for i in x2[sklet_confs] if i > conf_thresh_kpt])
                                lists_of_sims = []
                                for monk in range(classes_size):
                                    lists_of_sims.append(similarity_to_tube(lists_of_Tubes[monk], [c1, c2]))
                                any_added = 0

                                if count_sklet >= 8 or (count_sklet >= 7 and count_face_sklet >= 3):
                                    if appendedToTube[max_ind] == 0:
                                        if sum(x is not None for x in lists_of_Tubes[max_ind]) < 1: # max_ind of related Tube is empty
                                            # checking other Tubes have less than similarity_box_thresh2 intersection?
                                            other_Tube_sim = 0 # is there any Tube with more than similarity_box_thresh2 similarity?
                                            for index, other_Tube in enumerate(lists_of_Tubes):
                                                if index != max_ind:
                                                    if sum(x is not None for x in other_Tube) > 0: # other Tubes is not empty
                                                        if lists_of_sims[index] > similarity_box_thresh2:
                                                            other_Tube_sim = 1
                                                            break
                                            if other_Tube_sim == 0: # other Tubes have less than similarity_box_thresh2 similarity
                                                if np.abs(two_largest[0] - two_largest[1]) > unknown_thresh_2: # how much classifier is sure about max_ind?
                                                    classlabel = class_names[max_ind]
                                                    append_to_tube(lists_of_Tubes[max_ind], [c1, c2], tube_len)
                                                    append_to_tube(lists_of_Tubes_post[max_ind], [c1, c2, x2, predictions[idx][max_ind], confi], tube_len_seq)  # for post processing
                                                    appendedToTube[max_ind] = 1
                                                    any_added = 1
                                        else: # max_ind related Tube is not empty
                                            if lists_of_sims[max_ind] > similarity_box_thresh:
                                                other_sims_are_not_low = 0
                                                for index, sim_ in enumerate(lists_of_sims):
                                                    if index != max_ind:
                                                        if lists_of_sims[max_ind] < lists_of_sims[index]:
                                                            other_sims_are_not_low = 1
                                                if other_sims_are_not_low == 0:
                                                    classlabel = class_names[max_ind]
                                                    append_to_tube(lists_of_Tubes[max_ind], [c1, c2], tube_len)
                                                    append_to_tube(lists_of_Tubes_post[max_ind], [c1, c2, x2, predictions[idx][max_ind], confi], tube_len_seq)  # for post processing
                                                    appendedToTube[max_ind] = 1
                                                    any_added = 1
                                                else:
                                                    print('---------- other_sims_are_not_low == 0 -----------')
                                            else:
                                                print('---------- lists_of_sims[max_ind] > similarity_box_thresh -----------')

                                    if any_added == 0: # classifier predict wrongly but similarity with other classes is high
                                        if np.abs(two_largest[0] - two_largest[1]) < unknown_thresh_2:
                                            for index, other_Tube in enumerate(lists_of_Tubes):
                                                if index != max_ind:
                                                    if appendedToTube[index] == 0:
                                                        if sum(x is not None for x in lists_of_Tubes[index]) > 0:
                                                            if lists_of_sims[index] > similarity_box_thresh3:
                                                                classlabel = class_names[index]
                                                                append_to_tube(lists_of_Tubes[index], [c1, c2], tube_len)
                                                                append_to_tube(lists_of_Tubes_post[index], [c1, c2, x2, predictions[idx][index], confi], tube_len_seq)  # for post processing
                                                                appendedToTube[index] = 1
                                                                any_added = 1
                                                                print('---------- Classifier is wrong but similarity with other classes is high ----------')
                                                                break
                                else:
                                    print('-------- skeleton count less than 8 -------- : ', count_sklet)
                            else:
                                print('------ less than unknown_thresh --------')
                        else:
                            print('-------- more than number of monkeys classes identification happened --------')

                for index, other_Tube in enumerate(lists_of_Tubes):
                    if appendedToTube[index] == 0:
                        append_to_tube(lists_of_Tubes[index], None, tube_len)
                        append_to_tube(lists_of_Tubes_post[index], None, tube_len_seq)

                framesQueue.append(im0)

                if len(lists_of_Tubes_post[0]) <= len_postprocess_frames: #  and finished == 0
                    frame_counter += 1
                    im0 = framesQueue.pop(0)
                    for index, other_Tube in enumerate(lists_of_Tubes_post):
                        value_pred = other_Tube[-1]
                        if value_pred is not None:
                            classlabel = class_names[index]
                            if (shared_show or shared_write):
                                draw_box_and_kpt(im0, value_pred[0], value_pred[1], value_pred[2], classlabel, value_pred[3], conf_thresh_kpt, value_pred[4], None, colorlabel[classlabel])  # lists_of_Tubes_post[index] None
                            if shared_write_txt_results:
                                save_txt_results(value_pred[0], value_pred[1], value_pred[2], classlabel, value_pred[3], frame_counter, name_info)
                    if (shared_show or shared_write):
                        outShowQueue.put(im0)
                    else:
                        shared_processed_frame_.value+=1
                elif len(lists_of_Tubes_post[0]) == tube_len_seq:
                    frame_counter += 1
                    im0 = framesQueue.pop(0)
                    for index, other_Tube in enumerate(lists_of_Tubes_post):
                        value_pred = other_Tube[len_postprocess_frames]
                        if value_pred is None:
                            value_pred = post_process_box(other_Tube, len_postprocess_frames)
                            lists_of_Tubes_post[index][len_postprocess_frames] = value_pred
                            if value_pred is not None:
                                frame_counter_filled += 1
                                print('in: ', frame_counter, ' num filled frames: ', frame_counter_filled)
                        if value_pred is not None:
                            classlabel = class_names[index]
                            if (shared_show or shared_write):
                                draw_box_and_kpt(im0, value_pred[0], value_pred[1], value_pred[2], classlabel, value_pred[3], conf_thresh_kpt, value_pred[4], None, colorlabel[classlabel])  # lists_of_Tubes_post[index] None
                            if shared_write_txt_results:
                                save_txt_results(value_pred[0], value_pred[1], value_pred[2], classlabel, value_pred[3], frame_counter, name_info)
                    if (shared_show or shared_write):
                        outShowQueue.put(im0)
                    else:
                        shared_processed_frame_.value+=1
                print('frame_counter: ', frame_counter)
            elif finished == 1:
                for tube_i in range(len(framesQueue)):
                    frame_counter += 1
                    im0 = framesQueue.pop(0)
                    for index, other_Tube in enumerate(lists_of_Tubes_post):
                        value_pred = other_Tube[len_postprocess_frames + (tube_i + 1)]
                        if value_pred is not None:
                            classlabel = class_names[index]
                            if (shared_show or shared_write):
                                draw_box_and_kpt(im0, value_pred[0], value_pred[1], value_pred[2], classlabel, value_pred[3], conf_thresh_kpt, value_pred[4], None, colorlabel[classlabel])  # lists_of_Tubes_post[index] None
                            if shared_write_txt_results:
                                save_txt_results(value_pred[0], value_pred[1], value_pred[2], classlabel, value_pred[3], frame_counter, name_info)
                    if (shared_show or shared_write):
                        outShowQueue.put(im0)
                    else:
                        shared_processed_frame_.value+=1
                    print('frame_counter: ', frame_counter)
                if (shared_show or shared_write):
                    outShowQueue.put(None)
                break
            else:
                if outputQueue.qsize() > 0 and imageQueue.qsize() > 0:
                    outputList.append(outputQueue.get())
                    imageList.append(imageQueue.get())
                elif outputQueue.qsize() == 1 and imageQueue.qsize() == 0:
                    oq = outputQueue.get()
                    if oq is None: # end process
                        finished = 1
                    else:
                        outputQueue.put(oq)
                else:
                    time.sleep(0.01)
    except Exception as e:
        print(e)
        outShowQueue.put('ERROR')
        running.value = False

    return

@torch.no_grad()
def run(boxpose_weight,
        identification,
        identifier_weight,
        num_monkeys,
        input_queue,
        frame_width, frame_height,fps_video,
        output_file,
        show_video,
        write_video,
        write_txt_results,
        bbox_conf,
        iou_conf,
        kpt_conf,
        processed_frame_num,
        outShowQueue,
        is_running,
        is_waiting, 
        is_ready):

    try:

        # stream = Stream()
        # stream = stream.from_camera(video_file)
        # # (frame_width, frame_height) = stream.get_frame_size()
        # fps_video = stream.get_fps()
        # (start_frame, end_frame) = stream.get_length()
        # cap, frame_width, frame_height, fps_video, start_frame, end_frame = read_video_file(video_file, start_second, end_second)
        boxpose_path = boxpose_weight

        video_info = Queue()
        shared_show = Value("b", show_video)  # "b" indicates a boolean, "i" int
        shared_write = Value("b", write_video)  # "b" indicates a boolean, "i" int

        conf_thresh_kpt = kpt_conf
        if identification:
            shared_write_txt_results = Value("b", write_txt_results)  # "b" indicates a boolean, "i" int
            outputQueue = Queue()
            imageQueue = Queue()
            classifer_info = Queue()
            classifer_info.put(identifier_weight)
            classifer_info.put(conf_thresh_kpt)
            process = Process(target=classifier_Process, args=(outputQueue, imageQueue, video_info, classifer_info, shared_show, shared_write, shared_write_txt_results, processed_frame_num, outShowQueue, is_running, is_ready))
        # else:
        #     if show_video or write_video:
        #         outShowQueue = Queue()
        #         process_imshow = Process(target=imshow_imwrite_Process, args=(outShowQueue, video_info, shared_show, shared_write))
        #         process_imshow.daemon = True

        #select device
        print('torch.cuda.is_available: ', torch.cuda.is_available())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        print('device: ', device.type)
        # # Load model
        model = YOLO(boxpose_path, task='pose') # monkey_yolov8m_31.pt monkey_yolov8m.engine monkey_yolov8s_42_1280.pt monkey_yolov8m_70_1280.engine monkey_yolov8s_42_1280.engine
        imgsz = 1280  # input box pose img_size
        # bbox conf
        conf_thresh = bbox_conf # 0.2 default, 0.01 for top view
        iou_thresh = iou_conf # 0.75
        half = (device.type!='cpu') # True

        if device.type != 'cpu':
            model.predict(np.zeros((imgsz, imgsz, 3)), imgsz=imgsz, conf=conf_thresh, iou=iou_thresh, half=half, device=device, stream=True, save=False, show=False, save_txt=False, verbose=False)  # , stream=True # torch.zeros(1, 3, imgsz, imgsz)

        if identification:
            video_info.put(fps_video)
            video_info.put(output_file)
        video_info.put(output_file)
        video_info.put(fps_video)
        video_info.put(frame_width)
        video_info.put(frame_height)
        if identification:
            process.start()
        # elif (show_video or write_video):
        #     process_imshow.start()

        if write_txt_results:
            if os.path.exists(f"{output_file}" + '.txt'):
                os.remove(f"{output_file}" + '.txt')
            with open(f"{output_file}" + '.txt', 'a') as f:
                f.write(('%s' + '\n') % ('frame_number, monkey_ID, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_confidence, nose_x, nose_y, nose_confidence, left_eye_x, left_eye_y, left_eye_confidence, right_eye_x, right_eye_y, right_eye_confidence, left_ear_x, left_ear_y, left_ear_confidence, right_ear_x, right_ear_y, right_ear_confidence, left_shoulder_x, left_shoulder_y, left_shoulder_confidence, right_shoulder_x, right_shoulder_y, right_shoulder_confidence, left_elbow_x, left_elbow_y, left_elbow_confidence, right_elbow_x, right_elbow_y, right_elbow_confidence, left_wrist_x, left_wrist_y, left_wrist_confidence, right_wrist_x, right_wrist_y, right_wrist_confidence, left_hip_x, left_hip_y, left_hip_confidence, right_hip_x, right_hip_y, right_hip_confidence, left_knee_x, left_knee_y, left_knee_confidence, right_knee_x, right_knee_y, right_knee_confidence, left_ankle_x, left_ankle_y, left_ankle_confidence, right_ankle_x, right_ankle_y, right_ankle_confidence'))

        #count num of frames
        frame_count = 0

        start_runtime = time.time()
        prev_timez = time.time()

        # wait for identification model to load and ready to predict
        if identification:
            while not is_ready.value:
                time.sleep(0.01)
        else:
            is_ready.value = True

        
        start_runtime = time.time()
        while is_running.value:
            try:
                frame = input_queue.get()
                if isinstance(frame, str) and frame=='WAIT':
                    print("waiting for frame")
                    is_waiting.value = True
                    continue
                is_waiting.value = False
                if frame is None:
                    break
                frame_count += 1
                
                if frame_count % fps_video == 0:
                    print("\nTime Processing {}".format(time.time() - prev_timez))
                    print("Frame {} Processing".format(frame_count))
                    prev_timez = time.time()

                results = model.predict(frame, imgsz=imgsz, conf=conf_thresh, iou=iou_thresh, half=half, device=device, save=False, show=False, save_txt=False, verbose=False, stream=True)

                output = [] # each detection output len:  56 = 5 + 3*17
                for result in results:
                    result = result.to('cpu')
                    num_boxes = result.boxes.data.shape[0]
                    boxes = result.boxes
                    keypoints = result.keypoints.data.detach().numpy()  # x y conf
                    # print('boxes.data: ', boxes.data.detach().numpy())
                    # print('result.keypoints: ', keypoints)
                    for ibox in range(num_boxes):
                        internal_output = []
                        if ibox >= num_monkeys:
                            break
                        box = boxes[ibox]  # returns one box
                        box_xywh = box.xywh[0].detach().numpy()  # box with xywh format, (N, 4)
                        box_xyxy = box.xyxy[0].detach().numpy()  # box with xyxy format, (N, 4)
                        # print((box_xyxy[0], box_xyxy[1]))
                        box_conf = box.conf.detach().numpy()  # confidence score, (N, 1)
                        if identification:
                            internal_output.extend(box_xywh)
                            internal_output.extend(box_conf)
                        for ikey in range(len(keypoints[ibox])):
                            internal_output.extend(keypoints[ibox][ikey])
                        if identification:
                            output.append(internal_output)
                        else:
                            classlabel = 'monkey'
                        if (not identification) and (show_video or write_video):
                            draw_box_and_kpt(frame, (int(box_xyxy[0]), int(box_xyxy[1])), (int(box_xyxy[2]), int(box_xyxy[3])), internal_output, classlabel, 0.00,
                                            conf_thresh_kpt, box_conf[0], None, (0,255,0))  # Tube2 None
                        if (not identification) and write_txt_results:
                            save_txt_results((int(box_xyxy[0]), int(box_xyxy[1])), (int(box_xyxy[2]), int(box_xyxy[3])), internal_output, classlabel, box_conf[0], frame_count, output_file)

                if not identification:
                    if (show_video or write_video):
                        outShowQueue.put(frame)
                    else:
                        processed_frame_num.value+=1
                else:
                    outputQueue.put(np.array(output))
                    imageQueue.put(frame)

            except Exception as e:
                print(e)
                traceback.print_exc()
                break

        
        if not identification and (show_video or write_video):
            outShowQueue.put(None)
        elif identification:
            outputQueue.put(None)

        print("out of while")
        seconds_runtime = time.time() - start_runtime

        print('seconds_runtime without classifier: ', seconds_runtime)
        print('FPS without classifier: ', str(frame_count / seconds_runtime))

        if identification:
            process.join()
        # elif (show_video or write_video):
        #     process_imshow.join()

        active = active_children()
        print('main process, active_children: ', len(active))

        seconds_runtime = time.time() - start_runtime
        print('seconds_runtime with classifier, write, show: ', seconds_runtime)
        print('FPS with classifier, write, show: ', str(frame_count / seconds_runtime))
        print('---------- Process completed! ----------')
    except Exception as e:
        print("an error occured during prediction!", e)
        traceback.print_exc()
        print(outShowQueue.qsize())
        if (show_video or write_video):
            outShowQueue.put('ERROR')
        if identification:
            ch = active_children()
            for c in ch:
                c.kill()

    is_running.value = False

if __name__ == "__main__":
    freeze_support()  # for multiprocessing on windows
    torch.multiprocessing.set_start_method('spawn')

    # check cuda
    device_cuda = torch.cuda.is_available()
    if not device_cuda:
        print('Cuda is not available!')
        sys.exit()

    # ---------------- Parameters ----------------
    boxpose_weight = './models/box_pose_detector/monkey_box_pose.engine' # pt or engine
    identification = True # do you need to do identification?
    if identification:
        identifier_weight = './models/identifier/identifier_nathan_vin.engine' # pt or engine
    else:
        identifier_weight = ''
    num_monkeys = 2 # number of monkeys
    video_file = './input_videos/Camera_2.avi' # input path of video file
    output_path = './output_files/'  # output path
    start_second = -1 # starting second for process video, -1: start from beginning of file
    end_second = -1 # ending second for process video, -1: processing to the end of file
    show_video = True # show video result?
    write_video = True # write video results?
    write_txt_results = True # save text file results?
    bbox_conf = 0.2 # box_pose bounding_box_threshold, default value should be 0.2
    iou_conf  = 0.75 # box_pose iou_threshold, default value should be 0.75
    kpt_conf = 0.4 # box_pose pose_threshold, default value should be 0.4

    run(boxpose_weight, identification, identifier_weight, num_monkeys, video_file, output_path, start_second, end_second, show_video, write_video, write_txt_results, bbox_conf, iou_conf, kpt_conf)
