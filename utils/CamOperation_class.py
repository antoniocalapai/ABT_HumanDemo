# -- coding: utf-8 --
import sys
import threading
# import msvcrt
import _tkinter
import tkinter.messagebox
from tkinter import * 
from tkinter.messagebox import *
import tkinter as tk
import numpy as np
import cv2
import time
import sys, os
import datetime
import inspect
import ctypes
import random
from PIL import Image,ImageTk
from ctypes import *
from tkinter import ttk
from datetime import datetime
from utils import CvFpsCalc

sys.path.append("./MvImport")
from MvCameraControl_class import *


def Async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)

class CameraOperation():

    def __init__(self,obj_cam,st_device_list,n_connect_num=0,b_open_device=False,b_start_grabbing = False,h_thread_handle=None,\
                b_thread_closed=False,st_frame_info=None,b_exit=False,b_save_bmp=False,b_save_jpg=False,buf_save_image=None,\
                n_save_image_size=0,frame_rate=7,exposure_time=20000,gain=0,binning=2):

        self.obj_cam = obj_cam
        self.st_device_info = st_device_list
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing 
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.n_save_image_size = n_save_image_size
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain
        self.nPayloadSize = 0
        self.data_buf = 0
        self.binning = binning
        self.ActionDeviceKey = 1
        self.ActionGroupKey = 1
        self.ActionGroupMask = 1
        self.not_responding = False
        
    def To_hex_str(self,num):
        chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
        hexStr = ""
        if num < 0:
            num = num + 2**32
        while num >= 16:
            digit = num % 16
            hexStr = chaDic.get(digit, str(digit)) + hexStr
            num //= 16
        hexStr = chaDic.get(num, str(num)) + hexStr   
        return hexStr

    def Open_device(self, trigger_type_var):
        if self.b_open_device is False:
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_info, POINTER(MV_CC_DEVICE_INFO)).contents # self.st_device_info.pDeviceInfo[int(nConnectionNum)]
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                self.b_open_device = False
                self.b_thread_closed = False
                return ret
            self.b_open_device = True
            self.b_thread_closed = False

            # Detection network optimal package size (It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: packet size is invalid[%d]" % nPacketSize)

            ret = self.obj_cam.MV_CC_SetBoolValue("ReverseX", c_bool(False)) #False True
            if ret != 0:
                print("warning: set ReverseX off fail! ret[0x%x]" % ret)

            ret = self.obj_cam.MV_CC_SetBoolValue("ReverseY", c_bool(False))  # False True
            if ret != 0:
                print("warning: set ReverseY off fail! ret[0x%x]" % ret)

            ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode", "On") #"Off" "On"
            if ret != 0:
                print("warning: set trigger mode off fail! ret[0x%x]" % ret)

            if trigger_type_var == 1:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerSource", "Action1")
                if ret != 0:
                    print("warning: set TriggerSource Action1 fail! ret[0x%x]" % ret)
                ret = self.obj_cam.MV_CC_SetIntValue("ActionDeviceKey", self.ActionDeviceKey)
                if ret != 0:
                    print('show error: ', 'set ActionDeviceKey fail!')
                ret = self.obj_cam.MV_CC_SetIntValue("ActionGroupKey", self.ActionGroupKey)
                if ret != 0:
                    print('show error: ', 'set ActionGroupKey fail!')
                ret = self.obj_cam.MV_CC_SetIntValue("ActionGroupMask", self.ActionGroupMask)
                if ret != 0:
                    print('show error: ', 'set ActionGroupMask fail!')
                ret = self.obj_cam.MV_CC_SetBoolValue("GevIEEE1588", c_bool(True))
                if ret != 0:
                    print('show error: ', 'set GevIEEE1588 fail!')
            else:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerSource", "Software")
                if ret != 0:
                    print("warning: set TriggerSource Software fail! ret[0x%x]" % ret)

            # -------------
            # Set exposure, frame rate, as needed
            ret = self.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 2)
            if ret != 0:
                print('show error: ', 'set ExposureAuto fail!')
            ret = self.obj_cam.MV_CC_SetIntValue("AutoExposureTimeUpperLimit", 100000)
            if ret != 0:
                print('show error: ', 'set AutoExposureTimeUpperLimit fail!')
            ret = self.obj_cam.MV_CC_SetIntValue("AutoExposureTimeLowerLimit", 15)
            if ret != 0:
                print('show error: ', 'set AutoExposureTimeLowerLimit fail!')
            ret = self.obj_cam.MV_CC_SetEnumValue("GainAuto", 2)
            if ret != 0:
                print('show error: ', 'set GainAuto fail!')
            time.sleep(0.1)
            # ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(self.exposure_time))
            # if ret != 0:
            #     print('show error: ', 'set exposure time fail!')
            # ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(self.gain))
            # if ret != 0:
            #     print('show error: ', 'set Gain fail!')

            ret = self.obj_cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", c_bool(True))
            if ret != 0:
                print("set AcquisitionFrameRateEnable to True failed! Error code: 0x%x" % ret)
            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.frame_rate))
            if ret != 0:
                print('show error: ', 'set AcquisitionFrameRate fail!')

            ret = self.obj_cam.MV_CC_SetEnumValue("BinningHorizontal", self.binning)
            if ret != 0:
                print('show error: ', 'set BinningHorizontal fail!')
            ret = self.obj_cam.MV_CC_SetEnumValue("BinningVertical", self.binning)
            if ret != 0:
                print('show error: ', 'set BinningVertical fail!')
            ret = self.obj_cam.MV_CC_SetEnumValue("PixelFormat", 0x02100032)
            if ret != 0:
                print('show error: ', 'set PixelFormat fail!')

            default_BinningHorizontal = MVCC_ENUMVALUE()
            memset(byref(default_BinningHorizontal), 0, sizeof(MVCC_ENUMVALUE))
            ret = self.obj_cam.MV_CC_GetEnumValue("BinningHorizontal", default_BinningHorizontal)
            if ret != 0:
                print("get BinningHorizontal fail! ret[0x%x]" % ret)
            else:
                print(f"Default BinningHorizontal: {default_BinningHorizontal.nCurValue}")

            default_BinningVertical = MVCC_ENUMVALUE()
            memset(byref(default_BinningVertical), 0, sizeof(MVCC_ENUMVALUE))
            ret = self.obj_cam.MV_CC_GetEnumValue("BinningVertical", default_BinningVertical)
            if ret != 0:
                print("get BinningVertical fail! ret[0x%x]" % ret)
            else:
                print(f"Default BinningVertical: {default_BinningVertical.nCurValue}")

            default_PixelFormat = MVCC_ENUMVALUE()
            memset(byref(default_PixelFormat), 0, sizeof(MVCC_ENUMVALUE))
            ret = self.obj_cam.MV_CC_GetEnumValue("PixelFormat", default_PixelFormat)
            if ret != 0:
                print("get PixelFormat fail! ret[0x%x]" % ret)
            else:
                print(f"Default PixelFormat: {default_PixelFormat.nCurValue}")  # 34603058 == YUV422_YUYV_Packed
            # -------------

            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
            ret = self.obj_cam.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                print("get payload size fail! ret[0x%x]" % ret)
                return ret
            self.nPayloadSize = stParam.nCurValue
            self.data_buf = (c_ubyte * self.nPayloadSize)()

            return 0

    def Set_ReverseX(self, value):
        if True == self.b_open_device:
            if value == 0:
                ret = self.obj_cam.MV_CC_SetBoolValue("ReverseX", c_bool(False))  # False True
            else:
                ret = self.obj_cam.MV_CC_SetBoolValue("ReverseX", c_bool(True))  # False True
            if ret != 0:
                print("warning: set ReverseX fail! ret[0x%x]" % ret)
            return ret

    def Set_ReverseY(self, value):
        if True == self.b_open_device:
            if value == 0:
                ret = self.obj_cam.MV_CC_SetBoolValue("ReverseY", c_bool(False))  # False True
            else:
                ret = self.obj_cam.MV_CC_SetBoolValue("ReverseY", c_bool(True))  # False True
            if ret != 0:
                print("warning: set ReverseY fail! ret[0x%x]" % ret)
            return ret

    def Start_grabbing(self, index, frame_queue):
        if False == self.b_start_grabbing and True == self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            # if ret != 0:
            #     self.b_start_grabbing = False
            #     return ret
            self.b_start_grabbing = True
            try:
                self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread, args=(self, index, frame_queue))
                self.h_thread_handle.start()
                self.b_thread_closed = True
            except:
                print('show error',' error: unable to start thread Work_thread')
                self.b_start_grabbing = False
            return ret

    def Stop_grabbing(self):
        if True == self.b_start_grabbing and self.b_open_device == True:
            if self.b_thread_closed:
                self.b_exit = True
                # Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_StopGrabbing()
            # if ret != 0:
            #     self.b_start_grabbing = True
            #     return ret
            self.b_start_grabbing = False
            return 0

    def Close_device(self, trigger_type_var):
        if self.b_open_device:

            if self.b_thread_closed:
                self.b_exit = True
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if trigger_type_var == 1:
                ret = self.obj_cam.MV_CC_SetBoolValue("GevIEEE1588", c_bool(False))
                if ret != 0:
                    print('show error: ', 'set GevIEEE1588 fail!')
            ret = self.obj_cam.MV_CC_CloseDevice()
            return ret

        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False

    def Trigger_once(self):
        if True == self.b_open_device:
            ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
            return ret

    def Get_parameter(self, trigger_type_var, auto_exposure, auto_gain):
        if True == self.b_open_device:
            stFloatParam_FrameRate =  MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            if not auto_exposure:
                stFloatParam_exposureTime = MVCC_FLOATVALUE()
                memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            if not auto_gain:
                stFloatParam_gain = MVCC_FLOATVALUE()
                memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            stEnumParam_BinningHorizontal = MVCC_ENUMVALUE()
            memset(byref(stEnumParam_BinningHorizontal), 0, sizeof(MVCC_ENUMVALUE))
            if trigger_type_var == 1:
                stIntParam_ActionDeviceKey =  MVCC_INTVALUE()
                memset(byref(stIntParam_ActionDeviceKey), 0, sizeof(MVCC_INTVALUE))
                stIntParam_ActionGroupKey =  MVCC_INTVALUE()
                memset(byref(stIntParam_ActionGroupKey), 0, sizeof(MVCC_INTVALUE))
                stIntParam_ActionGroupMask =  MVCC_INTVALUE()
                memset(byref(stIntParam_ActionGroupMask), 0, sizeof(MVCC_INTVALUE))

            # ret = self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            ret = self.obj_cam.MV_CC_GetFloatValue("ResultingFrameRate", stFloatParam_FrameRate)
            self.frame_rate = stFloatParam_FrameRate.fCurValue
            if not auto_exposure:
                ret = self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
                self.exposure_time = stFloatParam_exposureTime.fCurValue
            if not auto_gain:
                ret = self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
                self.gain = stFloatParam_gain.fCurValue
            ret = self.obj_cam.MV_CC_GetEnumValue("BinningHorizontal", stEnumParam_BinningHorizontal)
            self.binning = stEnumParam_BinningHorizontal.nCurValue
            if trigger_type_var == 1:
                ret = self.obj_cam.MV_CC_GetIntValue("ActionDeviceKey", stIntParam_ActionDeviceKey)
                self.ActionDeviceKey = stIntParam_ActionDeviceKey.nCurValue
                ret = self.obj_cam.MV_CC_GetIntValue("ActionGroupKey", stIntParam_ActionGroupKey)
                self.ActionGroupKey = stIntParam_ActionGroupKey.nCurValue
                ret = self.obj_cam.MV_CC_GetIntValue("ActionGroupMask", stIntParam_ActionGroupMask)
                self.ActionGroupMask = stIntParam_ActionGroupMask.nCurValue

            return ret

    def Set_parameter(self,frameRate,exposureTime,gain,binning,DeviceKey,GroupKey,GroupMask, trigger_type_var, auto_exposure, auto_gain):
        if '' == frameRate or ('' == exposureTime and not auto_exposure) or ('' == gain and not auto_gain):
            return -1
        if True == self.b_open_device:
            if auto_exposure:
                ret = self.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 2)
                ret = self.obj_cam.MV_CC_SetIntValue("AutoExposureTimeUpperLimit", 100000)
                ret = self.obj_cam.MV_CC_SetIntValue("AutoExposureTimeLowerLimit", 15)
            else:
                ret = self.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 0)
                time.sleep(0.1)
                ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime",float(exposureTime))
            if auto_gain:
                ret = self.obj_cam.MV_CC_SetEnumValue("GainAuto", 2)
            else:
                ret = self.obj_cam.MV_CC_SetEnumValue("GainAuto", 0)
                time.sleep(0.1)
                ret = self.obj_cam.MV_CC_SetFloatValue("Gain",float(gain))

            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate",float(frameRate))
            ret = self.obj_cam.MV_CC_SetEnumValue("BinningHorizontal", binning)
            ret = self.obj_cam.MV_CC_SetEnumValue("BinningVertical", binning)
            if trigger_type_var == 1:
                ret = self.obj_cam.MV_CC_SetIntValue("ActionDeviceKey", int(float(DeviceKey)))
                ret = self.obj_cam.MV_CC_SetIntValue("ActionGroupKey", int(float(GroupKey)))
                ret = self.obj_cam.MV_CC_SetIntValue("ActionGroupMask", int(float(GroupMask)))

            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
            ret = self.obj_cam.MV_CC_GetIntValue("PayloadSize", stParam)
            self.nPayloadSize = stParam.nCurValue
            self.data_buf = (c_ubyte * self.nPayloadSize)()
            return ret

    def convert_yuv422_to_rgb(self, stOutFrame):
        nWidth = stOutFrame.stFrameInfo.nWidth
        nHeight = stOutFrame.stFrameInfo.nHeight

        # Convert YUV422 to RGB using OpenCV
        pData = stOutFrame.pBufAddr
        frame_data = cast(pData, POINTER(c_ubyte * stOutFrame.stFrameInfo.nFrameLen)).contents

        # Reshape YUYV data into Height x Width x 2
        yuyv_image = np.array(frame_data, dtype=np.uint8)
        yuyv_image = yuyv_image.reshape((nHeight, nWidth, 2))

        # Separate Y and UV channels
        y_channel = yuyv_image[:, :, 0]
        uv_channel = yuyv_image[:, :, 1]

        # Convert YUV to BGR (OpenCV uses BGR)
        numArray = cv2.cvtColor(cv2.merge([y_channel, uv_channel]), cv2.COLOR_YUV2RGB_YUYV)

        return numArray

    def Work_thread(self,index, frame_queue):
        pData = byref(self.data_buf)
        nDataSize = self.nPayloadSize

        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        num_not_responding = 0
        fps_calc1 = CvFpsCalc(buffer_len=10)
        while True:
            numArray = None
            if self.b_exit:
                frame_queue.put(None)
                print('now break')
                break
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 2000)
            if 0 == ret and None != stOutFrame.pBufAddr:
                self.not_responding = False
                num_not_responding = 0
                host_timestamp = stOutFrame.stFrameInfo.nHostTimeStamp
                timestamp_datetime = datetime.utcfromtimestamp(host_timestamp / 1000.0)
                formatted_time = timestamp_datetime.strftime('%H:%M:%S.%f')[:-3]
                print("Camera[%d]: camera clock time: " % (index), formatted_time)
                # dt = datetime.now()
                # print("Camera[%d]: time: " % (index), dt)
                # print("Camera[%d]:get one frame: Width[%d], Height[%d], PixelType[0x%x], nFrameNum[%d]" % (index, stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.enPixelType, stFrameInfo.nFrameNum))
                print("Camera[%d]:get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (index, stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
                if stOutFrame.stFrameInfo.enPixelType == PixelType_Gvsp_YUV422_YUYV_Packed:
                    numArray = self.convert_yuv422_to_rgb(stOutFrame)
                    # Get FPS for each video
                    fps1 = fps_calc1.get()
                    print("Camera[" + str(index) + "]: fps: ", fps1)
                    frame_queue.put(numArray)  # Put frames in the queue for the corresponding camera

                self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                print("Camera[" + str(index) + "]:no data, ret = "+self.To_hex_str(ret))
                if num_not_responding > 0:
                    self.not_responding = True
                num_not_responding += 1
                continue
