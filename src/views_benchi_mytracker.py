#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from flask import Flask, render_template, Blueprint,Response
from src.person_detect_used import preson_detect
from src.my_functions import *
#from concurrent.futures import ThreadPoolExecutor
import threading
import cv2
#import requests
import time, os, json, glob
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import warnings


from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from keras import backend
import tensorflow as tf
import requests
from collections import deque

url = "http://fabx.aidong-ai.com/snapshot/worker"

#====1mysetting======
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
warnings.filterwarnings('ignore')
# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
app = Flask(__name__, static_folder='../static', static_url_path='/static')
cors = CORS()
cors.init_app(app)
main = Blueprint('main', __name__)
app.register_blueprint(main,url_prefix = "")

camer_list = []
camer_count = 0
q_put_img = deque(maxlen=1)



# #是否显示中文标注
while True:
    a_in = input("Please input a number(0: simple, 1: complex)--->")
    if a_in=="0":
        control_color = False
        break
    elif a_in=="1":
        control_color = True
        break
    else:
        print("=================input error!====================")
        continue

@app.route('/')
def index():
    global camer_count
    #camer_count = getcountCameras() 
    camer_count = "1"  #修改让只出现一个窗口
    return render_template('index.html')  #====2mysetting======

def clearCapture(capture): 
    capture.release() 
    cv2.destroyAllWindows() 
  
def getcountCameras(): 
    global camer_list
    n = 0 

    for i in range(10): 
     try: 
      cap = cv2.VideoCapture(i) 
      ret, frame = cap.read() 

      if ret:
        camer_list.append(i)
        n += 1 
     except: 
      clearCapture(cap) 
      break 
    return str(n)

@app.route('/countCameras')
def countCameras(): 
    global camer_count
    return camer_count

@app.route('/detect_video')
def detect_video():
    global camer_list
    # get_detect()
    number = 0
    if len(camer_list)>=1:
        number = camer_list[0]
    return Response(get_detect(number),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_video2')
def detect_video2():
    global camer_list
    # get_detect()
    number = 0
    if len(camer_list)>=2:
        number = camer_list[1]
    return Response(get_detect(number),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_video3')
def detect_video3():
    global camer_list
    # get_detect()
    number = 0
    if len(camer_list)>=3:
        number = camer_list[2]
    return Response(get_detect(number),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_video4')
def detect_video4():
    global camer_list
    # get_detect()
    number = 0
    if len(camer_list)>=4:
        number = camer_list[3]
    return Response(get_detect(number),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def put_data():
    while True:
        time.sleep(0.5)
        try:
            data = q_put_img.pop()
            print("--------------------->put_data")
            if data:
                img_name, my_result, frame = data[0], data[1], data[2]
                frame_lt = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                # control_put = False
                header = {}
                cv2.imwrite('./logs_img/%s.jpg' % (img_name), frame_lt)
                files = {'file': open('./logs_img/%s.jpg' % (img_name), 'rb')}

                if "hat" in my_result:
                    lh = 0 if my_result['hat'] == "No" else 1
                else:
                    lh = 0

                if "coat" in my_result:
                    lf = 0 if my_result['coat'] == "No" else 1
                else:
                    lf = 0

                if "gloves" in my_result:
                    lg = 0 if my_result['gloves'] == "No" else 1
                else:
                    lg = 0

                if "shoes" in my_result:
                    ls = 0 if my_result['shoes'] == "No" else 1
                else:
                    ls = 0

                para = {"Camer_ID": '001',
                        "Worker_ID": "unknown",
                        "hat": lh,
                        "frock": lf,
                        "glove": lg,
                        "shoe": ls,
                        "timestamp": int(round(time.time() * 1000)),
                        "file_name": img_name}
                r = requests.post(url, files=files, data=para, headers=header)
                print(para)
                time.sleep(1)
        except KeyboardInterrupt:
            break

        except IndexError:
            print("--------------------->q_put_img empty!")

            

def read_video(dq):
    # video_reader = cv2.VideoCapture("rtsp://admin:ad123456@192.168.199.220/Streaming/Channels/1")
    # video_reader = cv2.VideoCapture("rtsp://demo.easydss.com:10554/aidong_demo")
    video_reader = cv2.VideoCapture(0)
    # video_reader = cv2.VideoCapture("./test_video/benchi_test_video.mp4")
    # sz = (640, 360) #the same as myout.write(img)

    # video_reader.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    # video_reader.set(3, 1280)  
    # video_reader.set(4, 720)
    # video_reader.set(cv2.CAP_PROP_FPS, 25)
    while True:
        time.sleep(0.01)
        ret1, img = video_reader.read()
        # print("-------------->", img.shape)

        if not ret1 or type(img) == type(None):
            print("=====read viedo error!")
            continue
        else:
            dq.append(img)

def mk_dir():
    # save images.
    base_file = "clip_img"
    for x, y, z in os.walk(base_file):
        print(y)
        filename_len = len(y)
        print(filename_len)
        break
    save_file = "clip_img/images" + str(filename_len)
    os.makedirs(save_file)
    return save_file

def thread_start(dq):
    t_read_video = threading.Thread(target=read_video, args=(dq,))
    t_put_data = threading.Thread(target=put_data)
    t_read_video.start() 
    t_put_data.start()  

def iou_cal(lis1, lis2):
    max_x = max(lis2[2], lis2[0], lis1[2], lis1[0])
    min_x = min(lis2[2], lis2[0], lis1[2], lis1[0])
    max_y = max(lis2[3], lis2[1], lis1[3], lis1[1])
    min_y = min(lis2[3], lis2[1], lis1[3], lis1[1])
    l1_x = abs(lis1[2] - lis1[0])
    l1_y = abs(lis1[3] - lis1[1])

    l2_x = abs(lis2[2] - lis2[0])
    l2_y = abs(lis2[3] - lis2[1])

    a = l1_x + l2_x - (max_x-min_x)
    b = l2_y + l1_y - (max_y-min_y)

    my_iou = a*b/(l1_x*l1_y + l2_x*l2_y-a*b)
    return my_iou

def new_loc(lis1, lis2):
    center_x = int((lis2[0] + lis2[2])/2)
    center_y = int((lis2[1] + lis2[3])/2)
    
    mid_lis = [int((x1+x2)/2) for x1, x2 in zip(lis1, lis2)]
    mid_x = int((mid_lis[0] + mid_lis[2])/2)
    mid_y = int((mid_lis[1] + mid_lis[3])/2)

    x_shift = center_x - mid_x
    y_shift = center_y - mid_y

    return [mid_lis[0]+x_shift, mid_lis[1]+y_shift, mid_lis[2]+x_shift, mid_lis[3]+y_shift]
          
def get_detect(id_video):
    global control_color
    dq = deque(maxlen=1)
    thread_start(dq)

    fbox = []
    bbox = []
    person_dit = {}
    threshold_iou = 0.8

    my_track_dict = {}
 
    save_file = mk_dir()
    num = 0
    t1 = time.time()
    while True:
        #avoid the memory error.
        if len(my_track_dict)>50:
            my_track_dict = {}
        if dq:
            img = dq.pop()
        else:
            time.sleep(0.05)
            continue
        
        start_time = time.time()

        num += 1 
        if num % 5000 == 1:
            cv2.imwrite(save_file+"/_{}.jpg".format(num), img)

        img_h, img_w, img_ch = img.shape
        print(img.shape)
        #2、防止裁剪或推理时把画的框裁剪上
        show_image = img.copy()
        frame = img.copy()

        #the predict of person.
        out = preson_detect(img)
        # print("------------->", out)

        boxes = []
        
        #transform the object detection data to input tracter
        for i in range(len(out)):
            #========my_setting==============
            if out[i, 2] > 0.7:
                # print(out[i])
                left = int(out[i, 3]*img_w)
                top = int(out[i, 4]*img_h)
                p_w = int(out[i, 5]*img_w-out[i, 3]*img_w)
                p_h = int(out[i, 6]*img_h-out[i, 4]*img_h)

                right = left + p_w
                bottom = top + p_h
                
                #detect the person in setting area.
                point1 = [int((left+right)/2), bottom] 
                my_index = inner_point(point1)
                # if my_index:
                #========my_setting==============
                if True:
                    boxes.append([left, top, p_w, p_h]) 
                    print(boxes)
        if not boxes:
            # time.sleep(0.02)
            continue

        
        #tracker algorithm 追踪算法
        if not fbox:
            fbox = boxes
            init_len = len(fbox)
            for i_d in range(1, init_len+1):
                person_dit[str(i_d)] = fbox[i_d-1]
            continue
        else:
            bbox = boxes

        print("---------->fbox", fbox)
        print("---------->bbox", bbox)
        person_new = person_dit.copy()
        #删除上一贞出现这贞消失的数据，并更新一直存在的坐标
        for key in person_dit:
            for one in bbox:
                my_iou = iou_cal(person_new[key], one)
                if my_iou > threshold_iou:
                    person_new[key] = new_loc(person_new[key], one)
                    break
            else:
                del person_new[key]
        
        #添加这贞新增而上一贞不存在的数据
        for one in bbox:
            for key in person_dit:
                my_iou = iou_cal(person_dit[key], one)
                if my_iou > threshold_iou:
                    break
            else:
                init_len += 1
                person_new[str(init_len)] = one    #等于the value of this frame 

        fbox = bbox
        
        if not person_dit:
            continue
        

        t2 = time.time()
        detect_time = t2-t1
        #========my_setting==============
        control_time = 1  #detect one time in m second
        if detect_time > control_time:
            t1 = time.time()
        
        indexIDs = []
        for my_key in person_new:
            #if my_key increase or time lt 3s, will be re_detection.
            if my_key not in my_track_dict.keys() or detect_time>control_time:
                # print(my_key)
                # print(my_track_dict.keys())
                #the code of processing the person box.
                label_dict = get_labels(img, person_new[my_key])
                print("**"*20, label_dict)

                if type(label_dict) == type(None):
                    continue

                if "coat" not in label_dict.keys():
                    continue
                my_track_dict[my_key] = label_dict
                            # draw the attr of person.

            indexIDs.append(int(my_key))
            frame = draw_person_attr(frame, my_track_dict[my_key], person_new[my_key], control_color)
            # color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            #define the color of rectangle.
            if my_track_dict[my_key]["coat"] == "Yes":
                color_rect = (0, 255, 0)
            else:
                color_rect = (0, 0, 255)

            cv2.rectangle(frame, (int(person_new[my_key][0]), int(person_new[my_key][1])), (int(person_new[my_key][2]), int(person_new[my_key][3])),(color_rect), 3)

            #draw the boxs of track.
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            if True:
                cv2.putText(frame,str(my_key),(int(person_new[my_key][0]), int(person_new[my_key][1] -50)),0, 5e-3 * 150, (225,0,0),2) 
                class_names = "person"
                cv2.putText(frame, str(class_names),(int(person_new[my_key][0]), int(person_new[my_key][1] -20)),0, 5e-3 * 150, (225,0,0),2)
            i += 1  

        cv2.imshow("img", frame)
        key = cv2.waitKey(33)
        if key == 27:
            cv2.destroyAllWindows()
            video_reader.release()
            break

  
if __name__ == "__main__":
    #ATTENTION!!!!!!
    #===================search "my_setting" to set the parameters.===============
    get_detect(0)

#frame = cv2.imread(filename)
#1、读取中文路径
#img = cv2.imdecode(np.fromfile(filename,dtype=np.uint8), cv2.IMREAD_COLOR)


# fps = 0
# fps1 = time.time()


# #TEST FPS
# fps2 = time.time()
# fps += 1
# if fps2 - fps1 > 1:
#     print(fps)
#     fps = 0
#     fps1 = time.time()


# cv2.namedWindow("YOLO4_Deep_SORT", 0)
# #cv2.resizeWindow('YOLO4_Deep_SORT', 640, 480)
# cv2.imshow('YOLO4_Deep_SORT', frame)
# #myout.write(frame)

# key = cv2.waitKey(1)
# if key == 27:
#     # myout.release()
#     # video_reader.release()
#     cv2.destroyAllWindows()            
#     break