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



#是否显示中文标注
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

    video_reader.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    video_reader.set(3, 1280)  
    video_reader.set(4, 720)
    # video_reader.set(cv2.CAP_PROP_FPS, 25)
    while True:
        time.sleep(0.01)
        ret1, img = video_reader.read()

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
          
def get_detect(id_video):
    global control_color
    dq = deque(maxlen=1)
    
    t_read_video = threading.Thread(target=read_video, args=(dq,))
    t_put_data = threading.Thread(target=put_data)
    t_read_video.start() 
    t_put_data.start() 

    # myout = save_video(video_reader, "./video.mp4", sz)
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    counter = []
    my_track_dict = {} #save the info of track_id
    track_smooth_dict = {} #smooth the imshow
    pts = [deque(maxlen=30) for _ in range(9999)]
    
    #deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    list_file = open('detection_rslt.txt', 'w')
    
    save_file = mk_dir()
    num = 0
    t1 = time.time()

    while True:
        #avoid the memory error.
        if len(my_track_dict)>50:
            my_track_dict = {}
        print(len(my_track_dict))

        if dq:
            img = dq.pop()
        else:
            time.sleep(0.05)
            continue
        
        start_time = time.time()

        num += 1 
        if num % 500 == 1:
            cv2.imwrite(save_file+"/_{}.jpg".format(num), img)
        img_h, img_w, img_ch = img.shape
        print(img.shape)
        #2、防止裁剪或推理时把画的框裁剪上
        show_image = img.copy()
        frame = img.copy()

        #the predict of person.
        boxs, confidence, class_names = [], [], []
        out = preson_detect(img)
    
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
                if my_index:
                    boxs.append([left, top, p_w, p_h]) 
                    class_names.append("person")
                    confidence.append(out[i, 2])

        #start use the tracker        
        features = encoder(frame,boxs)
        # score to 1.0 here.
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        i = int(0)
        indexIDs = []
        #setting detect time
        t2 = time.time()
        detect_time = t2-t1
        #========my_setting==============
        control_time = 0.2  #detect one time in m second
        if detect_time > control_time:
            t1 = time.time()
       
        for det, track in zip(detections, tracker.tracks):  
            # if not track.is_confirmed() or track.time_since_update > 1:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            #print(track.track_id)
            #draw the boxs of object detection.
            pbox = det.to_tlbr()
            #cv2.rectangle(frame,(int(pbox[0]), int(pbox[1])), (int(pbox[2]), int(pbox[3])),(255,255,255), 2)

            my_key = str(int(track.track_id))
            #========my_setting==============
            #if my_key increase or time lt 3s, will be re_detection.
            if my_key not in my_track_dict.keys() or detect_time>control_time:
                # print(my_key)
                # print(my_track_dict.keys())
                #the code of processing the person box.
                label_dict = get_labels(img, pbox)
                print("**"*20, label_dict)

                if type(label_dict) == type(None):
                    continue

                if "coat" not in label_dict.keys():
                    continue
                my_track_dict[my_key] = label_dict

            
            # draw the attr of person.
            frame = draw_person_attr(frame, my_track_dict[my_key], pbox, control_color)

            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            #define the color of rectangle.
            if my_track_dict[my_key]["coat"] == "Yes":
                color_rect = (0, 255, 0)
            else:
                color_rect = (0, 0, 255) 

            #center_loc = [int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)]
            if my_key not in track_smooth_dict.keys():
                print("---------------------------------------------------->")
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color_rect), 3)
                track_smooth_dict[my_key] = bbox
            else:
                fbox = track_smooth_dict[my_key]
                a = int((bbox[0]+fbox[0])/2)
                b = int((bbox[1]+fbox[1])/2)
                c = int((bbox[2]+fbox[2])/2)
                d = int((bbox[3]+fbox[3])/2)
                cv2.rectangle(frame, (a, b), (c, d),(color_rect), 3)
                track_smooth_dict[my_key] = bbox

            #draw the boxs of track.
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            if True:
                cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2) 
                if len(class_names) > 0:
                   class_name = class_names[0]
                   cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)
            i += 1
            
            # 控制上传频率
            if num % 200 == 1:
                my_result = my_track_dict[my_key]
                pic_name = str(int(time.time())) + "_" + my_key 
                # put_data(my_key, my_result, frame)
                q_put_img.append([pic_name, my_result, frame])

        count = len(set(counter))
        #draw the gurdline.
        draw_muti(frame)

        # cv2.putText(frame, "Total Pedestrian Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        # cv2.putText(frame, "Current Pedestrian Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)

        end_time = time.time()
        my_one_time = (end_time - start_time) * 1000
        print("====={}=====".format(num), my_one_time)

        frame = cv2.resize(frame, (640, 360))
        ret2, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'application/octet-stream: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
             
  
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