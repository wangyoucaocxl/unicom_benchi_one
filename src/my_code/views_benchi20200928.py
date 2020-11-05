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

# from deep_sort import preprocessing
# from deep_sort import nn_matching
# from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker
# from tools import generate_detections as gdet
# from deep_sort.detection import Detection as ddet

# from keras import backend
# import tensorflow as tf
import requests
from collections import deque

import warnings
warnings.filterwarnings("ignore")

import vlc
import ctypes
# import time
# import sys
# import cv2
# import numpy
# from PIL import Image

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


control_color = True
# #是否显示中文标注
# while True:
#     a_in = input("Please input a number(0: simple, 1: complex)--->")
#     if a_in=="0":
#         control_color = False
#         break
#     elif a_in=="1":
#         control_color = True
#         break
#     else:
#         print("=================input error!====================")
#         continue

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
            print("=========================put_data", data[2])
            if data:
                img_name, my_result, frame = data[0], data[1], data[2]
                frame_lt = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                # control_put = False
                header = {}
                cv2.imwrite('./logs_img/%s.jpg' % (img_name), frame_lt)
                files = {'file': open('./logs_img/%s.jpg' % (img_name), 'rb')}

                if "hat" in my_result:
                    lh = 0 if my_result['hat'] == 0 else 1
                else:
                    lh = 0

                if "coat" in my_result:
                    lf = 0 if my_result['coat'] == 0 else 1
                else:
                    lf = 0

                if "gloves" in my_result:
                    lg = 0 if my_result['gloves'] == 0 else 1
                else:
                    lg = 0

                if "shoes" in my_result:
                    ls = 0 if my_result['shoes'] == 0 else 1
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
    # video_reader = cv2.VideoCapture("rtsp://demo.easydss.com:10054/xDClMP5Mg")
    
    # video_reader = cv2.VideoCapture("./test_video/benchi_test_video.mp4")
    # sz = (640, 360) #the same as myout.write(img)

    video_reader = cv2.VideoCapture(0)
    video_reader.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    video_reader.set(3, 1280)  
    video_reader.set(4, 720)
    # video_reader.set(cv2.CAP_PROP_FPS, 25)
    while True:
        time.sleep(0.03)
        ret1, img = video_reader.read()
        # print(img.shape)
        if not ret1 or type(img) == type(None):
            print("=====read viedo error!")
            continue
        else:
            dq.append(img)

def read_video_vlc(dq):
     
    vlcInstance = vlc.Instance()
    # 机场内
    # m = vlcInstance.media_new("")
    # 机场外
    # 记得换url,最好也和上面一样进行测试一下
    # url = "rtsp://demo.easydss.com:10554/200825"
    # url = "rtsp://demo.easydss.com:10554/aidong_demo"
    url = "rtsp://admin:ad123456@192.168.199.220/Streaming/Channels/1"
    m = vlcInstance.media_new(url)
    mp = vlc.libvlc_media_player_new_from_media(m)
     
    # ***如果显示不完整，调整以下宽度和高度的值来适应不同分辨率的图像***
    video_width = 1280
    video_height = 720
     
    size = video_width * video_height * 4
    buf = (ctypes.c_ubyte * size)()
    buf_p = ctypes.cast(buf, ctypes.c_void_p)
     
    VideoLockCb = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
     
     
    @VideoLockCb
    def _lockcb(opaque, planes):
        # print("lock", file=sys.stderr)
        planes[0] = buf_p
     
    @vlc.CallbackDecorators.VideoDisplayCb
    def _display(opaque, picture):
        img = Image.frombuffer("RGBA", (video_width, video_height), buf, "raw", "BGRA", 0, 1)
        opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        print(opencv_image.shape)
        # print(np.mean(opencv_image))
        if type(img) == type(None):
            print("=====read viedo error!")
        else:
            dq.append(opencv_image)
            # dq.append(opencv_image)
        # cv2.imshow('image', opencv_image)
        # cv2.waitKey(50)
     
    vlc.libvlc_video_set_callbacks(mp, _lockcb, None, _display, None)
    mp.video_set_format("BGRA", video_width, video_height, video_width * 4)
    while True:
        mp.play()
        time.sleep(1)

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


'''
def get_detect(id_video):

    from collections import deque
    dq = deque(maxlen=1)
    t1 = threading.Thread(target=read_video, args=(dq,))
    t1.start() 


    my_num = 0
    num = 0

    while True:

        start_time = time.time()
        if dq:
            img = dq.pop()
        else:
            cv2.waitKey(20)
            continue
        
        show_image = img.copy()
        
        out = preson_detect(img)
        img_h, img_w, img_ch = img.shape
                           
        for i in range(len(out)):
            #人形框限定条件，根据置信度限定不符合要求的人形框。
            if out[i, 2] < 0.7:
                continue
            
            #attentions
            x0, y0, x1, y1 = out[i, 3:7]
            start_x = int(img_w * x0)
            start_y = int(img_h * y0)
            end_x = int(img_w * x1)
            end_y = int(img_h * y1)
            
            
            point1 = [int((start_x+end_x)/2), end_y] 

            my_index = inner_point(point1)
            if not my_index:
                continue

            clip_images = img[start_y:end_y, start_x:end_x]
            h, w, c = clip_images.shape
            
            #根据图像大小限定不符合要求的人形框。
            if h > 96 and w > 48:
                
                #3、保存图片,修改保存名称
                my_num += 1
                #save_name = os.path.join(save_dir, "v1_person{}.jpg".format(my_num))
                #cv2.imwrite(save_name, clip_images)
                
                label_dict = muti_attr(clip_images)
                print(label_dict)
                
                i_count=0
                font=cv2.FONT_HERSHEY_COMPLEX
                if "coat" in label_dict.keys():
                    #4、在原始图像上画矩形。
                    if label_dict["coat"] == "Yes":
                        cv2.rectangle(show_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(show_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)     
                        
                    #5、在原始图像上画矩形。
                    for key in label_dict:
                        text=key+":"+str(label_dict[key])
                        if label_dict[key] == "Yes":
                            cv2.putText(show_image,text,(end_x,start_y+i_count),font,0.7,(0,255,0),2)
                        else:
                            cv2.putText(show_image,text,(end_x,start_y+i_count),font,0.7,(0,0,255),2)
                        i_count+=20
        
        draw_muti(show_image)        
        end_time = time.time()
        mytime = (end_time-start_time)*1000
        print(mytime)

        print(show_image.shape)


#        ret2, jpeg = cv2.imencode('.jpg', show_image)
#        yield (b'--frame\r\n'
#               b'application/octet-stream: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
        cv2.namedWindow("image{}".format("benchi"), cv2.WINDOW_NORMAL)
        cv2.imshow("image{}".format("benchi"), show_image)
        key = cv2.waitKey(33)
        if key == 27:
            break
        
    cv2.destroyAllWindows()

'''
def get_detect(id_video):

    global control_color
    dq = deque(maxlen=1)
    t_read_video = threading.Thread(target=read_video, args=(dq,))
    # t_read_video = threading.Thread(target=read_video_vlc, args=(dq,))
    t_put_data = threading.Thread(target=put_data)
    t_read_video.start() 
    t_put_data.start() 
    
    # myout = save_video(video_reader, "./video.mp4", sz)
    my_track_dict = {"0": None} #save the info of track_id
    track_smooth_dict = {} #smooth the imshow

    
    #创建追踪器
    tracker = cv2.MultiTracker_create()
    init_once = False
    

    save_file = mk_dir()
    num = 0    #有数据循环次数
    success_num = 0 #有检测次数
    t1 = time.time()  #重新检测时间初始值
    my_result = {} #存储缓冲数据
    index_key = 0
    lenth_boxs = 0

    while True:
        
        #avoid the memory error.
        if len(my_track_dict)>50:
            my_track_dict = {}  #save the info of track_id
        if len(my_result) > 50:
            my_result = {}      #存储缓冲数据
        # print(len(my_track_dict))
        
        #read camera data
        if dq:
            img = dq.popleft()
            print("=====================", img.shape)
        else:
            time.sleep(0.05)
            continue
        
 
        start_time = time.time()  #开始计时,测试单贞照片处理时间
        num += 1 

        
        if num % 100 == 1:
            cv2.imwrite(save_file+"/_{}.jpg".format(num), img)
        img_h, img_w, img_ch = img.shape
        print(img.shape)

        #2、防止裁剪或推理时把画的框裁剪上
        show_image = img.copy()
        frame = img.copy()

        #the predict of person.
        boxs, confidence, class_names = [], [], []
        out = preson_detect(img)
    
        #过滤规定区域内的人性框进行判断，transform the object detection data to input tracter
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
                # my_index = inner_point(point1)
                my_index = True
                print("my_index", my_index)
                if my_index:
                    boxs.append([left, top, p_w, p_h]) 
                    class_names.append("person")
                    confidence.append(out[i, 2])
       

        #========my_setting==============
        t2 = time.time()
        detect_time = t2 - t1      
        control_time = 2  ##setting detect time， detect one time in m second
        control_buttom = False
        if detect_time > control_time:
            control_buttom =  True
            t1 = time.time()

        # if not init_once or control_time>5 or lenth_boxs!=len(boxs):
        if not init_once or control_buttom:
            # tracker = cv2.MultiTracker_create()
            for one_box in boxs: 
                ok = tracker.add(cv2.TrackerMIL_create(), frame, tuple(one_box))
            init_once = True
            index_key = max([int(i_key) for i_key in my_track_dict.keys()])


        ok, boxes = tracker.update(frame)
        print(ok, boxes)

        for index, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

            cv2.rectangle(frame, p1, p2, (200,0,0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(index), (int(newbox[0]), int(newbox[1]-20)), font, 1.2, (0,0,225), 2)

        cv2.namedWindow("aidong_unicom", 0)
        cv2.imshow('aidong_unicom', frame)
        key = cv2.waitKey(33)
        # if key == 27:
        #     myout.release()
        #     video_reader.release()
        #     cv2.destroyAllWindows()            
        #     break
       
'''
            my_key = str(int(index) + index_key)

            #========my_setting==============
            #if my_key increase or time lt xxs, will be re_detection.
            if my_key not in my_track_dict.keys() or detect_time>control_time:

                label_dict = get_labels(img, newbox)   #pbox为人单个人形框
                # print("============================================", label_dict)
                
                #label_dict 可能为None
                if type(label_dict) == type(None):
                    continue
                if "coat" not in label_dict.keys():
                    continue


                my_track_dict[my_key] = label_dict

            if type(my_track_dict[my_key]) == type(None):
                continue 
            
            frame = draw_person_attr(frame, my_track_dict[my_key], newbox, control_color)   # draw the attr of person.

            # define the color of rectangle.
            if my_track_dict[my_key]["coat"] == 1:
                color_rect = (0, 255, 0)
            else:
                color_rect = (0, 0, 255) 
            
            # smooth the rectangle. 平滑矩形框 
            #center_loc = [int((newbox[0]+newbox[2])/2), int((newbox[1]+newbox[3])/2)]
            if my_key not in track_smooth_dict.keys():
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2,(color_rect), 3)
                track_smooth_dict[my_key] = newbox
            else:
                fbox = track_smooth_dict[my_key]
                a = int((newbox[0]+fbox[0])/2)
                b = int((newbox[1]+fbox[1])/2)
                c = int((newbox[2]+fbox[2])/2)
                d = int((newbox[3]+fbox[3])/2)
                cv2.rectangle(frame, (a, b), (c, d),(color_rect), 3)
                track_smooth_dict[my_key] = newbox

            #draw the boxs of track.
            #cv2.rectangle(frame, (int(newbox[0]), int(newbox[1])), (int(newbox[2]), int(newbox[3])),(color), 3)

            #添加文字显示
            # if True:
            #     cv2.putText(frame,str(my_key),(newbox[0], newbox[1] -50),0, 5e-3 * 150, (0, 255, 0),2) 
            #     if len(class_names) > 0:
            #        class_name = class_names[0]
            #        cv2.putText(frame, str(class_names[0]),(newbox[0], newbox[1] -20),0, 5e-3 * 150, (0, 255, 0),2)

            print("---------2----------->")

            # print("my_track_dict[my_key]", my_track_dict[my_key])
            if my_key not in my_result:
                #不要直接等于，会出错，深浅拷贝问题
                my_result[my_key] = my_track_dict[my_key].copy()
                my_result[my_key]["mysum"] = 0
                # print("++++++init++++++++++",my_result[my_key])
            else:
                for key1 in  my_track_dict[my_key]:
                    if key1 in my_result[my_key]:
                        if key1 == "face":
                            if my_track_dict[my_key]["face"] != "Unknown":
                                my_result[my_key]["face"] = my_track_dict[my_key]["face"]
                        else:
                            my_result[my_key][key1] = my_result[my_key][key1] + my_track_dict[my_key][key1]
                    else:
                        my_result[my_key][key1] = my_track_dict[my_key][key1]

            my_result[my_key]["mysum"] = my_result[my_key]["mysum"] + 1                
            # print(my_key, "-------->", my_result[my_key])

            if my_result[my_key]["mysum"] > 50:
                # print("-------->", my_result[my_key])
                my_cum_result = {}
                for key2 in my_result[my_key]:
                    if key2 == "face":
                        my_cum_result[key2] = my_result[my_key][key2]
                    else:
                        if my_result[my_key][key2]/my_result[my_key]["mysum"] < 0.3:
                            my_cum_result[key2] = 0
                        else:
                            my_cum_result[key2] = 1

                del my_cum_result["mysum"]  #删除“mysum”属性再上传
                del my_result[my_key]    #减少内存负担，上传的人员要删除
                

                if "coat" in my_cum_result and "hat" in my_cum_result and "gloves" in my_cum_result and "shoes" in my_cum_result:
                    pic_name = str(id_video) +  "_" + str(int(time.time())) + "_" + my_key
                    # put_data(my_key, my_result, frame)
                    # print("--------put_data-------->", pic_name, newbox)
                    q_put_img.append([pic_name, my_cum_result, show_image[newbox[1]:newbox[3], newbox[0]:newbox[2]]])
                    # my_result[my_key] = {"hat":0, "coat":0, "gloves":0,"shoes":0, "mysum":0}
        
        #draw the gurdline.画警戒线
        draw_muti(frame)

        end_time = time.time()  #结束计时,测试单贞照片处理时间
        my_one_time = (end_time - start_time) * 1000
        print("====={}=====".format(num), my_one_time)  


        # frame = cv2.resize(frame, (640, 360))
        # ret2, jpeg = cv2.imencode('.jpg', frame)
        # yield (b'--frame\r\n'
        #        b'application/octet-stream: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
'''

             
  
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

# # 控制上传频率
# # print(bbox)
# success_num += 1
# if success_num % 50 == 1:
#     my_result[my_key] = {"hat":0, "coat":0, "gloves":0,"shoes":0, "mysum":0, "face":""}
# for key1 in my_track_dict[my_key]:
#     if key1 == "face":
#         if my_track_dict[my_key]["face"] != "Unknown":
#             my_result[my_key]["face"] = my_track_dict[my_key]["face"]
#     else:
#         try:
#             # 加入有新key出现会报错
#             my_result[my_key]
#         except:
#             my_result[my_key] = {"hat":0, "coat":0, "gloves":0,"shoes":0, "mysum":0, "face":""}

#         my_result[my_key][key1] = my_result[my_key][key1] + my_track_dict[my_key][key1]
# my_result[my_key]["mysum"] = my_result[my_key]["mysum"] + 1

# if success_num % 50 == 49:
#     print("-------->", my_result[my_key])
#     my_cum_result = {}

#     for key2 in my_result[my_key]:
#         if key2 == "face":
#             my_cum_result[key2] = my_result[my_key][key2]
#         else:
#             if my_result[my_key][key2]/my_result[my_key]["mysum"] < 0.3:
#                 my_cum_result[key2] = 0
#             else:
#                 my_cum_result[key2] = 1
#     del my_cum_result["mysum"]
    
#     # print(my_cum_result）
#     pic_name = str(id_video) + str(int(time.time())) + "_" + my_key
#     # put_data(my_key, my_result, frame)

#     print("--------put_data-------->", pic_name, bbox)
#     q_put_img.append([pic_name, my_cum_result, frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]])
#     my_result[my_key] = {"hat":0, "coat":0, "gloves":0,"shoes":0, "mysum":0}