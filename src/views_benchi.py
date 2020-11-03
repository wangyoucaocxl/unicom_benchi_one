#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from flask import Flask, render_template, Blueprint,Response,request
from src.person_detect_used import preson_detect
from src.my_functions import *
#from concurrent.futures import ThreadPoolExecutor
import threading
import cv2
import time, os, json, glob
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from urllib.request import urlretrieve
import sys
import warnings
import requests
from collections import deque

import vlc
import ctypes
import warnings
warnings.filterwarnings("ignore")

#url = "http://fabx.aidong-ai.com/snapshot/worker"
url = "http://fabx.aidong-ai.com/frock/api/snapshot"

#====1mysetting======
#os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
app = Flask(__name__, static_folder='../static', static_url_path='/static')
cors = CORS()
cors.init_app(app)
main = Blueprint('main', __name__)
app.register_blueprint(main,url_prefix = "")

camer_list = ["rtsp://demo.easydss.com:10054/xDClMP5Mg",
              "rtsp://demo.easydss.com:10054/shilei_kernel_test",
              "rtsp://demo.easydss.com:10554/aidong_demo",
              "rtsp://admin:ad123456@192.168.199.220/Streaming/Channels/1"
              ]
camer_count = 1
q_put_img = deque(maxlen=1)
control_color = True

@app.route('/')
def index():
    global camer_count
    #camer_count = getcountCameras() 
    camer_count = "1"  #修改让只出现一个窗口
    return render_template('index.html')  #====2mysetting======

def clearCapture(capture): 
    capture.release() 
    cv2.destroyAllWindows() 

@app.route('/countCameras',methods = ["GET"])
def countCameras():
    global camer_count
    try:
        camer_count = request.args.get("name")
        print("camer_count", not camer_count)
        print("camer_count", type(camer_count))
        if not camer_count:
            camer_count = str(1)
    except:
        camer_count = str(1)
    print("------>", camer_count)
    return camer_count

@app.route('/detect_video1')
def detect_video1():
    global camer_list
    rtsp_addr = camer_list[0]
    return Response(get_detect(rtsp_addr, "camera001"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_video2')
def detect_video2():
    global camer_list
    rtsp_addr = camer_list[1]
    return Response(get_detect(rtsp_addr, "camera002"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_video3')
def detect_video3():
    global camer_list
    rtsp_addr = camer_list[2]
    return Response(get_detect(rtsp_addr, "camera003"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_video4')
def detect_video4():
    global camer_list
    rtsp_addr = camer_list[3]
    return Response(get_detect(rtsp_addr, "camera004"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image/sync', methods=['POST'])
def image_sync():
    body = json.loads(request.data.decode())

    base_save_path = os.path.join(os.curdir, 'face_dlib/dataset') 
    
    if not isinstance(body, list):
        return {'code': -1, 'msg':'request data invalid'}

    success_info = {}
    
    for person_info in body:
        if not 'personId' in person_info:
                continue

        person_id = person_info['personId']
        image_urls = person_info['imageUrls'] if 'imageUrls' in person_info else []
        
        try:
            save_path =  os.path.join(base_save_path, person_id)

            # remove all images in every sync time
            if os.path.exists(save_path):
                os.system(f'rm -rf {save_path}')

            if not os.path.exists(save_path) and image_urls:
                os.mkdir(save_path)

            for image_url in image_urls:
                image_name = image_url.split('/')[-1]
                urlretrieve(image_url, os.path.join(save_path, image_name))
                success_info[person_id] = True
        except Exception as e:
            print(f'save person failed : {person_id}, message : {e}')
            success_info[person_id] = False
            
    return {'code': 0, 'successCount': len(list(filter(lambda x: x, success_info.values()))), 'info': success_info}

def get_fence(camera_id):
    '''获取电子围栏'''
    fence_res = requests.get(f'http://fabx.aidong-ai.com/frock/api/fence?camerId={camera_id}').json()
    pts = fence_res['data']["list"][0]["vertexes"]
    print("---------pts----------:", pts)
    return pts

def put_data(camerId):
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

                para = {"cameraId": camerId,
                        "workerNumber":"003",
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

def read_video(dq, rtsp_addr):
    # video_reader = cv2.VideoCapture("rtsp://admin:ad123456@192.168.199.220/Streaming/Channels/1")
    # video_reader = cv2.VideoCapture("rtsp://demo.easydss.com:10554/aidong_demo")
    #video_reader = cv2.VideoCapture("rtsp://demo.easydss.com:10054/xDClMP5Mg")
    #video_reader = cv2.VideoCapture("rtsp://demo.easydss.com:10054/shilei_kernel_test")
    #video_reader = cv2.VideoCapture("rtsp://demo.easydss.com:10054/Seven")
    
    # video_reader = cv2.VideoCapture("./test_video/benchi_test_video.mp4")
    # sz = (640, 360) #the same as myout.write(img)

    # video_reader = cv2.VideoCapture(0)
    #video_reader.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    #video_reader.set(3, 1280)  
    #video_reader.set(4, 720)
    # video_reader.set(cv2.CAP_PROP_FPS, 25)
    video_reader = cv2.VideoCapture(rtsp_addr)
    while True:
        time.sleep(0.03)
        ret1, img = video_reader.read()
        # print(img.shape)
        if not ret1 or type(img) == type(None):
            print("=====read viedo error!")
            continue
        else:
            dq.append(img)

def read_video_vlc(dq, rtsp_addr):
     
    vlcInstance = vlc.Instance()
    # 机场内
    # m = vlcInstance.media_new("")
    # 机场外
    # 记得换url,最好也和上面一样进行测试一下
    # url = "rtsp://demo.easydss.com:10554/200825"
    # url = "rtsp://demo.easydss.com:10554/aidong_demo"
    # url = "rtsp://admin:ad123456@192.168.199.220/Streaming/Channels/1"
    url = rtsp_addr
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
        filenamelen = len(y)
        print(filenamelen)
        break
    savefile = "clip_img/images" + str(filenamelen)
    os.makedirs(savefile)
    return savefile

def edn_distance(lis1, lis2):
    #print("========", lis1, lis2)
    x1, y1 = (lis1[0]+lis1[2])/2, (lis1[1]+lis1[3])/2
    x2, y2 = (lis2[0]+lis2[2])/2, (lis2[1]+lis2[3])/2
    if -20<x1-x2<20 and -20<y1-y2<20:
        return True
    else:
        return False

def new_loc(lis1, lis2):
    #print("========", lis1, lis2)
    x1 = int((lis1[0]+lis2[0])/2)
    y1 = int((lis1[1]+lis2[1])/2)
    x2 = int((lis1[2]+lis2[2])/2)
    y2 = int((lis1[3]+lis2[3])/2)
    return [x1, y1, x2, y2] 



def get_detect(rtsp_addr, camerId):

    global control_color
    dq = deque(maxlen=1)
    t_read_video = threading.Thread(target=read_video, args=(dq, rtsp_addr))
    # t_read_video = threading.Thread(target=read_video_vlc, args=(dq,))
    t_put_data = threading.Thread(target=put_data, args=(camerId,))
    t_read_video.start() 
    t_put_data.start() 
    
    # myout = save_video(video_reader, "./video.mp4", sz)
    my_track_dict = {} #save the info of track_id
    

    save_file = mk_dir()
    num = 0    #有数据循环次数
    t1 = time.time()  #重新检测时间初始值
    my_result = {} #存储缓冲数据

    fbox = []
    bbox = []
    person_dit = {}
    
    #加载电子围栏
    try:
        points2 = get_fence(camerId)
        print("-------"*100, points2)
        points2 = [[int(float(pt[0])*1280), int(float(pt[1])*720)]for pt in points2]
        print("-------"*100, points2)
    except:
        points2 = [[0,0], [0,10],[10,10],[10,0]]
        print("no gardline!please create new line.")

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

        if num % 1000 == 1:
            pass
            #cv2.imwrite(save_file+"/_{}.jpg".format(num), img)

        img_h, img_w, img_ch = img.shape
        print(img.shape)

        #2、防止裁剪或推理时把画的框裁剪上
        show_image = img.copy()
        frame = img.copy()

        #the predict of person.
        boxes = []
        out = preson_detect(img)
        
        #print(out)
        #过滤规定区域内的人性框进行判断，transform the object detection data to input tracter
        for i in range(len(out)):
            #========my_setting==============
            if out[i, 2] > 0.7:
                #print("------------------>===================================>", out)
                # print(out[i])
                left = int(out[i, 3]*img_w)
                top = int(out[i, 4]*img_h)
                right = int(out[i, 5]*img_w)
                bottom = int(out[i, 6]*img_h)
                
                #detect the person in setting area.
                point1 = [int((left+right)/2), bottom]
                my_index = inner_point(point1, points2)
                my_index = True
                print("my_index", my_index)
                if my_index:
                    boxes.append([left, top, right, bottom]) 
                    
        if boxes:
            #跟踪算法
            if not fbox:
                fbox = boxes
                init_len = len(fbox)
                for i_d in range(0, init_len):
                    person_dit[str(i_d)] = fbox[i_d][-4:]
                continue
            else:
                bbox = boxes
            #print("------------------>===================================>", boxes)
            # print("------------------>===================================>", 2)
            del_list = []
            for key_a in person_dit:
                for one_a in bbox:
                    my_flag = edn_distance(person_dit[key_a], one_a)
                    if my_flag:
                        person_dit[key_a] = new_loc(person_dit[key_a], one_a)
                        # person_dit[key_a] = one_a
                        break
                else:
                    del_list.append(key_a)

            for key_b in del_list:
                del person_dit[key_b]

            for one_c in bbox:
                for key_c in person_dit:
                    my_flag = edn_distance(person_dit[key_c], one_c)
                    if my_flag:
                        break
                else:
                    init_len += 1
                    person_dit[str(init_len)] = one_c[-4:]

            fbox = bbox

            #print("---------------------------------->===================================>", 3)
            #检测条件设置
            t2 = time.time()
            detect_time = t2 - t1      
            control_time = 1  ##setting detect time， detect one time in m second
            if detect_time > control_time:
                t1 = time.time() 
            
            #属性检测
            for my_key in person_dit:
                #加入控制条件
                # if my_key not in my_track_dict.keys() or detect_time>control_time:
                if True:
                    #the code of processing the person box.
                    label_dict = get_labels(frame, person_dit[my_key])   #pbox为人单个人形框
                    # print("============================================", label_dict)
                    
                    #label_dict 可能为None
                    if type(label_dict) == type(None):
                        continue
                    if "coat" not in label_dict.keys():
                        continue

                    my_track_dict[my_key] = label_dict

               
                #draw the boxs of track.
                new_box =  person_dit[my_key]
                p1 = (new_box[0], new_box[1])
                p2 = (new_box[2], new_box[3])
                cv2.rectangle(show_image, p1, p2, (255,0,0), 2, 1)
                #添加文字显示
                cv2.putText(show_image, "person:"+my_key, (new_box[0], new_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

                show_image = draw_person_attr(show_image, my_track_dict[my_key], new_box, control_color)   # draw the attr of person.


                #结果上传
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

                if my_result[my_key]["mysum"] > 10:
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
                    

                    #if "coat" in my_cum_result and "hat" in my_cum_result and "gloves" in my_cum_result and "shoes" in my_cum_result:
                    if "coat" in my_cum_result and "hat" in my_cum_result and "shoes" in my_cum_result:
                        pic_name = str(camerId) +  "_" + str(int(time.time())) + "_" + my_key
                        # put_data(my_key, my_result, frame)
                        print("--------put_data-------->")
                        loction = person_dit[my_key]
                        q_put_img.append([pic_name, my_cum_result, show_image[loction[1]:loction[3], loction[0]:loction[2]]])
                        # my_result[my_key] = {"hat":0, "coat":0, "gloves":0,"shoes":0, "mysum":0}
        
            #draw the gurdline.画警戒线
            draw_muti(show_image, points2)
            end_time = time.time()  #结束计时,测试单贞照片处理时间
            my_one_time = (end_time - start_time) * 1000
            print("====={}=====".format(num), my_one_time)  
            
            #print("-------------->", num, my_result)
            #cv2.namedWindow("aidong_unicom", 0)
            #cv2.imshow('aidong_unicom', show_image)
            #key = cv2.waitKey(1)

            show_image = cv2.resize(show_image, (640, 360))
            ret2, jpeg = cv2.imencode('.jpg', show_image)
            yield (b'--frame\r\n'
                   b'application/octet-stream: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            #draw the gurdline.画警戒线
            draw_muti(show_image, points2)
            end_time = time.time()  #结束计时,测试单贞照片处理时间
            my_one_time = (end_time - start_time) * 1000
            print("====={}=====".format(num), my_one_time)  
            
            print("-------------->", num, person_dit)
            #cv2.namedWindow("aidong_unicom", 0)
            #cv2.imshow('aidong_unicom', show_image)
            #key = cv2.waitKey(1)  

            show_image = cv2.resize(show_image, (640, 360))
            ret2, jpeg = cv2.imencode('.jpg', show_image)
            yield (b'--frame\r\n'
                   b'application/octet-stream: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')         


             

if __name__ == "__main__":
    #ATTENTION!!!!!!
    #===================search "my_setting" to set the parameters.===============
    get_detect(0)
