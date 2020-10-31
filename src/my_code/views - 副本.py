#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, Blueprint,Response
from src.shoe_show import predict_shoe
from src.hat_show import predict_hat
from src.frock_show import predict_frock
from src.glove_show import predict_glove
from src.person_detect_used import preson_detect

from concurrent.futures import ThreadPoolExecutor
#import threading
import cv2
import requests
import time


#import json,time
#url = "http://111.231.83.141:18890/snapshot/worker"
#url = "http://fabx.aidong-ai.com/snapshot/worker"
#from sklearn.externals import joblib
camer_list = []
#app = Flask(__name__)

#====1mysetting======
app = Flask(__name__, static_folder='../static', static_url_path='/static')
main = Blueprint('main', __name__)
app.register_blueprint(main,url_prefix = "")


time_count = 0
header ={}
log_list="a"

camer_count = 0
norm_size = 32

#clf = joblib.load('f_model.m')


@app.route('/')
def index():
    global camer_count
    camer_count = getcountCameras()
    return render_template('index3.html')  #====2mysetting======

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


#上传日志的线程
def myThread(label_dic,count_list,count,faces,img,Camer_ID):  
    global log_list,time_count
    time_count += 1
    print("**"*50, time_count)
    label_hat,label_frock,label_glove,label_shoe = label_dic.get("label_hat"),label_dic.get("label_frock"),label_dic.get("label_glove"),label_dic.get("label_shoe")
    if count_list[0] > 30 and count_list[1] > 30 and count_list[2] > 30 and count_list[3] > 30:
        if (str(label_hat) + str(label_frock) + str(label_glove) + str(label_shoe) == "1111") and (time_count % 20 == 1):
            cv2.imwrite('./logs_img/%s.jpg' % (count), img)
            files = {'file': open('./logs_img/%s.jpg' % (count), 'rb')}
            lh = 0 if label_hat == "1" else 1
            lf = 0 if label_frock == "1" else 1
            lg = 0 if label_glove == "1" else 1
            ls = 0 if label_shoe == "1" else 1
            para = {"Camer_ID": Camer_ID,
                    "Worker_ID": faces,
                    "hat": lh,
                    "frock": lf,
                    "glove": lg,
                    "shoe": ls,
                    "timestamp": int(round(time.time() * 1000)),
                    "file_name": count}
            #requests.post(url, files=files, data=para, headers=header)
            print(para)
        elif str(label_hat) + str(label_frock) + str(label_glove) + str(label_shoe) != "1111" and (time_count % 10 == 1):
            cv2.imwrite('./logs_img/%s.jpg' % (count), img)
            files = {'file': open('./logs_img/%s.jpg' % (count), 'rb')}
            lh = 0 if label_hat == "1" else 1
            lf = 0 if label_frock == "1" else 1
            lg = 0 if label_glove == "1" else 1
            ls = 0 if label_shoe == "1" else 1
            para = {"Camer_ID": Camer_ID,
                    "Worker_ID": faces,
                    "hat": lh,
                    "frock": lf,
                    "glove": lg,
                    "shoe": ls,
                    "timestamp": int(round(time.time() * 1000)),
                    "file_name": count}
            #requests.post(url, files=files, data=para, headers=header)
            print(para)



def drew_rectangle(img,label_dic,local_list,count_list):
    #print(label_dic.get('label_glove'))
    if label_dic.get("label_hat") == '1' and label_dic.get("label_frock") == '1':
        cv2.rectangle(img=img, pt1=(local_list[0], local_list[1]), pt2=(local_list[2], local_list[3]),
                      color=(0, 0, 255))
    elif label_dic.get("label_hat") == '0' and label_dic.get("label_frock") == '0':  #鞋子的逻辑没有要
        cv2.rectangle(img=img,pt1=(local_list[0], local_list[1]), pt2=(local_list[2], local_list[3]), color=(0,255 , 0))
    elif label_dic.get("label_hat") == '0':
           cv2.rectangle(img=img, pt1=(local_list[0], local_list[1]), pt2=(local_list[2], local_list[3]), color=(0,255, 255))
    elif label_dic.get("label_frock") == '0':
        cv2.rectangle(img=img, pt1=(local_list[0], local_list[1]), pt2=(local_list[2], local_list[3]), color=(255, 0, 255))
    elif label_dic.get("label_glove") == '0':
        cv2.rectangle(img=img, pt1=(local_list[0], local_list[1]), pt2=(local_list[2], local_list[3]), color=(255, 255,0 ))
    else:
         cv2.rectangle(img=img, pt1=(local_list[0], local_list[1]), pt2=(local_list[2], local_list[3]),
                      color=(0, 0, 255))


def get_detect(camer_id):
    global log_list
    # print("[INFO] Confirming inputs...")
    # mylog.addHandler(filehandler)

    image_paths = []
    batch_size = 1
    count = 0
    count_hat = 0
    count_frock = 0
    count_glove = 0
    count_shoe=0
    raw_hat = 0
    raw_frock = 0
    raw_glove = 0
    raw_shoe=0
    label_hat = '1'
    label_frock = '1'
    label_glove = '1'
    label_shoe = '1'

    faces_label = 1000
    count_faces = 0
#    last_faces = faces_label

    video_reader = cv2.VideoCapture(camer_id)
    video_reader.set(3, 640)
    
    with ThreadPoolExecutor(10) as executor:
        while (True):
            # while(True):
            ret, image = video_reader.read()
            if ret == True:
                image_paths += [image]
                count += 1
                # cv2.imshow('Image',image)
            if (len(image_paths) == batch_size) or (ret == False and len(image_paths) > 0):
                for img in image_paths:  
                    out = preson_detect(img)
                    img_h, img_w, img_ch = img.shape
                    for i in range(len(out)):
                        start = time.time()
                        x0, y0, x1, y1 = out[i, 3:7]
                        # print ("coordinate:",x0,y0,x1,y1)
                        start_x = int(img_w * x0)
                        start_y = int(img_h * y0)
                        end_x = int(img_w * x1)
                        end_y = int(img_h * y1)
                        if out[i, 2] < 0.8:
                            continue
                        if start_x<30 or end_x>580:
                        	   continue
                        w = end_x - start_x
                        h = end_y - start_y
                        images = img[start_y:end_y, start_x:end_x]
                        image_h, image_w, image_ch = images.shape
                        if image_h < norm_size or image_w < norm_size or image_h<300 or image_w>200:
                            continue
                            #    print("image")
                        ####head_start##
                        head_width = int(h*0.25+h*0.05)
                        if (int(start_x+end_x-head_width)/2<start_x or int((start_x+end_x+head_width)/2)>end_x):
                               continue              
                        if int((start_x+end_x-head_width)/2)> start_x+0.1*w:
                            image_head=img[int(start_y-h*0.05):int(start_y+h*0.25),int((start_x+end_x-head_width)/2):int((start_x+end_x+head_width)/2)]
                        else:
                            image_head=img[int(start_y-h*0.05):int(start_y+h*0.25),int(start_x+0.1*w):int(end_x-0.1*w)]
                        ####head_end##
    
    
                        image_hand = img[int(start_y + h * 0.2):int(start_y + h * 0.5), (start_x - 15):(end_x + 20)]
                        image_shoe=img[int(start_y+h*0.8):(end_y),start_x:end_x]
                        if image_shoe.shape[0] == 0:
                            continue
                        #    print("image_head",image_head)
                        if image_hand.shape[0] == 0:
                            continue
                        if image_hand.shape[1] == 0:
                            continue
                        if image_head.shape[0] == 0:
                            continue
                        image_body = img[int(start_y + w * 0.6):end_y, start_x:end_x]
                        if image_body.shape[0] == 0:
                            continue
                        label_hat = predict_hat(image_head)  # 0代表true
                        if raw_hat == int(label_hat):
                            count_hat += 1
                        else:
                            raw_hat = int(label_hat)
                            count_hat = 0
    
    
                        if label_hat=='1':
                            if img.shape[0]==0 or img.shape[1]==0:
                                continue
    
                        label_frock = predict_frock(image_body)  # 0代表true
                        if raw_frock == int(label_frock):
                            count_frock += 1
                        else:
                            raw_frock = int(label_frock)
                            count_frock = 0
    
                        if label_frock == '0':
                           label_glove = predict_glove(image_hand)
    
                        if raw_glove == int(label_glove):
                           	count_glove += 1
                        else:
                            raw_glove = int(label_glove)
                            count_glove = 0
    
                        label_shoe=predict_shoe(image_shoe)
                        # if int(label_shoe) == 0:
                        #     label_shoe = predict_edge(image_shoe)
                        if raw_shoe == int(label_shoe):
                            count_shoe += 1
                        else:
                            raw_shoe = int(label_shoe)
                            count_shoe = 0
    
                        label_dic = {"label_hat":label_hat,"label_frock":label_frock,"label_glove":label_glove,"label_shoe":label_shoe,\
                                        "count_hat":count_hat,"count_frock":count_frock, "count_glove":count_glove , "count_shoe":count_shoe
                                     }
                        local_list = [start_x, start_y, end_x, end_y]
                        count_list = [count_hat,count_frock,count_glove,count_shoe]
                        drew_rectangle(img,label_dic,local_list,count_list)
                        if count_faces < 1:
                            faces_label = "unknown"
#                        myThread(label_dic,count_list,count,faces_label,img,camer_id+1).start()
                        executor.submit(myThread, label_dic,count_list,count,faces_label,img,camer_id+1)
                        
                        # upload_log(label_dic,count_list,count,faces_label,img)
                        end = time.time()
                        print("="*50, (end-start)*1000)
                    image_paths = []
                    # cv2.imshow("image", img)
                    ret, jpeg = cv2.imencode('.jpg', img)
                    yield (b'--frame\r\n'
                           b'application/octet-stream: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


