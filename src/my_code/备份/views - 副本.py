#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, Blueprint,Response
from src.shoe_show import predict_shoe
from src.hat_show import predict_hat
from src.frock_show import predict_frock
from src.glove_show import predict_glove
import threading

import numpy as np
import cv2
import requests
import os, time
#import json,time
#url = "http://111.231.83.141:18890/snapshot/worker"
url = "http://fabx.aidong-ai.com/snapshot/worker"
from sklearn.externals import joblib
camer_list = []
app = Flask(__name__)
main = Blueprint('main', __name__)
app.register_blueprint(main,url_prefix = "")


time_count = 0
header ={}
log_list="a"
from openvino.inference_engine import IENetwork, IEPlugin
# intel_models = "/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/intel_models/pedestrian-detection-adas-0002/FP32"
#intel_models = "C:\\Intel\\computer_vision_sdk_2018.5.456\\deployment_tools\\intel_models\\pedestrian-detection-adas-0002\\FP32"
intel_models = "C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\intel_models\\pedestrian-detection-adas-0002\\FP32"
xml_path = os.path.join(intel_models, "pedestrian-detection-adas-0002.xml")
bin_path = os.path.join(intel_models, "pedestrian-detection-adas-0002.bin")
# location = '/opt/intel/computer_vision_sdk/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so'
location = 'C:\\Program Files (x86)\\IntelSWTools\\openvino\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll'
net = IENetwork(model=xml_path, weights=bin_path)
input_layer = next(iter(net.inputs))
n, c, h, w = net.inputs[input_layer].shape
#    print("network shape:",n,c,h,w)
# net.batch_size = 1
net.reshape({input_layer: (n, c, 320, 544)})
# plugin = IEPlugin(device="GPU") #you can choose CPU to call extensions
plugin = IEPlugin(device="CPU")  # you can choose CPU to call extensions
plugin.add_cpu_extension(location)
exec_net = plugin.load(network=net)  # create an executable network


camer_count = 0
norm_size = 32

clf = joblib.load('f_model.m')

#上传日志的线程
class myThread(threading.Thread):  
    def __init__(self, label_dic,count_list,count,faces,img,Camer_ID):
        threading.Thread.__init__(self)
        self.label_dic = label_dic
        self.count_list = count_list
        self.count = count
        self.faces = faces
        self.img = img
        self.Camer_ID = Camer_ID

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        global log_list,time_count
        time_count += 1
        print("**"*50, time_count)
        label_hat,label_frock,label_glove,label_shoe = self.label_dic.get("label_hat"),self.label_dic.get("label_frock"),self.label_dic.get("label_glove"),self.label_dic.get("label_shoe")
        #if self.count_list[0] > 20 and self.count_list[1] > 20 and self.count_list[2] > 30 and self.count_list[3] > 10 and log_list != str(label_hat) + str(label_frock) + str(label_glove) + str(label_shoe):
        if self.count_list[0] > 50 and self.count_list[1] > 50 and self.count_list[2] > 50 and self.count_list[3] > 50:
        # if self.count_list[0] > 0 and self.count_list[1] > 0 and self.count_list[2] > 0 and self.count_list[3] > 0:
            if (str(label_hat) + str(label_frock) + str(label_glove) + str(label_shoe) == "1111") and (time_count % 50 == 1):
                cv2.imwrite('./logs_img/%s.jpg' % (self.count), self.img)
                files = {'file': open('./logs_img/%s.jpg' % (self.count), 'rb')}
                lh = 0 if label_hat == "1" else 1
                lf = 0 if label_frock == "1" else 1
                lg = 0 if label_glove == "1" else 1
                ls = 0 if label_shoe == "1" else 1
                para = {"Camer_ID": self.Camer_ID,
                        "Worker_ID": self.faces,
                        "hat": lh,
                        "frock": lf,
                        "glove": lg,
                        "shoe": ls,
                        "timestamp": int(round(time.time() * 1000)),
                        "file_name": self.count}
                r = requests.post(url, files=files, data=para, headers=header)
                print(para)
            elif str(label_hat) + str(label_frock) + str(label_glove) + str(label_shoe) != "1111" and (time_count % 30 == 2):
                cv2.imwrite('./logs_img/%s.jpg' % (self.count), self.img)
                files = {'file': open('./logs_img/%s.jpg' % (self.count), 'rb')}
                lh = 0 if label_hat == "1" else 1
                lf = 0 if label_frock == "1" else 1
                lg = 0 if label_glove == "1" else 1
                ls = 0 if label_shoe == "1" else 1
                para = {"Camer_ID": self.Camer_ID,
                        "Worker_ID": self.faces,
                        "hat": lh,
                        "frock": lf,
                        "glove": lg,
                        "shoe": ls,
                        "timestamp": int(round(time.time() * 1000)),
                        "file_name": self.count}
                r = requests.post(url, files=files, data=para, headers=header)
                print(para)

        #保证如果全部都没穿的情况的多次低频上传
#        if log_list == "1111" and time_count > 20:
#            log_list = "1"
#            time_count = 0
#        if log_list != "1111":
#            time_count = 0


@app.route('/')
def index():
    global camer_count
    camer_count = getcountCameras()
    return render_template('index.html')

def clearCapture(capture): 
    capture.release() 
    cv2.destroyAllWindows() 
  
def getcountCameras(): 
    global camer_list
    n = 0 

    for i in range(2): 
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
        number = camer_list[1]
    return Response(get_detect(number),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#@app.route('/detect_video2')
#def detect_video2():
#    global camer_list
#    # get_detect()
#    number = 0
#    if len(camer_list)>=2:
#        number = camer_list[1]
#    return Response(get_detect(number),
#                    mimetype='multipart/x-mixed-replace; boundary=frame')
#
#@app.route('/detect_video3')
#def detect_video3():
#    global camer_list
#    # get_detect()
#    number = 0
#    if len(camer_list)>=3:
#        number = camer_list[2]
#    return Response(get_detect(number),
#                    mimetype='multipart/x-mixed-replace; boundary=frame')
#
#@app.route('/detect_video4')
#def detect_video4():
#    global camer_list
#    # get_detect()
#    number = 0
#    if len(camer_list)>=4:
#        number = camer_list[3]
#    return Response(get_detect(number),
#                    mimetype='multipart/x-mixed-replace; boundary=frame')





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

#def upload_log(label_dic,count_list,count,faces,img):
#    global log_list,time_count
#    time_count += 1
#    label_hat,label_frock,label_glove,label_shoe = label_dic.get("label_hat"),label_dic.get("label_frock"),label_dic.get("label_glove"),label_dic.get("label_shoe")
#    if count_list[0] > 30 and count_list[1] > 30 and count_list[2] > 50 and count_list[3] > 10 and log_list != str(label_hat) + str(label_frock) + str(label_glove) + str(label_shoe):
#
#    #if count_list[0] >= 0 and count_list[1] >= 0 and count_list[2] >= 0 and count_list[3] >= 0:
#
#        log_list = str(label_hat) + str(label_frock) + str(label_glove) + str(label_shoe)
#        cv2.imwrite('./logs_img/%s.jpg' % (count), img)
#        files = {'file': open('./logs_img/%s.jpg' % (count), 'rb')}
#        lh = 0 if label_hat == "1" else 1
#        lf = 0 if label_frock == "1" else 1
#        lg = 0 if label_glove == "1" else 1
#        ls = 0 if label_shoe == "1" else 1
#        para = {"Camer_ID": '001',
#                "Worker_ID": faces,
#                "hat": lh,
#                "frock": lf,
#                "glove": lg,
#                "shoe": ls,
#                "timestamp": int(round(time.time() * 1000)),
#                "file_name": count}
#        r = requests.post(url, files=files, data=para, headers=header)
#        print(para)
#    if log_list == "1111" and time_count>40:
#        log_list = "1"
#        time_count = 0
#    elif log_list !="1111":
#        time_count = 0

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

    while (True):
        # while(True):
        ret, image = video_reader.read()
        if ret == True:
            image_paths += [image]
            count += 1
            # cv2.imshow('Image',image)
        if (len(image_paths) == batch_size) or (ret == False and len(image_paths) > 0):
            for img in image_paths:
                

                new_img = img.copy()
                input_img = cv2.resize(new_img, (544, 320), interpolation=cv2.INTER_AREA)
                input_img = np.transpose(input_img, (2, 0, 1))
                input_img = input_img[np.newaxis, :]
                # print(input_img.shape)
                # step5. VINO related. call input request.
                cur_req = exec_net.requests[0]
                input_blob = next(iter(net.inputs))
                # step6. VINO rateld. Just call the inference. Sooooo simple!
                exec_net.requests[0].infer({input_blob: input_img})

                out = cur_req.outputs["detection_out"]
                out = out.reshape(-1, 7)
                #                    index = out.argmax(axis=2)
                #                    print(index[:,0,2].shape)
                img_h, img_w, img_ch = img.shape
                #                    print(out[0,0,index[:,0,2],3:7].shape)
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
                    myThread(label_dic,count_list,count,faces_label,img,camer_id+1).start()
                    
                    # upload_log(label_dic,count_list,count,faces_label,img)
                    end = time.time()
                    print("="*50, (end-start)*1000)
                image_paths = []
                # cv2.imshow("image", img)
                ret, jpeg = cv2.imencode('.jpg', img)
                yield (b'--frame\r\n'
                       b'application/octet-stream: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


#def face_det(img):
#    new_img=img.copy()
#    crop_img=new_img[:,:,:]
#    input_img = cv2.resize(crop_img,(300,300), interpolation=cv2.INTER_AREA)
#    input_img = np.transpose(input_img,(2,0,1))
#    input_img = input_img[np.newaxis,:]
#    if input_img.shape[0]==0 or input_img.shape[1]==0:
#        return 0,0,0,0
#    req=face_net.requests[0]
#    input_blob2 = next(iter(net2.inputs))
#    face_net.requests[0].infer({input_blob2:input_img})
#    out=req.outputs['detection_out']
##                print(out.shape)
##    print(out.shape)
#    out=out.reshape(-1,7)
##                print(out[2,2])
##                print(len(out))
#    img_h,img_w,img_ch = img.shape
#    start_x,start_y,end_x,end_y=0,0,300,300
#    for j in range(out.shape[0]):
#        if out[j,2]<0.8:
#            continue
#        x0,y0,x1,y1 =  out[j,3:7]
#        start_x = int(img_w*x0)
#        start_y = int(img_h*y0)
#        end_x = int(img_w*x1)
#        end_y = int(img_h*y1)
#        return start_x,start_y,end_x,end_y

# def predict_hat(image):
#     global model_hat
#     # load the trained convolutional neural network
#     #    print("[INFO] loading network...")
#     # load the image
#     # image = cv2.imread(img)
#     orig = image.copy()
#     #    print(orig.shape)
#     # pre-process the image for classification
#     # print(type(orig))
#     image = cv2.resize(orig, (norm_size, norm_size))
#     image = image.astype("float") / 255.0
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)

#     with graph.as_default():
#        result = model_hat.predict(image)[0]
#     #print (result.shape)
#     proba = np.max(result)
#     label = str(np.where(result == proba)[0][0])

#     # classify the input image
#     #with graph.as_default():
#     #    result = model_hat.predict(image)
#     #return str(np.argmax(result[2]))
#     return label

#def predict_glove(image):
#    global model_glove
#    # load the trained convolutional neural network
#    #    print("[INFO] loading network...")
#    # load the image
#    # image = cv2.imread(img)
#    orig = image.copy()
#
#    # pre-process the image for classification
#    # print(type(orig))
#    image = cv2.resize(orig, (norm_size, norm_size))
#    image = image.astype("float") / 255.0
#    image = img_to_array(image)
#    image = np.expand_dims(image, axis=0)
#
#    # classify the input image
#    with graph.as_default():
#        result = model_glove.predict(image)[0]
#    # print (result.shape)
#    proba = np.max(result)
#    label = str(np.where(result == proba)[0][0])
#    #    labels = "{}: {:.2f}%".format(label, proba * 100)
#    #    print(labels)
#    return label



#def predict_frock(image):
#    global model_frock
#    # load the trained convolutional neural network
#    #    print("[INFO] loading network...")
#    # load the image
#    # image = cv2.imread(args["image"])
#    orig = image.copy()
#
#    # pre-process the image for classification
#    # print(orig)
#    image = cv2.resize(orig, (norm_size, norm_size))
#    image = image.astype("float") / 255.0
#    image = img_to_array(image)
#    image = np.expand_dims(image, axis=0)
#
#    # classify the input image
#    with graph.as_default():
#        result = model_frock.predict(image)[0]
#    # print (result.shape)
#    proba = np.max(result)
#    label = str(np.where(result == proba)[0][0])
#    #    labels = "{}: {:.2f}%".format(label, proba * 100)
#    #    print(labels)
#    return label

#def predict_edge(image):
#    global model_edge
#    # load the trained convolutional neural network
#    #    print("[INFO] loading network...")
#    # load the image
#    # image = cv2.imread(img)
#    orig = image.copy()
#
#    # pre-process the image for classification
#    # print(type(orig))
#    image=cv2.resize(orig,(350,350))
#    img_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
#    
#    img=cv2.Canny(img_gray,30,80)
#    img = cv2.resize(img, (128, 128))
#    image = img.astype("float") / 255.0
#    image = img_to_array(image)
#    image = np.expand_dims(image, axis=0)
#
#    # classify the input image
#    with graph.as_default():
#        result = model_edge.predict(image)[0]
#    # print (result.shape)
#    proba = np.max(result)
#    label = str(np.where(result == proba)[0][0])
#    #    labels = "{}: {:.2f}%".format(label, proba * 100)
#    #    print(labels)
#    return label 