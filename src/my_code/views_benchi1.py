#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, Blueprint,Response
from src.person_detect_used import preson_detect
from src.detect_benchi import benchi_detect
#from concurrent.futures import ThreadPoolExecutor
#import threading
import cv2
#import requests
import time, os, json
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw, ImageFont
#import json,time
#url = "http://111.231.83.141:18890/snapshot/worker"
#url = "http://fabx.aidong-ai.com/snapshot/worker"
camer_list = []
#app = Flask(__name__)

#====1mysetting======
app = Flask(__name__, static_folder='../static', static_url_path='/static')
cors = CORS()
cors.init_app(app)
main = Blueprint('main', __name__)
app.register_blueprint(main,url_prefix = "")

time_count = 0
header ={}
log_list="a"

camer_count = 0

@app.route('/')
def index():
    global camer_count
    #camer_count = getcountCameras() 
    camer_count = "1"  #修改让只出现一个窗口
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



#向量叉乘
def determinant(A, B):
    #行列式, 相当于求三角形面积
    #print(v1, v2, v3, v4)
    return (A[0]*B[1]-A[1]*B[0])

def point_mul(A, B):
    #点乘，|A||B|*cos
    return A[0]*B[0]+A[1]*B[1]

def lenth_l(A):
    return (A[0]**2+A[1]**2)**(1/2)

#判断两线段相交坐标
def cup_intersect(a1,a2,b1,b2):
    '''
    #求相交关系，并解算相交坐标
    #构建方程
    x = a1 + tA
    x = b1 + uB
    (a1+tA) x B=(b1+uB) x B
    #由于b x b=0, 可得 
    a1 x B + tAx B=b1 x B 
    #解出参数t 
    t=(b1-a1)x B/(A x B) 
    #同理,解出参数u 
    u=A x (a1-b1)/(A x B)
    #解算交点坐标
    x = a1 + tA
    '''
    
    A = [a2[0]-a1[0], a2[1]-a1[1]]
    B = [b2[0]-b1[0], b2[1]-b1[1]]
    key = determinant(A, B)
    #print("key:", key)
    if key == 0:
        #print("两条线平行")
        return [0, None]
    else:
        #求AxB叉乘，计算行列式
        W = [b1[0]-a1[0], b1[1]-a1[1]]
        Q = [a1[0]-b1[0], a1[1]-b1[1]]
    
        t = determinant(W, B)/determinant(A, B)
        u = determinant(A, Q)/determinant(A, B)
        #print(t, u)

        if 0<=t<=1 and 0<=u<=1:
            result = [a1[0]+A[0]*t, a1[1]+A[1]*t]
            #print("交点为：", result)
            return [1, result]
        else:
            #print("线段不相交")
            return [0, None]

#判断一个多边形的点是否在另一个多边形的内部
def inner_point(point1):
    filename = "./myguard.json"
    if not os.path.exists(filename):
        print("myguard.json is error!")    
    with open(filename, "r") as f:
        load_dict = json.load(f) #多边形的顶点坐标
    points2 = np.array(load_dict["base1"])    

    a1 = [0, 0]
    a2 = point1
    num = 0
    result_point = []
    for j in range(len(points2)):
        if j < len(points2)-1:
            b1 = points2[j]
            b2 = points2[j+1]
        elif j == len(points2)-1:
            b1 = points2[j]
            b2 = points2[0]                
        else:
            print("Input points err!")
        result = cup_intersect(a1,a2,b1,b2)
        #print(result)
        if result[0] == 1:
            if result[1] not in result_point:
                result_point.append(result[1])
                num += 1
    if num%2 == 1:
        return True
    else:
        return False

def draw_muti(image):
    filename = "./myguard.json"
    if not os.path.exists(filename):
        print("myguard.json is error!")
        
    with open(filename, "r") as f:
        load_dict = json.load(f) #多边形的顶点坐标
    pts = np.array(load_dict["base1"])
    cv2.polylines(image,[pts],True,(0,255,255), 2)  #画任意多边形

def muti_attr(image):
    out = benchi_detect(image)
    label_dict = {}
    lis = []
    for i in range(len(out)):
        #人形框限定条件，根据置信度限定不符合要求的人形框。
        if out[i, 2] > 0.5:
            label = int(out[i,1])
            lis.append(label)
        
    myset = sorted(set(lis))
    
    for key in myset:
        if key == 1:
            label_dict["hat"] = "Yes"
        elif key == 2:
            label_dict["hat"] = "No"
        elif key == 3:
            label_dict["coat"] = "Yes"
        elif key == 4:
            label_dict["coat"] = "No"
        elif key == 5:
            label_dict["gloves"] = "Yes"
        elif key == 6:
            label_dict["gloves"] = "No"
        elif key == 7:
            label_dict["shoes"] = "Yes"
        elif key == 8:
            label_dict["shoes"] = "No"
    
    return label_dict

def add_ch_text(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否Opencv2图片类型
       img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simhei.ttf", textSize, encoding="utf-8")
    # 绘制文本e
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回Opencv格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  


def get_detect(id_video):
    
    my_num = 0
    num = 0
    video_reader = cv2.VideoCapture(0)
    #video_reader = cv2.VideoCapture("rtsp://admin:ad123456@192.168.199.200/Streaming/Channels/1")
    #video_reader.set(cv2.CAP_PROP_FPS, 20)
    video_reader.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    video_reader.set(3, 1280)  
    video_reader.set(4, 720)
    video_reader.set(cv2.CAP_PROP_FPS, 25)  
    
    while True:
        start_time = time.time()
        num += 1
        print("======", num)
        
        ret1, img = video_reader.read()
        
        if not ret1:
            continue

        if type(img) == type(None):
            continue
        #frame = cv2.imread(filename)
        #1、读取中文路径
        #img = cv2.imdecode(np.fromfile(filename,dtype=np.uint8), cv2.IMREAD_COLOR)
        print(img.shape)
        #2、防止裁剪或推理时把画的框裁剪上
        show_image = img.copy()
        if num%2==1:
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
                            if key == "hat":
                                a = "帽子"
                                if label_dict[key] == "Yes":
                                    b = "合格"
                                else:
                                    b = "不合格"
                            elif key == "coat":
                                a = "衣服"
                                if label_dict[key] == "Yes":
                                    b = "合格"
                                else:
                                    b = "不合格"
                            elif key == "gloves":
                                a = "手套"
                                if label_dict[key] == "Yes":
                                    b = "合格"
                                else:
                                    b = "不合格"
                            elif key == "shoes":
                                a = "鞋子"
                                if label_dict[key] == "Yes":
                                    b = "合格"
                                else:
                                    b = "不合格"
                            text= a + ":" + str(b)
                            print(text)
                            if label_dict[key] == "Yes":
                                show_image = add_ch_text(show_image,  text, end_x, start_y+i_count, textColor=(0, 255, 0), textSize=30)
                                #cv2.putText(show_image,text,(end_x,start_y+i_count),font,0.7,(0,255,0),2)
                            elif label_dict[key] == "No":
                                show_image = add_ch_text(show_image, text, end_x, start_y+i_count, textColor=(255, 0, 0), textSize=30)
                                #cv2.putText(show_image,text,(end_x,start_y+i_count),font,0.7,(0,0,255),2)
                            i_count+=35
            
            draw_muti(show_image)        
            ret2, jpeg = cv2.imencode('.jpg', show_image)
            yield (b'--frame\r\n'
                   b'application/octet-stream: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            time.sleep(0.01)

        end_time = time.time()
        mytime = (end_time-start_time)*1000
        print(mytime)
        
#        cv2.namedWindow("image{}".format("benchi"), cv2.WINDOW_NORMAL)
#        cv2.imshow("image{}".format("benchi"), show_image)
#        key = cv2.waitKey(33)
#        if key == 27:
#            break
#        
#    cv2.destroyAllWindows()

  
if __name__ == "__main__":
    get_detect(0)

'''
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
'''
