from PIL import Image, ImageDraw, ImageFont
from src.detect_benchi import benchi_detect
import numpy as np
import cv2
from face_dlib.face_detection_new import once_detect



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
def inner_point(point1, points2):
    # filename = "./myguard.json"
    # if not os.path.exists(filename):
        # print("myguard.json is error!")    
    # with open(filename, "r") as f:
        # load_dict = json.load(f) #多边形的顶点坐标
    # points2 = np.array(load_dict["base1"])    

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

def draw_muti(image, point2):
    # filename = "./myguard.json"
    # if not os.path.exists(filename):
        # print("myguard.json is error!")
        
    # with open(filename, "r") as f:
        # load_dict = json.load(f) #多边形的顶点坐标
    # pts = np.array(load_dict["base1"])
    # print(pts)
    pts = np.array(point2)
    cv2.polylines(image,[pts],True,(0,255,255), 2)  #画任意多边形

#clip face img
def face_img(img, out_lis):
    img_h, img_w, img_ch = img.shape
    left = int(out_lis[3]*img_w)
    top = int(out_lis[4]*img_h)
    right = int(out_lis[5]*img_w)
    bottom = int(out_lis[6]*img_h)
    face_img = img[top:bottom, left:right]

    return face_img

def muti_attr(image, mystage):
    out = benchi_detect(image)
    label_dict = {}
    lis = []
    #face_location = []
    label_dict["face"] = ""
    for i in range(len(out)):
        #根据置信度限定不符合要求的attr。
        #========my_setting==============
        control = True     #only once face detection.
        if out[i, 2] > 0.5:
            label = int(out[i,1])
            lis.append(label)
            if label == 2 and control:
                # print("-------------->", out[i])
                clip_face_img = face_img(image, out[i])

                #保存人脸
                # save_file = "face_dlib/face_image"
                # cv2.imwrite(save_file+"/_{}.jpg".format(int(time.time())), clip_face_img)

                face_result = once_detect(clip_face_img)
                if face_result:
                    label_dict["face"] = face_result[0]
                    control = False
   
    myset = sorted(set(lis))
    if mystage == 2:
        for key in myset:
            if key == 5:
                label_dict["coat"] = 1    #deep_coat,去奔驰部署时可更改设置
            elif key == 6:
                label_dict["coat"] = 1    #blue_coat
            elif key == 7:
                label_dict["coat"] = 0    #no_coat
            elif key == 8:
                label_dict["shoes"] = 0   #deep__shoes,去奔驰部署时可更改设置
            elif key == 9:
                label_dict["shoes"] = 1   #star_shoes
            elif key == 10:
                label_dict["shoes"] = 0    #no_shoes,去奔驰部署时可更改设置
    
                
    elif mystage == 1:
        for key in myset:
            if key == 1:
                label_dict["hat"] = 1    #hat
            elif key == 2:
                label_dict["hat"] = 0    #no_hat
            elif key == 3:
                label_dict["gloves"] = 1  #gloves
            elif key == 4:
                label_dict["gloves"] = 0   #no_gloves
            elif key == 5:
                label_dict["coat"] = 1    #deep_coat,去奔驰部署时可更改设置
            elif key == 6:
                label_dict["coat"] = 1    #blue_coat
            elif key == 7:
                label_dict["coat"] = 0    #no_coat
            elif key == 8:
                label_dict["shoes"] = 0   #deep__shoes,去奔驰部署时可更改设置
            elif key == 9:
                label_dict["shoes"] = 1   #star_shoes
            elif key == 10:
                label_dict["shoes"] = 0    #no_shoes,去奔驰部署时可更改设置
    
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

def get_labels(img, pbox, mystage):
    start_x = int(pbox[0])
    start_y = int(pbox[1])
    end_x = int(pbox[2])
    end_y = int(pbox[3]) 
    
    clip_images = img[start_y:end_y, start_x:end_x]
    h, w, c = clip_images.shape
    #根据图像大小限定不符合要求的人形框。
    if h > 96 and w > 48:
        label_dict = muti_attr(clip_images, mystage)
        return label_dict
    return None

def draw_person_attr(show_image, label_dict, pbox, control_color):

#    start_x = int(pbox[0])
    start_y = int(pbox[1])
    end_x = int(pbox[2])
#    end_y = int(pbox[3])
    i_count=0
#    font=cv2.FONT_HERSHEY_COMPLEX
    
    if not control_color:
        return show_image

    #5、在原始图像上画矩形。
    for key in label_dict:
        if key == "hat":
            a = "帽子"
            if label_dict[key] == 1:
                #cv2.rectangle(show_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                b = "合格"
            else:
                #cv2.rectangle(show_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
                b = "不合格"
        elif key == "coat":
            a = "衣服"
            if label_dict[key] == 1:
                #cv2.rectangle(show_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                b = "合格"
            else:
                #cv2.rectangle(show_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2) 
                b = "不合格"
        elif key == "gloves":
            a = "手套"
            if label_dict[key] == 1:
                b = "合格"
            else:
                b = "不合格"
        elif key == "shoes":
            a = "鞋子"
            if label_dict[key] == 1:
                b = "合格"
            else:
                b = "不合格"
        elif key == "face":
            a = "工号"
            b = label_dict[key]
            if b == "-99":
                b = "非法人员"
            text= a + ":" + str(b)
            if label_dict[key] == "" or label_dict[key] == "-99":
                show_image = add_ch_text(show_image,  text, end_x, start_y+i_count, textColor=(255, 0, 0), textSize=30)
                #cv2.putText(show_image,text,(end_x,start_y+i_count),font,0.7,(0,255,0),2)
            else:
                show_image = add_ch_text(show_image, text, end_x, start_y+i_count, textColor=(0, 255, 0), textSize=30)
                #cv2.putText(show_image,text,(end_x,start_y+i_count),font,0.7,(0,0,255),2)
            i_count+=35

        text= a + ":" + str(b)
        # print("text----->", text)

        if label_dict[key] == 1:
            show_image = add_ch_text(show_image,  text, end_x, start_y+i_count, textColor=(0, 255, 0), textSize=30)
            #cv2.putText(show_image,text,(end_x,start_y+i_count),font,0.7,(0,255,0),2)
        elif label_dict[key] == 0:
            show_image = add_ch_text(show_image, text, end_x, start_y+i_count, textColor=(255, 0, 0), textSize=30)
            #cv2.putText(show_image,text,(end_x,start_y+i_count),font,0.7,(0,0,255),2)
        i_count+=35
    return show_image

def save_video(cap, save_file, sz):
    # sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = 20
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    myout = cv2.VideoWriter(save_file,fourcc,fps,sz)
    return myout


