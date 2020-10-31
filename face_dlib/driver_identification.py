import face_recognition
import os
import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
#opencv zh_ch
def add_ch_text(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否Opencv2图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本e
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回Opencv格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  

# ----------------------------load_face_dataset---------------------
class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        print("images",images)
        image_paths = [os.path.join(facedir,img) for img in images]
        print("image_paths", image_paths)
    return image_paths

def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    #class name
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset
#dataset
#name,imagepaths(type=list) ,such as (xiaoming,及其文件夹下的所有图片路径)
# get face_dataset __________________________________________


#dataset--------------------------------->>>encoding
def dataset_encoding(dataset):
    known_face_names=[]
    known_face_encodings=[]
    #遍历文件夹名
    for i in range(len(dataset)):
        #遍历文件夹下的图片
        for j in range(len(dataset[i].image_paths)):
            image=face_recognition.load_image_file(dataset[i].image_paths[j])
            #face_data too small to clip
            if(len(face_recognition.face_encodings(image))>=1):
                encoding=face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(dataset[i].name)

    return known_face_encodings,known_face_names


#detection_face return-------------------face_names
def detection_face(rgb_small_frame,face_locations,known_face_encodings,known_face_names):
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        #面与面之间的距离应视为匹配。越低越严格，默认tolerance=0.6，0.6是典型的最佳性能
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.55)
        name = "Unknown"
        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    return face_names


#两个矩形的相交面积：判断驾驶员前倾、后仰，追踪，是否存在身份更换
'''
input: rect1, rect2, 均为list,其分别为
yt(top),xr(right),yb(bottom),xl(left)
face_locations[0]: top, right, bottom, left
'''
def calc_area(rect1, rect2):
    yt1, xr1, yb1, xl1 =rect1
    yt2, xr2, yb2, xl2 =rect2

    #forward_backward
    last_area = abs(xl1 - xr1) * abs(yb1 - yt1)
    area = abs(xl2 - xr2) * abs(yb2 - yt2)
    #mean_area = (last_area + area) / 2
    forward_backward = (area - last_area)/last_area
    #cover_area
    xmax = max(xr1, xr2)
    ymax = max(yb1, yb2)
    xmin = min(xl1, xl2)
    ymin = min(yt1, yt2)
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        cover_square=0
    else:
        cover_square = width * height
    return forward_backward,cover_square
# # Display the results
# def display_rec_name(frame, face_locations, face_names):
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         #Scale back up face locations since the frame we detected in was scaled to 1/4 size

#         #top *= 4
#         #right *= 4
#         #bottom *= 4
#         #left *= 4
       
#         # Draw a box around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         # Draw a label with a name below the face
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


# Display the results
def display_rec_name(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        #Scale back up face locations since the frame we detected in was scaled to 1/4 size

        #top *= 4
        #right *= 4
        #bottom *= 4
        #left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # cv2.putText(frame, "diver:"+name, (70, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0),
        #             2, cv2.LINE_AA)
        


# two points of distance
def distance_two(point1,point2):
    distance=math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return distance
#dilib location --->face recognition
def dlib_location_change(faces):
    face_locations=[]
    face_num=0
    dist_list=[]
    mid_rec=[]
    mid_point=[]
    area_list=[]
    for i, d in enumerate(faces): 
        list_location=[]
        y1 = d.top() if d.top() > 0 else 0
        y2 = d.bottom() if d.bottom() > 0 else 0
        x1 = d.left() if d.left() > 0 else 0
        x2 = d.right() if d.right() > 0 else 0
        list_location.append(y1)
        list_location.append(x2)
        list_location.append(y2)
        list_location.append(x1)
        tuple_location=tuple(list_location)
        # if (face_num==0):
        face_locations.append(tuple_location)
        face_num +=1
        #mid point
        mid_rec.append((x1+x2)/2)
        mid_rec.append((y1+y2)/2)
        mid_point.append(320)
        mid_point.append(240)
        dist_list.append(distance_two(mid_rec,mid_point))
        area_list.append((x2-x1)*(y2-y1))
    #最大面积的人脸框
    # if(len(area_list)>0):
    #     max_index=area_list.index(max(dist_list))
    #     temp=face_locations[max_index]
    #     face_locations=[]
    #     face_locations.append(temp)
    if (len(dist_list)>0):
        min_index=dist_list.index(min(dist_list))
        temp=face_locations[min_index]
        face_locations=[]
        face_locations.append(temp)

    return face_locations


# index
def find_index(time_list, d_time):
    index = None
    max_index = len(time_list) - 1
    for i in range(max_index - 1):
        tmp_d_time = time_list[max_index] - time_list[max_index - 1 - i]
        if (tmp_d_time >= d_time):
            index = max_index - 1 - i
            break
    return index


