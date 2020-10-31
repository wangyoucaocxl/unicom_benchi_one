# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:13:37 2020

@author: 13373
"""

from openvino.inference_engine import IENetwork, IEPlugin # 导入openvino库
import numpy as np
import cv2
import os

location = 'C:\\Program Files (x86)\\IntelSWTools\\openvino\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll'
NCS = False
if not NCS:
    # 用cpu和gpu加速，选择FP32
    basepath1 = "../model_fp/pedestrian-detection-adas-0002/FP32/"
    model_xml = os.path.join(basepath1, "pedestrian-detection-adas-0002.xml") # xml文件路径
    model_bin = os.path.join(basepath1, "pedestrian-detection-adas-0002.bin") # bin文件路径
else:
    # 用神经棒（NCS）加速，选择FP16
    basepath2 = "./model_fp/pedestrian-detection-adas-0002/FP16/"    # 
    model_xml = os.path.join(basepath2, "pedestrian-detection-adas-0002.xml") # xml文件路径
    model_bin = os.path.join(basepath2, "pedestrian-detection-adas-0002.bin") # bin文件路径

net = IENetwork(model=model_xml, weights=model_bin)
input_layer = next(iter(net.inputs))
devices = 'CPU'
if devices == 'CPU':
    plugin1 = IEPlugin(device=devices)  # you can choose CPU to call extensions
    plugin1.add_cpu_extension(location)
else:
    plugin1 = IEPlugin(device=devices)
exec_net = plugin1.load(network=net)  # create an executable network 


def preson_detect(img):
    
    new_img = img.copy()
    input_img = cv2.resize(new_img, (672, 384), interpolation=cv2.INTER_AREA)
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = input_img[np.newaxis, :]
    print(input_img.shape)
    outputs = exec_net.infer(inputs={input_layer: input_img}) # 推理
    out = outputs["detection_out"]
    out = out.reshape(-1, 7)   #size(200, 7)
    print(out.shape)
    img_h, img_w, img_ch = img.shape
    print(img_h, img_w, img_ch)

    for i in range(len(out)):
        
        if out[i, 2] < 0.8:
            continue
        # print ("coordinate:",x0,y0,x1,y1)
        x0, y0, x1, y1 = out[i, 3:7]
        start_x = int(img_w * x0)
        start_y = int(img_h * y0)
        end_x = int(img_w * x1)
        end_y = int(img_h * y1)
        if start_x<30 or  end_x>580:
            continue
        return [start_x, start_y, end_x, end_y]
    
    return [0,0,0,0]

def draw_rectangle(img, local_list):
    print(local_list)
    cv2.rectangle(img=img, pt1=(local_list[0], local_list[1]), pt2=(local_list[2], local_list[3]), color=(0, 0, 255))

if __name__ == "__main__":
#    cap = cv2.VideoCapture("C:/deploy/logs_img/216.jpg")
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
#    ret, img = cap.read()
#    out = preson_detect(img)
#    print(out)
    
    while True:
        ret, img = cap.read()
        print("="*20, img.shape)
        local_list = preson_detect(img)
        draw_rectangle(img, local_list)
        cv2.imshow("aaa", img)

        if cv2.waitKey(33) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        
    
    
    
