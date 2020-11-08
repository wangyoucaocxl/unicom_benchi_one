
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:13:37 2020

@author: 13373
"""

from openvino.inference_engine import IENetwork, IEPlugin 
import numpy as np
import cv2, json
import os

filename = "./mysetting"
with open(filename, "r") as f:
    load_dict = json.load(f)


location = load_dict["location"]
NCS = False
if not NCS:
    # 用cpu和gpu加速，选择FP32
    basepath1 = load_dict["basefile1"]
    model_xml = os.path.join(basepath1, "mul_stage_model.xml") # xml文件路径
    model_bin = os.path.join(basepath1, "mul_stage_model.bin") # bin文件路径
else:
    # 用神经棒（NCS）加速，选择FP16
    basepath2 = load_dict["basefile2"]    # 
    model_xml = os.path.join(basepath2, "mul_stage_model.xml") # xml文件路径
    model_bin = os.path.join(basepath2, "mul_stage_model.bin") # bin文件路径


net = IENetwork(model=model_xml, weights=model_bin)
input_layer = next(iter(net.inputs))
devices = load_dict["devices"]
if devices == 'CPU':
    plugin1 = IEPlugin(device=devices)  # you can choose CPU to call extensions
    # plugin1.add_cpu_extension(location)   #2020.3版本不需要cpu模块
else:
    plugin1 = IEPlugin(device=devices)
exec_net = plugin1.load(network=net)  # create an executable network 


def mul_stage_model(img):
    new_img = img.copy()
    input_img = cv2.resize(new_img, (300, 300), interpolation=cv2.INTER_AREA)
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = input_img[np.newaxis, :]
#    print(input_img.shape)
    outputs = exec_net.infer(inputs={input_layer: input_img}) # 推理
    out = outputs["DetectionOutput"]
    print("*******", out.shape)
    out = out.reshape(-1, 7)   #size(200, 7)
    return out

if __name__ == "__main__":
#    cap = cv2.VideoCapture("C:/deploy/logs_img/216.jpg")
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
#    ret, img = cap.read()
#    out = preson_detect(img)
#    print(out)
    
    while True:
        ret, img = cap.read()
#        print("="*20, img.shape)
        out = mul_stage_model(img)
        print(out)
        cv2.imshow("aaa", img)

        if cv2.waitKey(33) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        
    
    
    
