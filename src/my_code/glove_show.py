# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:30:52 2020

@author: 13373
"""

from openvino.inference_engine import IENetwork, IEPlugin # 导入openvino库
import numpy as np
import cv2, json
import os, time

#注意：手套模型预测结果显示为2，不预测
filename = "./mysetting"
with open(filename, "r") as f:
    load_dict = json.load(f)

    
width = 64
high = 64

location = load_dict["location"]
NCS = False
if not NCS:
    # 用cpu和gpu加速，选择FP32
    basepath1 = load_dict["basefile1"]
    model_xml = os.path.join(basepath1, "model_glove.xml") # xml文件路径
    model_bin = os.path.join(basepath1, "model_glove.bin") # bin文件路径
else:
    # 用神经棒（NCS）加速，选择FP16
    basepath2 = load_dict["basefile2"]   # 
    model_xml = os.path.join(basepath2, "model_glove.xml") # xml文件路径
    model_bin = os.path.join(basepath2, "model_glove.bin") # bin文件路径
    
net = IENetwork(model=model_xml, weights=model_bin)
input_layer = next(iter(net.inputs))
devices = load_dict["devices"]
if devices == 'CPU':
    plugin1 = IEPlugin(device=devices)  # you can choose CPU to call extensions
    plugin1.add_cpu_extension(location)
else:
    plugin1 = IEPlugin(device=devices)
exec_net = plugin1.load(network=net)  # create an executable network 


def predict_glove(image):
    
    image = cv2.resize(image, (width, high))
    img = image.astype("float")/255.0
#    print(img.shape)

    img = np.expand_dims(img, axis=0)    #扩展一维
    img = img.transpose(0,3,1,2) #千万不要用np.reshape
    
    outputs = exec_net.infer(inputs={input_layer: img}) # 推理
    result = outputs["dense_2/Softmax"]
    print(result)
    label = str(np.argmax(result, 1)[0])
    
#    if (result[0]>0.3):
#        labell = "1"
#    else:
#        labell = "0"

    return label
    #return "1"

if __name__ == "__main__":

#    imagePath = "D:/7 input/data/1/ue5.jpg"
    imagePath = "D:/7 input/data/12/r35.jpg"
    
    image = cv2.imread(imagePath)
    for i in range(100):
        start = time.time()
        label =  predict_glove(image)
        end = time.time()
        print("=======", (end-start)*1000)
        
        print(label)
    
