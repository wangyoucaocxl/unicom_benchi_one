#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import utils3 as utils
from openvino.inference_engine import IENetwork, IEPlugin # 导入openvino库

num_classes     = 80
input_size      = 416

# 搭建网络，调用IENetwork，返回net对象，参数为xml和bin文件的路径
model_xml_CPU = r'yolov3_coco.xml'
model_bin_CPU = r'yolov3_coco.bin'
net = IENetwork(model=model_xml_CPU, weights=model_bin_CPU)

# 定义输入输出
input_blob = next(iter(net.inputs)) # 迭代器
out_blob   = next(iter(net.outputs))
print(input_blob, out_blob)
n, c, h, w = net.inputs[input_blob].shape
print(n,c,h,w)

# 加载设备，调用IEPlugin，返回设备对象，参数为设备名，如CPU、GPU、MYRIAD
plugin = IEPlugin(device='CPU')
# 加载网络，调用设备对象的load方法，返回执行器，参数为网络
exec_net = plugin.load(network=net) 
print('load ok!')

# 获取图像
img_path = r'market.jpg'
original_image = cv2.imread(img_path)

# 图像处理，从(416, 416, 3)到(1, 3, 416, 416)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]
# from (1, 416, 416, 3) to (1, 416, 416, 3)
op_img_data = np.rollaxis(image_data, 3, 1)

# 推理模型，调用执行器的infer方法，返回模型输出，输入为格式化图像
outputs = exec_net.infer(inputs={input_blob: op_img_data}) 

# from (1, 6, 52, 52, 3) to (1, 52, 52, 3, 6)
pred_sbbox = np.rollaxis(outputs['pred_sbbox/concat_2'], 1, 5)
pred_mbbox = np.rollaxis(outputs['pred_mbbox/concat_2'], 1, 5)
pred_lbbox = np.rollaxis(outputs['pred_lbbox/concat_2'], 1, 5)

pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')
image = utils.draw_bbox(original_image, bboxes)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
