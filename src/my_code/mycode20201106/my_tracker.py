# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 07:47:49 2020

@author: 13373
"""


def iou_cal(lis1, lis2):
    max_x = max(lis2[2], lis2[0], lis1[2], lis1[0])
    min_x = min(lis2[2], lis2[0], lis1[2], lis1[0])
    max_y = max(lis2[3], lis2[1], lis1[3], lis1[1])
    min_y = min(lis2[3], lis2[1], lis1[3], lis1[1])
    l1_x = abs(lis1[2] - lis1[0])
    l1_y = abs(lis1[3] - lis1[1])

    l2_x = abs(lis2[2] - lis2[0])
    l2_y = abs(lis2[3] - lis2[1])

    a = l1_x + l2_x - (max_x-min_x)
    b = l2_y + l1_y - (max_y-min_y)


    my_iou = a*b/(l1_x*l1_y + l2_x*l2_y-a*b)
    return my_iou

def new_loc(lis1, lis2):
    center_x = int((lis2[0] + lis2[2])/2)
    center_y = int((lis2[1] + lis2[3])/2)
    
    mid_lis = [int((x1+x2)/2) for x1, x2 in zip(lis1, lis2)]
    mid_x = int((mid_lis[0] + mid_lis[2])/2)
    mid_y = int((mid_lis[1] + mid_lis[3])/2)

    x_shift = center_x - mid_x
    y_shift = center_y - mid_y

    return [mid_lis[0]+x_shift, mid_lis[1]+y_shift, mid_lis[2]+x_shift, mid_lis[3]+y_shift]


def my_tracker():
    boxes = [[4,5,6,7],[4,5,6,7]]

    fbox = []
    bbox = []
    person_dit = {}
    threshold_iou = 0.8
    while True:
        if not fbox:
            fbox = boxes
            init_len = len(fbox)
            for i_d in range(1, init_len+1):
                person_dit[str(i_d)] = fbox
            continue
        else:
            bbox = boxes

        person_new = person_dit.copy()
        #删除上一贞出现这贞消失的数据，并更新一直存在的坐标
        for key in person_new:
            for one in bbox:
                my_iou = iou_cal(person_new[key], one)
                if my_iou > threshold_iou:
                    person_new[key] = new_loc(person_new[key], one)
                    break
            else:
                del person_new[key]
        
        #添加这贞新增而上一贞不存在的数据
        for one in bbox:
            for key in person_dit:
                my_iou = iou_cal(person_dit[key], one)
                if my_iou > threshold_iou:
                    break
            else:
                init_len += 1
                person_new[str(init_len)] = one    #等于the value of this frame 
        fbox = bbox
        
        if not person_dit:
            continue


def main():
    lis1 = [1,1,3,3]
    lis2 = [2,2,4,4]
    print(max(lis2[2], lis2[0], lis1[2], lis1[0]))

    result = iou_cal(lis1, lis2) 
    print(result)
    
if __name__ == '__main__':
    main()
        

            
            
            
            
            
            
            
            
            