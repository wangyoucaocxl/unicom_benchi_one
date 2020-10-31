# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:49:38 2020

@author: 13373
"""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
# 定义一个准备作为线程任务的函数
def action(max):
    my_sum = 0
    for i in range(max):
        print(threading.current_thread().name + '  ' + str(i))
        my_sum += i
    return my_sum
# 创建一个包含4条线程的线程池

        
        
#上传日志的线程
def myThread(lis):  
    global log_list,time_count
    label_dic = lis[0]
    count_list = lis[1]
    count = lis[2]
    faces = lis[3]
    img = lis[4]
    Camer_ID = lis[5]
    
    
 
    time_count += 1
    print("**"*50, time_count)
    label_hat,label_frock,label_glove,label_shoe = label_dic.get("label_hat"),label_dic.get("label_frock"),label_dic.get("label_glove"),label_dic.get("label_shoe")
    if count_list[0] > 50 and count_list[1] > 50 and count_list[2] > 50 and count_list[3] > 50:
        if (str(label_hat) + str(label_frock) + str(label_glove) + str(label_shoe) == "1111") and (time_count % 50 == 1):
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
            requests.post(url, files=files, data=para, headers=header)
            print(para)
        elif str(label_hat) + str(label_frock) + str(label_glove) + str(label_shoe) != "1111" and (time_count % 30 == 2):
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
            requests.post(url, files=files, data=para, headers=header)
            print(para)

with ThreadPoolExecutor(max_workers=10) as pool:
    # 使用线程执行map计算
    # 后面元组有3个元素，因此程序启动3条线程来执行action函数
    results = pool.map(action, (50, 100, 150))
    print('--------------')
    for r in results:
        print(r)
        
with ThreadPoolExecutor(3) as executor:
    for each in seed:
        executor.submit(sayhello,each)
    