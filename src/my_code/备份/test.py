# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:23:51 2020

@author: 13373
"""

import requests
import glob
import time 


filenames = glob.glob("../logs_img/*.*")
print(filenames)
url = "http://fabx.aidong-ai.com/snapshot/worker"
header = {}


for i, filename in enumerate(filenames):
    time.sleep(5)
    files = {'file': open(filename, 'rb')}
    para = {"Camer_ID": 0,
            "Worker_ID": "unknown",
            "hat": 0,
            "frock": 0,
            "glove": 0,
            "shoe": 0,
            "timestamp": int(round(time.time() * 1000)),
            "file_name": i}
    start = time.time()
    r = requests.post(url, files=files, data=para, headers=header)
    end = time.time()
    print((end-start)*1000)
    print(r)