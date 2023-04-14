from Params import *


import subprocess
import time

import os,shutil

starttime = time.time()

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Directory")

shutil.copy("Params.py",SaveDirName)

plist = []

for F in F_List:
    for N in PNum_List:
        p=subprocess.Popen(['nice','-n','19','python','Script.py','-F',str(F),'-N',str(N)])
        plist.append(p)

for p in plist:
    p.wait()

endtime = time.time()


print("Time taken:",endtime-starttime)

