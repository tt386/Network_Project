import os
import shutil

import numpy as np

#number of sites
n = 1000#1000

#Random graph connection probability
p = 1#0.1#0.002

#Probability of being a patch
#PNum_Min = 1
#PNum_Max = n
PNum = [2,3,4,5,6,7,8,16,32,64,128,256,512,600,700,800,900]
#PNum = [2,3,4,5,6,8,10,12,14,16]
#PNum_List = np.linspace(PNum_Min,PNum_Max,PNum_Num)

#Timesteps
T = 100000#1000000

#M fitness
"""
F_Min = 0.1#0.1
F_Max = 1#1
F_step = 0.1
F_List = np.arange(0.1,1,0.1)
"""
F = 0.9#0.9
#Repetas

Repeats = 200

#Savedirname
SaveDirName = "SaveFiles/ConnectionProb_%0.3f_NodeNum_%d_Repeats_%d_Timesteps_%d_F_%0.3f_MaxPNum_%d"%(p,n,Repeats,T,F,PNum[-1])
#SaveDirName = "SaveFiles/ConnectionProb_%0.3f_NodeNum_%d_Repeats_%d_Timesteps_%d_MinF_%0.3f_MaxF_%0.3f_FStep_%0.3f_PNum_%d"%(p,n,Repeats,T,F_Min,F_Max,F_step,PNum)#"SaveFiles/NodeNum_%d_PatchNum_%d_Fitness_%0.3f_Repeats_%d_Timesteps_%d_GeoRad_%0.3f_RandConnectionProb_%0.3f"%(n,PNum,F,Repeats,T,R,p)

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
shutil.copyfile("Script.py", SaveDirName+"/Script.py")

