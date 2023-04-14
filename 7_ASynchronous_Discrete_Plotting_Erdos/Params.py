import os
import shutil

import numpy as np

#number of sites
n = 1000

#Random graph connection probability
p = 0.002

#Probability of being a patch
PNum_Min = 1
PNum_Max = n
PNum_Num = 10
PNum_List = np.linspace(PNum_Min,PNum_Max,PNum_Num)

#Timesteps
T = 10000

#M fitness
F_Min = 0.1
F_Max = 1
F_step = 0.1
F_List = np.arange(0.1,1,0.1)

#Repetas

Repeats = 100

#Savedirname
SaveDirName = "SaveFiles/ConnectionProb_%0.3f_NodeNum_%d_Repeats_%d_Timesteps_%d_MinF_%0.3f_MaxF_%0.3f_FStep_%0.3f_PNumMin_%d_PNum_Max_%d_PNum_Num_%d"%(p,n,Repeats,T,F_Min,F_Max,F_step,PNum_Min,PNum_Max,PNum_Num)#"SaveFiles/NodeNum_%d_PatchNum_%d_Fitness_%0.3f_Repeats_%d_Timesteps_%d_GeoRad_%0.3f_RandConnectionProb_%0.3f"%(n,PNum,F,Repeats,T,R,p)

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
shutil.copyfile("Script.py", SaveDirName+"/Script.py")

