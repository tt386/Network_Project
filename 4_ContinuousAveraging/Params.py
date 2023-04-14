import os
import shutil


#number of sites
n = 100#50

#Geographical network connection radius
R = 0.3

#Random graph connection probability
p = 0.2

#Probability of being a patch
P_Num = 1

#Timesteps
T = 200

#M fitness
F = 0.99

#Repetas
Repeats = 100

#Savedirname
SaveDirName = "SaveFiles/NodeNum_%d_PatchNum_%0.3f_Fitness_%0.3f_Repeats_%d_Timesteps_%d_GeoRad_%0.3f_RandConnectionProb_%0.3f"%(n,P_Num,F,Repeats,T,R,p)

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
