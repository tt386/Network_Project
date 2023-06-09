import os
import shutil

import numpy as np

import matplotlib.pyplot as plt




#number of sites
n = 1000#1000

#Mean number of connections
#CList = np.arange(2,41)#np.linspace(2,n/10,100).astype(int)#[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

C = 2

#Corresponding edge probability
p = C/n

#Prob of Patch sites
P_Min = 1/n
P_Max = 1
P_Num = 10
P_List = np.linspace(P_Min,P_Max,P_Num)#0.08#0.025#50/n
P_List = P_List[:-1]

#Time taken for sim to run
T = 2000000#00

#Fitness of the mutant
F_Min = 0.1
F_Max = 1
F_step = 0.1
F_List = np.arange(F_Min,F_Max,F_step)


Repeats = 40

#PicTime is the time steps of the snapshorts of the system
PicTime = T/100



SaveDirName= ("SaveFiles/C_%0.3f_NodeNum_%d_MinZProb_%0.5f_MaxZProb_%0.5f_ZProbNum_%d_MinF_%0.3f_MaxF_%0.3f_Fstep_%0.3f_Timesteps_%d_Repeats_%d"%
                (C,n,P_Min,P_Max,P_Num,F_Min,F_Max,F_step,T,Repeats))



if not os.path.isdir("SaveFiles"):
    os.mkdir("SaveFiles")

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
shutil.copyfile("Script.py", SaveDirName+"/Script.py")
