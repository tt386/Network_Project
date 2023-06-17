import os
import shutil

import numpy as np

import matplotlib.pyplot as plt




#number of sites
n = 1000#1000

#Mean number of connections
CList = np.linspace(1,10,50)#np.arange(2,41)#np.linspace(2,n/10,100).astype(int)#[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

"""
#Corresponding edge probability
p = C/n
"""

#Prob of Patch sites
P = 0.05#0.025#50/n


#Time taken for sim to run
T = 2000000#00

#Fitness of the mutant
F = 0.9#9#8


Repeats = 20

#PicTime is the time steps of the snapshorts of the system
PicTime = T/100



SaveDirName= ("SaveFiles/CMin_%d_Cmax_%d_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_Repeats_%d"%
                (CList[0],CList[-1],n,P,F,T,Repeats))



if not os.path.isdir("SaveFiles"):
    os.mkdir("SaveFiles")

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
shutil.copyfile("Script.py", SaveDirName+"/Script.py")
