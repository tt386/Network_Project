import os
import shutil

import numpy as np

import matplotlib.pyplot as plt

#number of sites
n = 1000#1000

#Mean number of connections
C = 2#10

#Corresponding edge probability
p = C/n


#Number of Patch sites
P = 50/n


#Time taken for sim to run
T = 10000000

#Fitness of the mutant
F = 0.9#8

#PicTime is the time steps of the snapshorts of the system
PicTime = T/100


SaveDirName= ("SaveFiles/C_%d_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d"%
                (C,n,P,F,T))














if not os.path.isdir("SaveFiles"):
    os.mkdir("SaveFiles")

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
shutil.copyfile("Script.py", SaveDirName+"/Script.py")
