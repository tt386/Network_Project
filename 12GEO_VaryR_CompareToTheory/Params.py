import os
import shutil

import numpy as np

import matplotlib.pyplot as plt

"""
Types:
    ER
    SmallWorld
    Geometric
"""

Type = "Geometric"

#number of sites
n = 1000#1000


#ER Stats
#Mean number of connections
C = 1.6#10
#Corresponding edge probability
p = C/n

#Small World Stats
#Mean number of connections
k = 2
#rewiring probability
r = 0.2
#number of tries
t = 100

#Geometric Stats
#radius of the sense
minr = 1/np.sqrt(n)
maxr = np.sqrt(2)
rnum = 40
radiuslist = np.linspace(minr,maxr,rnum)


#General stats
Repeats = 20
#Whether all the patches start infected or not
SingleActive = False
#Prob of Patch 
P = 0.05
#Time taken for sim to run
T = 2000000
#Fitness of the mutant
F = 0.9
#PicTime is the time steps of the snapshorts of the system
PicTime = T/1000









GraphDict = {
        "N":n,
        "Type":Type,
        "P":P,
        "SingleActive":SingleActive
        }

if Type == "ER":
    GraphDict["C"] = C
    GraphDict["p"] = p

    SaveDirName= ("SaveFiles/ER_C_%0.3f_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d"%
                (C,n,P,F,T,SingleActive,Repeats))

elif Type == "SmallWorld":
    GraphDict["k"] = k
    GraphDict["r"] = r
    GraphDict["t"] = t

    SaveDirName= ("SaveFiles/SW_k_%0.3f_r_%0.3f_t_%0.3f_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d"%
                (k,r,t,n,P,F,T,SingleActive,Repeats))

elif Type == "Geometric":
    GraphDict["radius"] = 0

    SaveDirName= ("SaveFiles/Geo_minr_%0.5f_maxr_%0.5f_rnum_%d_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d"%
                (minr,maxr,rnum,n,P,F,T,SingleActive,Repeats))


if not os.path.isdir("SaveFiles"):
    os.mkdir("SaveFiles")

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
shutil.copyfile("Script.py", SaveDirName+"/Script.py")

