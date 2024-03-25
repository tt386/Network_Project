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

Type = "ER"


Repeats = 100
#number of sites
n = 1000


#ER Stats
#Mean number of connections
minC = 0
maxC = 2
Cnum = 40
Clist = np.linspace(minC,maxC,Cnum)


C = 1.6#10
#Corresponding edge probability
p = C/n

#Small World Stats
#Mean number of connections
k = 4
#rewiring probability
r = 0.3
#number of tries
t = 100

#Geometric Stats
#radius of the sense
minr = 0.#1/np.sqrt(n)
maxr = 0.05#np.sqrt(1/2)
rnum = 40
radiuslist = np.linspace(minr,maxr,rnum)


SingleActive = False

#Prob of Patch 
P = 0.3#0.4

#Time taken for sim to run
T = 10000000

#Fitness of the mutant
F = 0.5

#PicTime is the time steps of the snapshorts of the system
PicTime = T/1000

LargestComponent = False




GraphDict = {
        "N":n,
        "Type":Type,
        "P":P,
        "SingleActive":SingleActive,
        "LargestComponent": False
        }

if Type == "ER":
    GraphDict["C"] = C
    GraphDict["p"] = p

    SaveDirName= ("SaveFiles/ER_minC_%0.3f_maxC_%0.3f_Cnum_%d_NodeNum_%d_ZealotProb_%0.5f_Repeats_%d_Timesteps_%d_SingleActive_%r"%
                (minC,maxC,Cnum,n,P,Repeats,T,SingleActive))

elif Type == "SmallWorld":
    GraphDict["k"] = k
    GraphDict["r"] = r
    GraphDict["t"] = t

    SaveDirName= ("SaveFiles/SW_k_%0.3f_r_%0.3f_t_%0.3f_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r"%
                (k,r,t,n,P,F,T,SingleActive))

elif Type == "Geometric":
    GraphDict["radius"] = radius

    SaveDirName= ("SaveFiles/Geo_radius_%0.5f_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r"%
                (radius,n,P,F,T,SingleActive))

elif Type == "Geometric_Torus":
    GraphDict["radius"] = 0

    SaveDirName= ("SaveFiles/GeoTorus_minr_%0.5f_maxr_%0.5f_rnum_%d_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d_LargestComponent_%r"%
                (minr,maxr,rnum,n,P,F,T,SingleActive,Repeats,LargestComponent))


if not os.path.isdir("SaveFiles"):
    os.mkdir("SaveFiles")

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
shutil.copyfile("Script.py", SaveDirName+"/Script.py")
