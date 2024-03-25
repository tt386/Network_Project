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

#number of sites
n = 1000


#ER Stats
#Mean number of connections
C = 4
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
radius = 0.01#np.sqrt(-(1/(np.pi*n)) * np.log(1-0.69))#1/np.sqrt(n)



SingleActive = False

#Prob of Patch 
P = 0.4

#Time taken for sim to run
T = 10000000

#Fitness of the mutant
F = 0.5

#PicTime is the time steps of the snapshorts of the system
PicTime = T/1000






GraphDict = {
        "N":n,
        "Type":Type,
        "P":P,
        "SingleActive":SingleActive,
        "LargestComponent": True#False
        }

if Type == "ER":
    GraphDict["C"] = C
    GraphDict["p"] = p

    SaveDirName= ("SaveFiles/ER_C_%0.3f_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r"%
                (C,n,P,F,T,SingleActive))

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
    GraphDict["radius"] = radius

    SaveDirName= ("SaveFiles/GeoTorus_radius_%0.5f_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r"%
                (radius,n,P,F,T,SingleActive))


if not os.path.isdir("SaveFiles"):
    os.mkdir("SaveFiles")

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
shutil.copyfile("Script.py", SaveDirName+"/Script.py")
