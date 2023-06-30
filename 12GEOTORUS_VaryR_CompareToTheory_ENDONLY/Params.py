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

Type = "Geometric_Torus"

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
minr = 0.#1/np.sqrt(n)
maxr = 0.1#05#np.sqrt(1/2)
rnum = 40
radiuslist = np.linspace(minr,maxr,rnum)


#General stats
Repeats = 20
#Whether all the patches start infected or not
SingleActive = False
#Prob of Patch 
P = 0.4
#Time taken for sim to run
T = 1000000#100000000
#Fitness of the mutant
F = 0.5#25
#Whether I just use the largest component
LargestComponent = False

#PicTime is the time steps of the snapshorts of the system
PicTime = T/1000

#Number of points sampled at the end
DataPoints = int(1e6)







GraphDict = {
        "N":n,
        "Type":Type,
        "P":P,
        "SingleActive":SingleActive,
        "LargestComponent":LargestComponent
        }

if Type == "ER":
    GraphDict["C"] = C
    GraphDict["p"] = p

    SaveDirName= ("SaveFiles/ER_C_%0.3f_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d_LargestComponent_%r"%
                (C,n,P,F,T,SingleActive,Repeats,LargestComponent))

elif Type == "SmallWorld":
    GraphDict["k"] = k
    GraphDict["r"] = r
    GraphDict["t"] = t

    SaveDirName= ("SaveFiles/SW_k_%0.3f_r_%0.3f_t_%0.3f_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d_LargestComponent_%r"%
                (k,r,t,n,P,F,T,SingleActive,Repeats,LargestComponent))

elif Type == "Geometric":
    GraphDict["radius"] = 0

    SaveDirName= ("SaveFiles/Geo_minr_%0.5f_maxr_%0.5f_rnum_%d_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d_LargestComponent_%r"%
                (minr,maxr,rnum,n,P,F,T,SingleActive,Repeats,LargestComponent))


elif Type == "Geometric_Torus":
    GraphDict["radius"] = 0

    SaveDirName= ("SaveFiles/GeoTorus_minr_%0.5f_maxr_%0.5f_rnum_%d_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d_LargestComponent_%r"%
                (minr,maxr,rnum,n,P,F,T,SingleActive,Repeats,LargestComponent))

else:
    raise Exception("Incorrect type of Graph")

SaveDirName += "_DataPoints_%d"%(DataPoints)


if not os.path.isdir("SaveFiles"):
    os.mkdir("SaveFiles")

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
shutil.copyfile("Script.py", SaveDirName+"/Script.py")

