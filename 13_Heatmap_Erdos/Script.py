from Params import *

import sys
sys.path.insert(0,'../CoreFunctions')

from Core import Initialise, Iterate,  Observe, MeasureMutants, Plot

import time

#################################
###Argparse
#################################
from argparse import ArgumentParser
parser = ArgumentParser(description='Different F and Num')
parser.add_argument('-F','--Fitness',type=float,required=True,help='Fitness')
parser.add_argument('-P','--ZProb',type=float,required=True,help='Zealot prob')
args = parser.parse_args()

P = args.ZProb
F = args.Fitness

SubSaveDirName = (SaveDirName +
    "/P_%0.3f_F_%0.3f"%(P,F))

if not os.path.isdir(SubSaveDirName):
    os.mkdir(SubSaveDirName)
    print("Created Directory for ZProb",P)
    print("Created Directory for Fitness",F)

print("Starting F: %0.3f, ZProb: %0.3f"%(F,P))
#################################
#################################

starttime = time.time()

#Lists to be saved
MPropMatrix = []
GraphSizeList = []
ZealotNumList = []

#Repeated iterations:

for R in range(Repeats):
    print("ZProb",P,"F",F,"Repeat",R)

    InitDict = Initialise(n,p,P)

    PNum = InitDict["PNum"]

    ParamsDict = {
            "Graph":InitDict["RandomGraph"],
            "InactivePatchIDs":InitDict["InactivePatchIDs"],
            "MNum":InitDict["MNum"],
            "F":F}

    GraphSize = InitDict["NodeNum"]
    GraphSizeList.append(GraphSize)
    ZealotNumList.append(PNum)

    MPropList = []
    ZealotInvadedTime = []
    MZealotInvadedTime = []
    for t in range(T):
        ParamsDict["t"] = t

        ParamsDict = Iterate(ParamsDict)

        MPropList.append(ParamsDict["MNum"]/GraphSize)

        if ParamsDict["InactivePatchActivated"]:
            ZealotInvadedTime.append(t)
            MZealotInvadedTime.append(MPropList[-1])

    MPropMatrix.append(MPropList)

############################################################################
############################################################################
############################################################################

endtime = time.time()

timetaken = endtime-starttime

print("Time Taken:",timetaken)

############################################################################
###Saving###################################################################
############################################################################
OutputDatafilename = SubSaveDirName + '/datafile.npz'
np.savez(OutputDatafilename,
    n=n,
    C=C,
    Repeats=Repeats,
    T=T,
    p=p,
    P=P,
    F=F,
    PNum=PNum,
    MPropMatrix=MPropMatrix,
    GraphSizeList=GraphSizeList,
    ZealotNumList=ZealotNumList,
    timetaken=timetaken)

