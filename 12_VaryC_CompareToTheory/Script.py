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
parser.add_argument(
        '-C',
        '--MeanConnection',
        type=float,
        required=True,
        help='Mean Number of Connection')

args = parser.parse_args()

C = args.MeanConnection
p = C/n

SubSaveDirName = (SaveDirName +
    "/C_%0.3f"%(C))

if not os.path.isdir(SubSaveDirName):
    os.mkdir(SubSaveDirName)
    print("Created Directory for C",C)

print("Starting C: %0.3f"%(C))

#############################################################
#############################################################
#############################################################



starttime = time.time()


#Lists to be saved
MNumMatrix = []
GraphSizeList = []
ZealotNumList = []

#Repeated iterations:

for R in range(Repeats):
    print("C",C,"Repeat",R)

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

    MNumList = []
    ZealotInvadedTime = []
    MZealotInvadedTime = []
    for t in range(T):
        ParamsDict["t"] = t

        ParamsDict = Iterate(ParamsDict)

        MNumList.append(ParamsDict["MNum"])

        if ParamsDict["InactivePatchActivated"]:
            ZealotInvadedTime.append(t)
            MZealotInvadedTime.append(MNumList[-1])

    MNumMatrix.append(MNumList)

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
    MNumMatrix=MNumMatrix,
    GraphSizeList=GraphSizeList,
    ZealotNumList=ZealotNumList,
    timetaken=timetaken)



"""
plt.figure()
plt.plot(np.arange(0,len(MNumList)),MNumList/GraphSize)
for i in ZealotInvadedTime:
    plt.axvline(x=i,color='red',alpha = 0.1)

plt.plot([0,len(MNumList)],[Theory,Theory],color='black')

plt.savefig(os.path.abspath(SaveDirName) + "/MNum.png")
plt.close



#Correllations
ZTimegaplist = []
MZTimegaplist = []
for i in range(len(ZealotInvadedTime)-1):
    ZTimegaplist.append(ZealotInvadedTime[i+1]- ZealotInvadedTime[i])
    MZTimegaplist.append(MZealotInvadedTime[i+1]- MZealotInvadedTime[i])



plt.figure()
for i in range(len(ZTimegaplist)-1):
    plt.scatter([ZTimegaplist[i]],[ZTimegaplist[i+1]],color='blue')

plt.savefig(os.path.abspath(SaveDirName) + "/xgapCorrelations.png")
plt.close()


plt.figure()
for i in range(len(ZTimegaplist)-1):
    plt.scatter([ZTimegaplist[i]],[MZTimegaplist[i+1]],color='blue')

plt.savefig(os.path.abspath(SaveDirName) + "/xygapCorrelations.png")
plt.close()
"""
endtime = time.time()

timetaken = endtime-starttime

print("Time Taken:",timetaken)
