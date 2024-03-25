from Params import *

import sys
sys.path.insert(0,'../CoreFunctions')

from Core import Init, Iterate,  Observe, MeasureMutants, Plot, GraphStats

import time




#################################
###Argparse
#################################
from argparse import ArgumentParser
parser = ArgumentParser(description='Different F and Num')

"""
parser.add_argument(
        '-C',
        '--C',
        type=float,
        required=True,
        help='The radius within which nodes are connected')
"""

parser.add_argument(
        '-P',
        '--P',
        type=float,
        required=True,
        help='Proportion of Zealots')


parser.add_argument(
        '-F',
        '--F',
        type=float,
        required=True,
        help='M Fitness')

parser.add_argument(
        '-d',
        '--dir',
        type=str,
        required=True,
        help='directory save data in')



args = parser.parse_args()

C = n
P = float(args.P)
F = float(args.F)
SaveDirName = str(args.dir)


GraphDict["C"] = C
GraphDict["p"] = C/n
GraphDict["P"] = P


SubSaveDirName = (SaveDirName +
    "/P_%0.3f_F_%0.3f"%(P,F))

if not os.path.isdir(SubSaveDirName):
    os.mkdir(SubSaveDirName)
    print("Created Directory for P,F",P,F)

print("Starting C: %0.3f"%(C))

#############################################################
#############################################################
#############################################################



starttime = time.time()


#Lists to be saved
MNumMatrix = []
GraphSizeList = []
ZealotNumList = []


#Stats about the Graphs
deg_listList = []
deg_cnt_listList = []
MeanClusterCoeffList = []
MeanDegreeList = []

#Repeated iterations:

for R in range(Repeats):
    print("P,F",P,F,"Repeat",R)

    InitDict = Init(GraphDict)

    PNum = InitDict["PNum"]

    ParamsDict = {
            "Graph":InitDict["Graph"],
            "InactivePatchIDs":InitDict["InactivePatchIDs"],
            "MNum":InitDict["MNum"],
            "F":F}

    GraphSize = InitDict["NodeNum"]
    GraphSizeList.append(GraphSize)
    ZealotNumList.append(PNum)

    #Generate statdictionary and unpack
    StatDict = GraphStats(ParamsDict["Graph"])
    deg_listList.append(StatDict["deg_list"])
    deg_cnt_listList.append(StatDict["deg_cnt_list"])
    MeanClusterCoeffList.append(StatDict["MeanClusterCoeff"])
    MeanDegreeList.append(StatDict["MeanDegree"])    


    MNumList = []
    ZealotInvadedTime = []
    MZealotInvadedTime = []
    for t in range(T):
        ParamsDict["t"] = t

        ParamsDict = Iterate(ParamsDict)

        if t > T-DataPoints:
            MNumList.append(ParamsDict["MNum"])

        """
        if ParamsDict["InactivePatchActivated"]:
            ZealotInvadedTime.append(t)
            MZealotInvadedTime.append(MNumList[-1])
        """
    MNumMatrix.append(MNumList)

MNumMatrix = np.asarray(MNumMatrix)
GraphSizeList = np.asarray(GraphSizeList)

Ratio = np.divide(MNumMatrix,GraphSizeList[:,np.newaxis])

Mean = np.mean(Ratio,axis=0)
Median = np.median(Ratio,axis=0)

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
    P=P,
    F=F,
    PNum=PNum,
    Mean=Mean,
    #Matrix=Matrix,
    #MNumMatrix=MNumMatrix,
    GraphSizeList=GraphSizeList,
    ZealotNumList=ZealotNumList,
    #deg_listList=deg_listList,
    #deg_cnt_listList=deg_cnt_listList,
    MeanClusterCoeffList=MeanClusterCoeffList,
    MeanDegreeList=MeanDegreeList,
    DataPoints=DataPoints,
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
