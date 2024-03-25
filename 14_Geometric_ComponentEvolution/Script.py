from Params import *

import sys
sys.path.insert(0,'../CoreFunctions')

from Core import Init, Iterate,  Observe, MeasureMutants, Plot, GraphStats

import networkx as nx

import collections
import time




#################################
###Argparse
#################################
from argparse import ArgumentParser
parser = ArgumentParser(description='Different F and Num')
parser.add_argument(
        '-radius',
        '--radius',
        type=float,
        required=True,
        help='The radius within which nodes are connected')

args = parser.parse_args()

radius = args.radius

GraphDict["radius"] = radius

SubSaveDirName = (SaveDirName +
    "/radius_%0.3f"%(radius))

if not os.path.isdir(SubSaveDirName):
    os.mkdir(SubSaveDirName)
    print("Created Directory for radius",radius)

print("Starting radius: %0.3f"%(radius))

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
EdgeMatrix = []
MeanDegreeMatrix = []
for i in range(n+1):
    EdgeMatrix.append([])
    MeanDegreeMatrix.append([])
for R in range(10):#(Repeats):
    print("radius",radius,"Repeat",R)

    InitDict = Init(GraphDict)

    Graph = InitDict["Graph"]

    print(Graph)

    S = [Graph.subgraph(c).copy() for c in nx.connected_components(Graph)]

    for i in S:
        #print(i.number_of_nodes())
        EdgeMatrix[i.number_of_nodes()].append(i.number_of_edges())


        ################################################
        #Find the Graph Degree Distribution
        degree_sequence = sorted([d for n, d in i.degree()], reverse=True)  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, deg_cnt = zip(*degreeCount.items())
        ################################################

        deg = np.asarray(deg)
        deg_cnt = np.asarray(deg_cnt)

        ################################################
        #Use above to find Mean Graph Degree Distribution
        MeanDegree = np.sum(deg*deg_cnt)/sum(deg_cnt)
        ################################################

        MeanDegreeMatrix[i.number_of_nodes()].append(MeanDegree)

        #print(i)
DataMatrix = []
for i in range(len(EdgeMatrix)):
    sublist = []
    sublist.append(np.mean(EdgeMatrix[i]))
    sublist.append(np.mean(MeanDegreeMatrix[i]))
    print(i,sublist)
    DataMatrix.append(sublist)
    #print("Edge Num",i,EdgeMatrix[i])
    #print("MeanDegree",i,MeanDegreeMatrix)
sys.exit()
"""

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
    radius=radius,
    Repeats=Repeats,
    T=T,
    p=p,
    P=P,
    F=F,
    PNum=PNum,
    MNumMatrix=MNumMatrix,
    GraphSizeList=GraphSizeList,
    ZealotNumList=ZealotNumList,
    #deg_listList=deg_listList,
    #deg_cnt_listList=deg_cnt_listList,
    MeanClusterCoeffList=MeanClusterCoeffList,
    MeanDegreeList=MeanDegreeList,
    timetaken=timetaken)
"""


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
