from pylab import *
import networkx as nx

from Params import *

import random

import copy

import matplotlib.pyplot as plt

import time

import sys


def Initialise(n,p,PNum):


    #Create random graph:
    RandomGraph = nx.gnp_random_graph(n, p, seed=None, directed=False)

    #Isolate the largest component
    #ConnectedComponents = sorted(nx.connected_components(RandomGraph),key=len,reverse=True)
    #RandomGraph = G.subgraph(Gcc[0])
    LargestComponent = max(nx.connected_components(RandomGraph), key=len)

    Nodes = set(RandomGraph.nodes())

    Difference = Nodes - LargestComponent

    #print("Largest comp",LargestComponent)
    #print("Nodes",Nodes)

    for i in Difference:
        RandomGraph.remove_node(i)

    Nodes = RandomGraph.nodes()

    positions = []
    CompleteGraph = 0
    MList = []
    SepList = []

    #Datalist: we record:
    #Distance from origin node
    #Degree of node
    #List of the number of times it is invaded by M.
    DataList = []
    

    #Set patches and infections
    NodesPlaced = 0

    """
    MAKE A LIST THAT KEEPS TRACK OF INACTIVE PATCHES - DELETE AS INVASIONS HAPPEN, SIM ENDS WHEN THIS LIST EMPTY
    CREATE NEW ASPECT FOR PATCHES THAT INDICATE IF ACTIVE OR NOT
    CHANGE UPDATE DYNAMICS TO ACCOUNT FOR INACTIVE PATCHES HAVING NO MEMBERS
    CHANGE UPDATE DYNAMICS SO INACTIVE PATCHES CAN BE INVADED BY M
    """
    InactivePatchIDs = []

    for i in Nodes:#RandomGraph.nodes(data=True):
        """    
        Dist = nx.shortest_path_length(RandomGraph,source=0,target=i)        
        Degree = RandomGraph.degree[i]
        TimeInvadedList = []
        DataList.append([Dist,Degree,TimeInvadedList])
        """
        if NodesPlaced < PNum:
            RandomGraph.nodes(data=True)[i]["patch"] = 1
            RandomGraph.nodes(data=True)[i]["infection"] = "M"        
            RandomGraph.nodes(data=True)[i]["label"] = "X"
            RandomGraph.nodes(data=True)[i]["active"] = True


            #Make it an inactive site
            if NodesPlaced > 0:
                RandomGraph.nodes(data=True)[i]["active"] = False
                RandomGraph.nodes(data=True)[i]["infection"] = "WT"
                InactivePatchIDs.append(i)
            """
            CompleteGraph.nodes(data=True)[i]["patch"] = 1
            CompleteGraph.nodes(data=True)[i]["infection"] = "M"
            CompleteGraph.nodes(data=True)[i]["label"] = "X"
            """
            MList.append(i)
            SepList.append(0)

            NodesPlaced += 1
            #GeoGraph.nodes(data=True)[i]["patch"] = 1
            #GeoGraph.nodes(data=True)[i]["infection"] = "M"
            #GeoGraph.nodes(data=True)[i]["label"] = "X"


        else:
            RandomGraph.nodes(data=True)[i]["patch"] = 0
            RandomGraph.nodes(data=True)[i]["infection"] = "WT"
            RandomGraph.nodes(data=True)[i]["label"] = ""
            """
            CompleteGraph.nodes(data=True)[i]["patch"] = 0
            CompleteGraph.nodes(data=True)[i]["infection"] = "WT"
            CompleteGraph.nodes(data=True)[i]["label"] = ""
            """
            #GeoGraph.nodes(data=True)[i]["patch"] = 0
            #GeoGraph.nodes(data=True)[i]["infection"] = "WT"
            #GeoGraph.nodes(data=True)[i]["label"] = ""

    #print(InactivePatchIDs)
    #sys.exit()

    return positions, CompleteGraph, RandomGraph, InactivePatchIDs

"""
def Observe(t,CompleteGraph,RandomGraph,GeometricGraph,positions):
    labels = nx.get_node_attributes(CompleteGraph, 'label')

    #Draw Complete
    nx.draw(CompleteGraph,      #Graph
        positions,  #Defined positions
        labels = labels,
        node_color=['blue' if i[1]["infection"]=="WT" else 'red' for  i in CompleteGraph.nodes(data=True)]
        )
    savefig(SaveDirName + "/Complete_%d.png"%(t))
    close()


    #Draw Complete
    nx.draw(RandomGraph,      #Graph
        positions,  #Defined positions
        labels = labels,
        node_color=['blue' if i[1]["infection"]=="WT" else 'red' for  i in RandomGraph.nodes(data=True)]
        )
    savefig(SaveDirName + "/Random_%d.png"%(t))
    close()


    #Draw Geometric
    nx.draw(GeometricGraph,      #Graph
        positions,  #Defined positions
        labels = labels,
        node_color=['blue' if i[1]["infection"]=="WT" else 'red' for  i in GeometricGraph.nodes(data=True)]
        )
    savefig(SaveDirName + "/Geometric_%d.png"%(t))
    close()


def Measure(Graph):
    MNum = 0
    Num = 0

    for i in Graph.nodes(data=True):
        if i[1]["infection"] == "M":
            MNum += 1

        Num += 1

    return MNum
"""

def MeasureSepDist(Graph,MList):
    SepList = []

    for i in MList:
        if i!= 0:
            Sep = nx.shortest_path_length(Graph,source=0,target=i)
            SepList.append(Sep)

    return SepList


def Iterate(t,Graph,F,InactivePatchIDs):

    #Create Graph node list now from profiling
    Nodes = Graph.nodes()

    NodeKeyList = list(Nodes.keys())
    
    #print("NODES",Nodes)
    #sys.exit()

    InactivePatchActivated = False

    #Choose a random index
    randindex = random.choice(NodeKeyList)#random.randint(0,len(Nodes)-1)
    randnode = Nodes[randindex]#random.choice(Graph.nodes())

    while randnode["patch"] and randnode["active"]:
        randindex =  random.choice(NodeKeyList)#random.randint(0,len(Nodes)-1)
        randnode = Nodes[randindex]

    MNum = 0
    Num = 0

    #Count yourself first
    if randnode["infection"] == "M":
        MNum += 1

    Num += 1

    if (randnode["patch"]) and (not randnode["active"]):
        Num -= 1


    #Count your neighbours
    for j in list(Graph.neighbors(randindex)):
        Num += 1

        if Nodes[j]["infection"] == "M":
            MNum += 1

        if Nodes[j]["patch"] and (not Nodes[j]["active"]):
            Num -= 1


    if Num == 0:
        Num = 1
    #Adjust self
    if random.uniform(0,1) < MNum*F/((Num-MNum) + MNum*F):
        Nodes(data=True)[randindex]["infection"] = "M"

        """
        if randindex not in MList:
            MList.append(randindex)
            Sep = nx.shortest_path_length(Graph,source=0,target=randindex)
            SepDist.append(Sep)

            DataList[randindex][2].append(t)
        """
        if randnode["patch"]:
            randnode["active"] = 1
            InactivePatchIDs.remove(randindex)
            InactivePatchActivated = True        
            #SepDist = MeasureSepDist(Graph,MList)
        
    else:
        Nodes(data=True)[randindex]["infection"] = "WT"

        """
        if randindex in MList:
            index = MList.index(randindex)
            #MList.remove(randindex)
            del MList[index]
            del SepDist[index]
            #SepDist = MeasureSepDist(Graph,MList)
        """
    return Graph, InactivePatchIDs,InactivePatchActivated#MList, SepDist, DataList


#################################
###Argparse
#################################
from argparse import ArgumentParser
parser = ArgumentParser(description='Different F and Num')
parser.add_argument('-N','--Number',type=float,required=True,help='PNumber')
#parser.add_argument('-N','--Num',type=float,required=True,help='Number')
args = parser.parse_args()

#PNum = args.Num
PNum = args.Number

SubSaveDirName = (SaveDirName +
    "/PNum_%d"%(PNum))

if not os.path.isdir(SubSaveDirName):
    os.mkdir(SubSaveDirName)
    print("Created Directory for PNum",PNum)

print("Starting PNum: %0.3f"%(PNum))

starttime = time.time()


#Create Savelists
Complete_Mnum = []
Random_Mnum = []
Geometric_Mnum = []


Random_MSepDist = []

DataMatrix = []

for i in range(Repeats):    
    Complete_Mnum.append([])
    Random_Mnum.append([])

    Random_MSepDist.append([])


TimeMatrix = np.zeros((Repeats,int(PNum)-1))

for Rep in range(Repeats):
    print("Repeat: %d"%(Rep))

    TimeList = []

    #Initialise
    positions, CompleteGraph, RandomGraph, InactivePatchIDs = Initialise(n,p,PNum)

    infectednum = 0
    t = 0
    while True:
        #Interation
        #CompleteGraph = Iterate(CompleteGraph,F)
        RandomGraph, InactivePatchIDs, InactivePatchActivated  = Iterate(t,RandomGraph,F,InactivePatchIDs)

        if InactivePatchActivated:
            TimeMatrix[Rep][infectednum] = t
            infectednum += 1
            #TimeList.append(t)

            if len(InactivePatchIDs) == 0:
                break
        t += 1
    #TimeMatrix.append(np.asarray(TimeList))
endtime = time.time()
timetaken = endtime-starttime
print("Before save time taken:",timetaken)

OutputDatafilename = SubSaveDirName + '/datafile.npz'
np.savez(OutputDatafilename,
    n=n,
    Repeats=Repeats,
    T=T,
    p=p,
    F=F,
    PNum=PNum,
    TimeMatrix = TimeMatrix,
    #Mnum=Random_Mnum,
    #MeanN=Random_MeanN,
    #MedianN=Random_MedianN,
    #Random_MSepDist=Random_MSepDist,
    #DataMatrix=DataMatrix,
    timetaken=timetaken)

print("Finished F: %0.3f, PNum: %d"%(F,PNum))
print("Time Taken:",timetaken)
