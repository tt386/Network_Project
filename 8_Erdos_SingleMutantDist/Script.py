from pylab import *
import networkx as nx

from Params import *

import random

import copy

import matplotlib.pyplot as plt

import time

def Initialise(n,p,PNum):

    #Create a graph of on massive loop
    RandomGraph = nx.empty_graph(n)
    
    #Remove this as it appears to be a time sink
    #positions = nx.spring_layout(RandomGraph)
    positions =  []

    #Create a loop
    RandomGraph.add_edge(0,n-1)
    for i in range(n-1):
        RandomGraph.add_edge(i,i+1)

    Edges = RandomGraph.edges()

    #Add random edges
    for i in range(n):
        for j in range(i+2,n):
            #if (i,j) not in Edges:
            if random.uniform(0,1) < p:
                RandomGraph.add_edge(i,j)



    """
    #Create Random Graph
    RandomGraph = nx.gnp_random_graph(n, p, seed=None, directed=False)

    #Set the actual physical pixel positions of the nodes
    positions = nx.spring_layout(RandomGraph)


    #Create the same for random geometric
    #GeoGraph = nx.random_geometric_graph(n, R, dim=2, pos=positions, p=2, seed=None)
    """

    """
    #Create CompleteGraph
    CompleteGraph = nx.complete_graph(n)
    """
    CompleteGraph = 0
    MList = []
    SepList = []

    #Datalist: we record:
    #Distance from origin node
    #Degree of node
    #List of the number of times it is invaded by M.
    DataList = []
    

    #Set patches and infections
    for i in range(n):#RandomGraph.nodes(data=True):
        Dist = nx.shortest_path_length(RandomGraph,source=0,target=i)        
        Degree = RandomGraph.degree[i]
        TimeInvadedList = []
        
        DataList.append([Dist,Degree,TimeInvadedList])

        if i < PNum:
            RandomGraph.nodes(data=True)[i]["patch"] = 1
            RandomGraph.nodes(data=True)[i]["infection"] = "M"        
            RandomGraph.nodes(data=True)[i]["label"] = "X"

            """
            CompleteGraph.nodes(data=True)[i]["patch"] = 1
            CompleteGraph.nodes(data=True)[i]["infection"] = "M"
            CompleteGraph.nodes(data=True)[i]["label"] = "X"
            """
            MList.append(i)
            SepList.append(0)

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




    return positions, CompleteGraph, RandomGraph, MList,SepList, DataList

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


def Iterate(t,Graph,F,MList,SepDist,DataList):

    #Create Graph node list now from profiling
    Nodes = Graph.nodes()


    #Choose a random index
    randindex = random.randint(0,len(Nodes)-1)
    randnode = Nodes[randindex]#random.choice(Graph.nodes())

    while randnode["patch"]:
        randindex = random.randint(0,len(Nodes)-1)
        randnode = Nodes[randindex]

    MNum = 0
    Num = 0

    #Count yourself first
    if randnode["infection"] == "M":
        MNum += 1

    Num += 1


    #Count your neighbours
    for j in list(Graph.neighbors(randindex)):
        Num += 1

        if Nodes[j]["infection"] == "M":
            MNum += 1

    #Adjust self
    if randnode["patch"] == 0:
        if random.uniform(0,1) < MNum*F/((Num-MNum) + MNum*F):
            Nodes(data=True)[randindex]["infection"] = "M"

            if randindex not in MList:
                MList.append(randindex)
                Sep = nx.shortest_path_length(Graph,source=0,target=randindex)
                SepDist.append(Sep)

                DataList[randindex][2].append(t)
                #SepDist = MeasureSepDist(Graph,MList)

        else:
            Nodes(data=True)[randindex]["infection"] = "WT"

            if randindex in MList:
                index = MList.index(randindex)
                #MList.remove(randindex)
                del MList[index]
                del SepDist[index]
                #SepDist = MeasureSepDist(Graph,MList)

    return Graph, MList, SepDist, DataList


#################################
###Argparse
#################################
from argparse import ArgumentParser
parser = ArgumentParser(description='Different F and Num')
parser.add_argument('-F','--Fitness',type=float,required=True,help='Fitness')
#parser.add_argument('-N','--Num',type=float,required=True,help='Number')
args = parser.parse_args()

#PNum = args.Num
F = args.Fitness

SubSaveDirName = (SaveDirName +
    "/F_%0.3f"%(F))

if not os.path.isdir(SubSaveDirName):
    os.mkdir(SubSaveDirName)
    print("Created Directory for Fitness",F)

print("Starting F: %0.3f"%(F))

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

for Rep in range(Repeats):
    print("Repeat: %d"%(Rep))

    #Initialise
    positions, CompleteGraph, RandomGraph, MList, SepDist, DataList = Initialise(n,p,PNum)

    for t in range(T):
        #Interation
        #CompleteGraph = Iterate(CompleteGraph,F)
        RandomGraph, MList, SepDist, DataList  = Iterate(t,RandomGraph,F,MList,SepDist,DataList)

        #Measure
        #Complete_Mnum[Rep].append(Measure(CompleteGraph)/n)
        #Random_Mnum[Rep].append(Measure(RandomGraph)/n)

        Random_Mnum[Rep].append(len(MList)/n)

        Random_MSepDist[Rep].append(copy.copy(SepDist))

    DataMatrix.append(DataList)

    print("MList,",MList)
    print("SepDist,",SepDist)


#Correct Data Presentation
DataMatrix = np.array(DataMatrix,dtype=object)

#Complete_MeanN = np.mean(np.asarray(Complete_Mnum),axis=0)
Random_MeanN = np.mean(np.asarray(Random_Mnum),axis=0)


#Complete_MedianN = np.median(np.asarray(Complete_Mnum),axis=0)
Random_MedianN = np.median(np.asarray(Random_Mnum),axis=0)

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
    Mnum=Random_Mnum,
    MeanN=Random_MeanN,
    MedianN=Random_MedianN,
    Random_MSepDist=Random_MSepDist,
    DataMatrix=DataMatrix,
    timetaken=timetaken)

print("Finished F: %0.3f, PNum: %d"%(F,PNum))
print("Time Taken:",timetaken)
