from pylab import *
import networkx as nx

from Params import *

import random

import copy

import matplotlib.pyplot as plt

import time

def Initialise(n,p,R,PNum):

    #Create Random Graph
    RandomGraph = nx.gnp_random_graph(n, p, seed=None, directed=False)

    #Set the actual physical pixel positions of the nodes
    positions = nx.spring_layout(RandomGraph)


    #Create the same for random geometric
    GeoGraph = nx.random_geometric_graph(n, R, dim=2, pos=positions, p=2, seed=None)


    #Create CompleteGraph
    CompleteGraph = nx.complete_graph(n)


    #Set patches and infections
    for i in range(n):#RandomGraph.nodes(data=True):
        if i < PNum:
            RandomGraph.nodes(data=True)[i]["patch"] = 1
            RandomGraph.nodes(data=True)[i]["infection"] = "M"        
            RandomGraph.nodes(data=True)[i]["label"] = "X"

            CompleteGraph.nodes(data=True)[i]["patch"] = 1
            CompleteGraph.nodes(data=True)[i]["infection"] = "M"
            CompleteGraph.nodes(data=True)[i]["label"] = "X"


            GeoGraph.nodes(data=True)[i]["patch"] = 1
            GeoGraph.nodes(data=True)[i]["infection"] = "M"
            GeoGraph.nodes(data=True)[i]["label"] = "X"


        else:
            RandomGraph.nodes(data=True)[i]["patch"] = 0
            RandomGraph.nodes(data=True)[i]["infection"] = "WT"
            RandomGraph.nodes(data=True)[i]["label"] = ""

            CompleteGraph.nodes(data=True)[i]["patch"] = 0
            CompleteGraph.nodes(data=True)[i]["infection"] = "WT"
            CompleteGraph.nodes(data=True)[i]["label"] = ""

            GeoGraph.nodes(data=True)[i]["patch"] = 0
            GeoGraph.nodes(data=True)[i]["infection"] = "WT"
            GeoGraph.nodes(data=True)[i]["label"] = ""

    return positions, CompleteGraph, RandomGraph, GeoGraph


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


def Iterate(Graph):

    randindex = random.randint(0,len(Graph.nodes())-1)

    randnode = Graph.nodes()[randindex]#random.choice(Graph.nodes())
    #print(randnode)

    MNum = 0
    Num = 0

    #Count yourself first
    if randnode["infection"] == "M":
        MNum += 1

    Num += 1


    #Count your neighbours
    for j in list(Graph.neighbors(randindex)):
        Num += 1

        if Graph.node[j]["infection"] == "M":
            MNum += 1

    #Adjust self
    if randnode["patch"] == 0:
        if random.uniform(0,1) < MNum*F/((Num-MNum) + MNum*F):
            Graph.nodes(data=True)[randindex]["infection"] = "M"

        else:
            Graph.nodes(data=True)[randindex]["infection"] = "WT"

    return Graph

    """
    tempgraph = copy.deepcopy(Graph)
    
    #For a loop of many
    for i in Graph.nodes(data=True):
        MNum = 0
        Num = 0

        #Count yourself first
        if i[1]["infection"] == "M":
            MNum += 1

        Num += 1


        #Count your neighbours
        for j in list(Graph.neighbors(i[0])):
            Num += 1
    
            if Graph.node[j]["infection"] == "M":
                MNum += 1

        #Adjust self
        if i[1]["patch"] == 0:
            if random.uniform(0,1) < MNum*F/((Num-MNum) + MNum*F):
                tempgraph.nodes(data=True)[i[0]]["infection"] = "M"

            else:
                tempgraph.nodes(data=True)[i[0]]["infection"] = "WT"
    
    return tempgraph
    """
starttime = time.time()


#Create Savelists
Complete_Mnum = []
Random_Mnum = []
Geometric_Mnum = []

for i in range(Repeats):    
    Complete_Mnum.append([])
    Random_Mnum.append([])
    Geometric_Mnum.append([])


for Rep in range(Repeats):
    print("Repeat: %d"%(Rep))

    #Initialise
    positions, CompleteGraph, RandomGraph, GeometricGraph = Initialise(n,p,R,PNum)


    for t in range(T):
        #print("Repeat: %d, timestep: %d"%(Rep,t))

        """
        if Rep == 0:
            Observe(t,CompleteGraph,RandomGraph,GeometricGraph,positions)
        """
        #Interation
        CompleteGraph = Iterate(CompleteGraph)
        #RandomGraph = Iterate(RandomGraph)
        #GeometricGraph = Iterate(GeometricGraph)

        #Measure
        Complete_Mnum[Rep].append(Measure(CompleteGraph)/n)
        #Random_Mnum[Rep].append(Measure(RandomGraph)/n)
        #Geometric_Mnum[Rep].append(Measure(GeometricGraph)/n)

        

Complete_MeanN = np.mean(np.asarray(Complete_Mnum),axis=0)
#Random_MeanN = np.mean(np.asarray(Random_Mnum),axis=0)
#Geometric_MeanN = np.mean(np.asarray(Geometric_Mnum),axis=0)


Complete_MedianN = np.median(np.asarray(Complete_Mnum),axis=0)
#Random_MedianN = np.median(np.asarray(Random_Mnum),axis=0)
#Geometric_MedianN = np.median(np.asarray(Geometric_Mnum),axis=0)



#Plotting
plt.figure()
for i in range(len(Complete_Mnum)):
    plt.plot(np.arange(T),Complete_Mnum[i],linewidth=0.5)

plt.plot(np.arange(T),Complete_MeanN,'k',linewidth=3,label="Mean")
plt.plot(np.arange(T),Complete_MedianN,'y',linewidth=3,label="Median")

plt.legend(loc='lower right')

plt.savefig(SaveDirName + "/MRatio_Complete.png")
plt.close()

"""
plt.figure()
for i in range(len(Random_Mnum)):
    plt.plot(np.arange(T),Random_Mnum[i])

plt.plot(np.arange(T),Random_MeanN,'k',linewidth=3,label="Mean")
plt.plot(np.arange(T),Random_MedianN,'--k',linewidth=3,label="Median")

plt.legend(loc='lower right')

plt.savefig(SaveDirName + "/MRatio_Random.png")
plt.close()



plt.figure()
for i in range(len(Geometric_Mnum)):
    plt.plot(np.arange(T),Geometric_Mnum[i])

plt.plot(np.arange(T),Geometric_MeanN,'k',linewidth=3,label="Mean")
plt.plot(np.arange(T),Geometric_MedianN,'--k',linewidth=3,label="Median")

plt.legend(loc='lower right')

plt.savefig(SaveDirName + "/MRatio_Geometric.png")
plt.close()
"""
endtime = time.time()
print("Time taken:",endtime-starttime)
