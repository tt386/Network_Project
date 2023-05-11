import networkx as nx

from pylab import *

import random

import copy

import matplotlib.pyplot as plt

import time

import sys


def Initialise(n,p,P):
    """Generates Network and key parameters of that network

    Arguments:
        n:      the number of nodes
        p:      the probability two nodes are connected
        P:      the proportion of zealots
    
    Returns:
        positions:          List positions of nodes useful for plotting
        CompleteGraph:      CompleteGraph networkx object
        RandomGraph:        RandomGraph Networkx object
        InactivePatchIDs:   List ID's of inactive Zealots
    """

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

    NodeNum = RandomGraph.number_of_nodes()

    positions = nx.spring_layout(RandomGraph)#[]
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

    MNum = 0

    #Number of patches
    PNum = np.round(P * NodeNum).astype(int)

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
            RandomGraph.nodes(data=True)[i]["label"] = "Z"
            RandomGraph.nodes(data=True)[i]["active"] = True
            MNum += 1

            #Make it an inactive site
            if NodesPlaced > 0:
                RandomGraph.nodes(data=True)[i]["active"] = False
                RandomGraph.nodes(data=True)[i]["infection"] = "WT"
                InactivePatchIDs.append(i)
                MNum -= 1
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

    InitDict = {
            "positions":positions,
            "CompleteGraph":CompleteGraph,
            "RandomGraph":RandomGraph,
            "InactivePatchIDs":InactivePatchIDs,
            "MNum":MNum,
            "NodeNum":NodeNum,
            "PNum":PNum}

    return InitDict#positions, CompleteGraph, RandomGraph, InactivePatchIDs






def Iterate(ParamDict):#t,Graph,F,InactivePatchIDs):
    """Iterate a single infection probability event

    """

    t = ParamDict["t"]
    Graph = ParamDict["Graph"]
    F = ParamDict["F"]
    InactivePatchIDs = ParamDict["InactivePatchIDs"]
    GraphMNum = ParamDict["MNum"]

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
    InitialInfection = "WT"

    if randnode["infection"] == "M":
        MNum += 1
        InitialInfection = "M"

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
    FinalInfection = "M"
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
        FinalInfection = "WT"
        """
        if randindex in MList:
            index = MList.index(randindex)
            #MList.remove(randindex)
            del MList[index]
            del SepDist[index]
            #SepDist = MeasureSepDist(Graph,MList)
        """


    if InitialInfection != FinalInfection:
        if FinalInfection == "M":
            GraphMNum += 1
        else:
            GraphMNum -= 1

    """
    ResultsDict = {
            "Graph": Graph,
            "InactivePatchIDs": InactivePatchIDs,
            "InactivePatchActivated": InactivePatchActivated,
            "MNum": GraphMNum}
    """

    ParamDict["MNum"] = GraphMNum
    ParamDict["InactivePatchActivated"] = InactivePatchActivated

    return ParamDict#Graph, InactivePatchIDs,InactivePatchActivated



def Observe(ObserveDict):#(t,Graph,positions,SaveDirName):
    """Save an image of the graphs

    """
    Graph = ObserveDict["Graph"]
    t = ObserveDict["t"]
    positions = ObserveDict["positions"]
    SaveDirName = ObserveDict["SaveDirName"]


    labels = nx.get_node_attributes(Graph, 'label')

    """
    print("Labels:",labels)

    print("Graph Nodes:",Graph.nodes(data=True))

    print("Positons:",positions)


    for i in Graph.nodes(data=True):
        if i[1]["infection"] == "WT":
            print("Blue")

        else:
            print("Red")


    print("Finished List")
    """
    colorlist = []
    for i in Graph.nodes(data=True):
        color = None
        if i[1]["infection"]=="WT":
            color = "blue"
        else:
            color = "red"

        if i[1]["patch"]  and not i[1]["active"]:
            color = "white"
        
        colorlist.append(color)


    nx.draw(Graph,
            positions,
            labels=labels,
            node_color=colorlist,
            node_size=100#['blue' if i[1]["infection"]=="WT" else 'red' for  i in Graph.nodes(data=True)]
            )

    print("MADE")
    savefig(SaveDirName + "/Snapshot_t_" + str(t).zfill(5))
    close()



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
"""


def MeasureMutants(Graph):
    """Count number of mutants in Graph
    """
    MNum = 0
    Num = 0

    for i in Graph.nodes(data=True):
        if i[1]["infection"] == "M":
            MNum += 1

        Num += 1

    return MNum


def MeasureSepDist(Graph,MList):
    """Generate a list of Mutant distances from initial node.
    """
    SepList = []

    for i in MList:
        if i!= 0:
            Sep = nx.shortest_path_length(Graph,source=0,target=i)
            SepList.append(Sep)

    return SepList




def Plot(PlotDict):

    plt.figure()
    plt.plot(PlotDict["xlist"],PlotDict["ylist"])
    plt.savefig(PlotDict["SaveDirName"] + PlotDict["FigName"])
    plt.close()


