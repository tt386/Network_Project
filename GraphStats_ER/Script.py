from Params import *

import sys
sys.path.insert(0,'../CoreFunctions')

from Core import Init, Initialise, Iterate,  Observe, MeasureMutants, Plot, GraphStats

import time

from matplotlib import pyplot as plt


#################################
###Argparse
#################################
from argparse import ArgumentParser
parser = ArgumentParser(description='Different F and Num')
parser.add_argument(
        '-C',
        '--C',
        type=float,
        required=True,
        help='The prob within which nodes are connected')

args = parser.parse_args()

C = args.C

p = C/n

GraphDict["C"] = C
GraphDict["p"] = C/n

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

meanlist = []
prob2list = []
theorylist = []

GraphSizeList = []
ZealotNumList = []


#Stats about the Graphs
deg_listList = []
deg_cnt_listList = []
MeanClusterCoeffList = []
MeanDegreeList = []

ComponentList = []

for Rep in range(Repeats):
    print("Rep:",Rep)
    InitDict = Init(GraphDict)

    PNum = InitDict["PNum"]

    """
    ParamsDict = {
            "Graph":InitDict["Graph"],
            "InactivePatchIDs":InitDict["InactivePatchIDs"],
            "MNum":InitDict["MNum"],
            "F":F}


    ObserveDict = {
            "Graph":InitDict["Graph"],
            "positions":InitDict["positions"],
            "SaveDirName":os.path.abspath(SaveDirName)
            }
    """

    GraphSize = InitDict["NodeNum"]

    StatDict = GraphStats(InitDict["Graph"])

    ComponentDist = StatDict["ComponentDist"]
    ComponentList += ComponentDist
    MeanClusterCoeffList.append(StatDict["MeanClusterCoeff"])
    MeanDegreeList.append(StatDict["MeanDegree"])

# Count the frequency of each unique integer
unique_integers, counts = np.unique(ComponentList, return_counts=True)

# Create the NumPy array with two columns
Histogram = np.column_stack((unique_integers, counts)).astype(float)

#Histogram[:,1] = Histogram[:,1] / Histogram[:,1].sum()

#print(result_array)

"""
mean = 0
for i in result_array:
    size = i[0]
    prob = i[1]

    mean += prob * size * (1-(1-P)**size)

print("Mean endstate number of M",mean)

if len(result_array) > 1:
    prob2list.append(result_array[1][1])
else:
    prob2list.append(0)
meanlist.append(mean)

theorylist.append(P + n*np.pi*radius**2 * (1-P - (1-P)**2))
"""
"""
print(meanlist)
rlist = np.asarray(rlist)
plt.figure()
plt.scatter(rlist**2,meanlist,label='Mean')
plt.scatter(rlist**2,theorylist,label='Theory')

plt.legend(loc='lower right')

plt.show()

plt.figure()
plt.scatter(rlist**2,prob2list)
plt.show()
"""
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
    GraphSizeList=GraphSizeList,
    MeanClusterCoeffList=MeanClusterCoeffList,
    MeanDegreeList=MeanDegreeList,
    Histogram=Histogram,
    timetaken=timetaken)

