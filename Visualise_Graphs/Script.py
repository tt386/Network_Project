from Params import *

import sys
sys.path.insert(0,'../CoreFunctions')

from Core import Init, Initialise, Iterate,  Observe, MeasureMutants, Plot, GraphStats



import time


starttime = time.time()

#positions, CompleteGraph, RandomGraph, InactivePatchIDs = Initialise(n,p,PNum)


InitDict = Init(GraphDict)

PNum = InitDict["PNum"]

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


GraphSize = InitDict["NodeNum"]

StatDict = GraphStats(ParamsDict["Graph"])
print("MeanClusterCoeff:",StatDict["MeanClusterCoeff"])
print("degree list:",StatDict["deg_list"])
print("degree list counts",StatDict["deg_cnt_list"])

print("Mean degree:",np.sum(StatDict["deg_list"]*StatDict["deg_cnt_list"])/sum(StatDict["deg_cnt_list"]))
print("Graph Size:",StatDict["GraphSize"])


print("Connected components",StatDict["ComponentDist"])

MNumList = []
ZealotInvadedTime = []
MZealotInvadedTime = []
for t in range(T):
    ParamsDict["t"] = t

   # RandomGraph, InactivePatchIDs, InactivePatchActivated  = Iterate(t,RandomGraph,F,InactivePatchIDs)

    ParamsDict = Iterate(ParamsDict)


    """
    ResultsDict = Iterate(ParamsDict)

    ParamsDict["Graph"] = ResultsDict["Graph"]
    ParamsDict["InactivePatchIDs"] = ResultsDict["InactivePatchIDs"]
    ParamsDict["MNum"] = ResultsDict["MNum"]
    """

    MNumList.append(ParamsDict["MNum"])

    if ParamsDict["InactivePatchActivated"]:
        ZealotInvadedTime.append(t)
        MZealotInvadedTime.append(MNumList[-1])

    if t%PicTime == 0:
        ObserveDict["t"] = t
        ObserveDict["Graph"] = ParamsDict["Graph"]
        print("Saving for t=",t)

        #Observe(t,RandomGraph,positions,os.path.abspath(SaveDirName))
        Observe(ObserveDict)
        print("Proportion of Mutants:", MeasureMutants(ParamsDict["Graph"])/StatDict["GraphSize"]) 

#print(MNumList)

MNumList = np.asarray(MNumList)

Theory = (PNum/GraphSize) /(1-F)

print("GraphSize:",GraphSize)

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







endtime = time.time()

timetaken = endtime-starttime

print("Time Taken:",timetaken)
