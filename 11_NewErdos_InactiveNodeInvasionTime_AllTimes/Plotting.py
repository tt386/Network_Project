import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as patches
#plt.rcParams['text.usetex'] = True

import numpy as np
import time

from scipy.stats import linregress

from scipy.optimize import curve_fit

import sys

import scipy.integrate as integrate
import scipy.special as special

from scipy.signal import argrelextrema

from scipy.signal import savgol_filter

from itertools import chain

starttime = time.time()
################################
##ArgParse######################
################################
import os.path

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The Directory %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

from argparse import ArgumentParser

parser = ArgumentParser(description='Plotting')
parser.add_argument('-d','--directory',help='The directory of the data')
args = parser.parse_args()

###############################
##Extract Data#################
###############################
#Find all the directories
templist = os.listdir(args.directory)
print(templist)

dirlist = []

for i in range(len(templist)):
    if os.path.isdir(args.directory + '/' + templist[i]):
        print("Is directory!")
        npzlist = os.listdir(args.directory + '/' + templist[i])
        for j in range(len(npzlist)):
            if npzlist[j].endswith(".npz"):
                dirlist.append(templist[i])

FList = []
PNumList = []
FinalMean = []

DataList = []
for d in dirlist:
    filelist = os.listdir(args.directory + "/" + d)
    for names in filelist:
        if names.endswith(".npz"):
            filename = names
            print("Found File!")

    with np.load(os.path.join(args.directory,d,filename)) as data:
        Repeats = data['Repeats']
        nodenum = data['n']
        T = data['T']
        p = data['p']
        F = data['F']
        PNum = data['PNum']
        TimeMatrix = np.array(data["TimeMatrix"])

        timetaken = data['timetaken']
        print("Time Taken:",timetaken)

    print(PNum,TimeMatrix)
    #Find mean, taking into account that there may be empty elements due to largest element of graph too small.
    MeanList = np.true_divide(TimeMatrix.sum(0),(TimeMatrix!=0).sum(0))#np.mean(TimeMatrix, axis=0)
    MedianList =np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, TimeMatrix)
    #MedianList[np.isnan(MedianList)]=0#np.median(TimeMatrix, axis=0)

    print(MeanList)

    plt.figure()
    #plt.xscale('log')

    """
    for i in range(len(TimeMatrix[0])):
        InvadedNum = np.ones(len(TimeMatrix)) * (i+1)
        Times = TimeMatrix[:,i]

        plt.scatter(InvadedNum,Times)
    """
    for i in range(len(TimeMatrix)):
        InvadedNum = np.arange(len(TimeMatrix[i]))+1#np.ones(len(TimeMatrix[i])) * (i+1)
        Times = TimeMatrix[i,:]
        plt.plot(InvadedNum,Times)    


    plt.plot(np.arange(len(MeanList))+1,MeanList,"k",linewidth=5,label='Mean')
    plt.plot(np.arange(len(MedianList))+1,MedianList,"--k",linewidth=5,label='Median')

    plt.xscale('log')

    plt.legend(loc='upper left')

    plt.xlabel("Patch")
    plt.ylabel("Time Patch Invaded")

    plt.savefig(args.directory + "/" + d + "/Scatter.png")

    plt.close()



    ############
    try:
        FirstTimeGap = []
        for i in range(len(TimeMatrix)):
            FirstTimeGap.append(TimeMatrix[i][1] -TimeMatrix[i][0])


        plt.figure()
        plt.hist(FirstTimeGap)#,bins=100)
        plt.yscale('log')
        plt.savefig(args.directory + "/" + d + "/DistBetweenFirstInvasions.png")
        plt.close()
    except:
        print("Whoops for Patch num:",PNum)


    X = np.log(np.arange(len(MeanList))+1)
    
    linearregression = linregress(X[X<len(X)/2], MeanList[X<len(X)/2])
    slope = linearregression[0]
    intercept = linearregression[1]

    intercept = MedianList[0]

    print("Slope:",slope)

    DataList.append([PNum,MeanList[-1],MedianList[-1],MeanList[int(len(MeanList)/2)],MedianList[int(len(MedianList)/2)],slope,intercept])

    """
    bins = np.arange(0,max(TimeList),1000)


    plt.figure()
    plt.hist(TimeList,bins=bins)
    plt.xlabel("Time taken for all patches invaded")
    plt.ylabel("Freq")
    plt.savefig(args.directory + "/" + d + "/Hist.png")
    plt.close()
    """


DataList = sorted(DataList, key=lambda x: x[0])

DataList = np.asarray(DataList)

plt.figure()
plt.semilogx(DataList[:,0],DataList[:,1],label="Mean")
plt.semilogx(DataList[:,0],DataList[:,2],label="Median")
plt.legend(loc='upper right')
plt.xlabel("Number of Patches")
plt.ylabel("Time taken for all Patches to be invaded")
plt.savefig(str(args.directory)+"Time.png")
plt.close()


plt.figure()
plt.semilogx(DataList[:,0],DataList[:,3],label="Mean")
plt.semilogx(DataList[:,0],DataList[:,4],label="Median")
plt.legend(loc='upper right')
plt.xlabel("Number of Patches")
plt.ylabel("Time taken for 50 Percent Patches to be invaded")
plt.savefig(str(args.directory)+"Time50Percent.png")
plt.close()


plt.figure()
plt.loglog(DataList[:,0],DataList[:,5],label='slope')
plt.loglog(DataList[:,0],DataList[:,6],label='intercept')
plt.legend(loc='upper right')
plt.xlabel("Number of Patches")
plt.ylabel("Slope")
plt.savefig(str(args.directory)+"Slope.png")
plt.close()


print("Ratio of 1st intercept to second:",DataList[:,6][0]/DataList[:,6][1])


"""
        Mnum = data['Mnum']
        MeanN = data['MeanN']
        MedianN = data['MedianN']
        MSepDist = data['Random_MSepDist']
        DataMatrix = data['DataMatrix']
        timetaken = data['timetaken']

    FList.append(F)
    PNumList.append(PNum)
    FinalMean.append(MeanN[-1])

    #print("MSepDist",MSepDist)

    print("Starting Histlist")
    transposed = np.transpose(MSepDist)
    #print("Transposed", transposed)

    #Creating snapshots of the total of all repeats with a snapshot
    HistList = []
    b = [10-1,100-1,1000-1,10000-1,100000-1,1000000-1]
    for i in b:
        if i < len(HistList):
            HistList.append(list(chain.from_iterable(transposed[i])))
    

    print("Starting History")
    #Creating a visualisation of the entire history with single histogram
    History_HistList = []
    for i in MSepDist:
        History_HistList.append(list(chain.from_iterable(i)))


    #print(HistList)


    #Plotting Trajectories
    plt.figure()
    for i in range(len(Mnum)):
        plt.plot(np.arange(T),Mnum[i],linewidth=0.5)

    plt.plot(np.arange(T),np.ones(T)*min((PNum/nodenum)/(1-F),1),'blue',linewidth=5)

    plt.plot(np.arange(T),MeanN,'k',linewidth=3,label="Mean")
    plt.plot(np.arange(T),MedianN,'y',linewidth=3,label="Median")

    plt.legend(loc='lower right')

    plt.savefig(args.directory + "/MRatio_Complete_PNum_%d_F_%0.3f.png"%(PNum,F))
    plt.close()



    #Plotting histograms:
    print(F)
    b = [10-1,100-1,1000-1,10000-1,100000-1,1000000-1]


    for i in range(len(HistList)):
        #if i in b:
        plt.figure()
        
        if len(HistList[i]) > 0:
            bins = np.arange(0.5,max(HistList[i])+0.5,1)

            plt.hist(HistList[i],bins=bins)

        else:
            plt.hist(HistList[i])

        plt.savefig(args.directory + "/" + d + "/Hist_t_%d.png"%(i))
        plt.close()





    for i in range(len(History_HistList)):
        #if i in b:
        plt.figure()

        if len(History_HistList[i]) > 0:
            bins = np.arange(0.5,max(History_HistList[i])+0.5,1)

            plt.hist(History_HistList[i],bins=bins)

        else:
            plt.hist(History_HistList[i])

        plt.savefig(args.directory + "/" + d + "/TotalHistory_Repeat_%d.png"%(i))
        plt.close()



    #########################################
    #########################################
    #########################################
    #Create a plot of the sep and time of first invasion
    print("Create Scatters")
    SepList = []
    TimeList = []

    TotalSepList = np.arange(10)+1
    TotalTimeList = []

    for i in TotalSepList:
        TotalTimeList.append([])


    for i in range(len(DataMatrix)):
        SepList = []
        TimeList = []

        DataList = DataMatrix[i]
        for j in range(len(DataList)):
            try:
                Sep = DataList[j][0]
                Time = DataList[j][2][0]
                
                TotalTimeList[Sep-1].append(Time)

                SepList.append(Sep)
                TimeList.append(Time)
            except Exception as e:
                l = 1
                #print("No time here")

        plt.figure()
        plt.scatter(SepList,TimeList)

        plt.savefig(args.directory + "/" + d + "/SepTimeScatter_Repeat_%d.png"%(i))
        plt.close()



    #Distributuon of the times of invasion for each Node distance
    for S in range(len(TotalSepList)):
        Sep = TotalSepList[S]
        Dist = TotalTimeList[S]

        plt.figure()
        plt.hist(Dist)
        plt.title("Separation: %d"%(Sep))
        plt.xlabel("Time invaded")
        plt.ylabel("Frequency")
        plt.savefig(args.directory + "/" + d + "/InvasionTimeDist_Sep_%d.png"%(Sep))
        plt.close()
        





    MeanList = []
    MedianList = []
    for i in TotalTimeList:
        MeanList.append(np.mean(i))
        MedianList.append(np.median(i))


    xlist = np.arange(len(MeanList)) + 1
    plt.figure()
    plt.plot(xlist,MeanList,label = 'mean')
    plt.plot(xlist,MedianList,label='median')
    plt.title("F: %0.3f"%(F))
    plt.xlabel("Distance from Zealot")
    plt.ylabel("Time taken for M to reach")
    plt.legend(loc='upper left')
    plt.savefig(args.directory + "/" + d + "/SepTimeMean.png")
    plt.savefig(args.directory + "/SepTimeMean_F_%0.3f.png"%(F))
    plt.close()


F = np.asarray(FList)
N = np.asarray(PNumList/nodenum)
Mean = np.asarray(FinalMean)
"""
"""
fig,ax=plt.subplots(1,1)
cp = ax.tricontourf(N.ravel(), F.ravel(), Mean.ravel(),10,cmap='coolwarm')

cbar = fig.colorbar(cp,label='Final Mean')
cbar.ax.tick_params(labelsize=20)

plt.title("NodeNum: %d, Time: %d"%(nodenum,T))

ax.set_ylabel('Mutant Fitness F',fontsize=15)

ax.set_xlabel('Number of Patches',fontsize=15)

ax.tick_params(axis='both', which='major', labelsize=20)

plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.tight_layout()
plt.grid(True)

plt.savefig(str(args.directory) + '/Final_Mean.png')

plt.close()
"""


"""
print("Starting mean heatmap")

#cp = ax.tricontourf(N.ravel(), F.ravel(), Mean.ravel(),10,cmap='coolwarm')

SortedFList = list(set(F))
SortedNList = list(set(N))

SortedFList.sort()
SortedNList.sort()

dF = (SortedFList[1] - SortedFList[0])/2
dN = (SortedNList[1] - SortedNList[0])/2

MeanMatrix = np.zeros((len(SortedFList),len(SortedNList)))

for row in range(len(MeanMatrix)):
    for col in range(len(MeanMatrix[row])):
        targetN = SortedNList[col]
        targetF = SortedFList[row]

        for i in range(len(Mean)):
            if (abs(F[i] - targetF)<dF) and (abs(N[i]-targetN)<dN):
                MeanMatrix[row][col] = Mean[i]
                break

SortedFList.append(SortedFList[-1]*2 - SortedFList[-2])
SortedNList.append(SortedNList[-1]*2 - SortedNList[-2])


fig=plt.figure(2)
ax = fig.add_subplot(1,1,1)
cp = ax.tricontourf(N.ravel(), F.ravel(), Mean.ravel(),10,cmap='coolwarm')
ax.pcolor(np.asarray(SortedNList)-dN,np.asarray(SortedFList)-dF,np.array(MeanMatrix),cmap='coolwarm')

TheoryNList = np.linspace(min(N),max(N),100)
plt.plot(TheoryNList,1-TheoryNList,'k',linewidth=5)

ax.set_xticks(np.asarray(SortedNList).round(decimals=3))
plt.xticks(rotation=45)
ax.set_yticks(SortedFList)

ax.set_ylabel(r'mutant fitness $F$',fontsize=15)
ax.set_xlabel(r'patch ratio',fontsize=15)

cbar = fig.colorbar(cp)
cbar.set_label(label=r'Endstate Mean')
cbar.ax.tick_params(labelsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)

plt.title("NodeNum: %d, Time: %d"%(nodenum,T),fontsize=20)


plt.tight_layout()
#plt.grid(True)

plt.savefig(str(args.directory) + '/Final_Mean.png')
"""
