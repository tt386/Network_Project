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
        Mnum = data['Mnum']
        MeanN = data['MeanN']
        MedianN = data['MedianN']
        timetaken = data['timetaken']

    FList.append(F)
    PNumList.append(PNum)
    FinalMean.append(MeanN[-1])

    #Plotting
    plt.figure()
    for i in range(len(Mnum)):
        plt.plot(np.arange(T),Mnum[i],linewidth=0.5)

    plt.plot(np.arange(T),np.ones(T)*min((PNum/nodenum)/(1-F),1),'blue',linewidth=5)

    plt.plot(np.arange(T),MeanN,'k',linewidth=3,label="Mean")
    plt.plot(np.arange(T),MedianN,'y',linewidth=3,label="Median")

    plt.legend(loc='lower right')

    plt.savefig(args.directory + "/MRatio_Complete_PNum_%d_F_%0.3f.png"%(PNum,F))
    plt.close()



F = np.asarray(FList)
N = np.asarray(PNumList/nodenum)
Mean = np.asarray(FinalMean)

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
