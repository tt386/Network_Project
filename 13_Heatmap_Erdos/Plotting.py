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

import copy

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
PList = []
FinalMean = []
FinalMedian = []
Theory  =[]
for d in dirlist:
    filelist = os.listdir(args.directory + "/" + d)
    for names in filelist:
        if names.endswith(".npz"):
            filename = names
            print("Found File!")
    try:
        with np.load(os.path.join(args.directory,d,filename)) as data:
            Repeats = data['Repeats']
            nodenum = data['n']
            T = data['T']
            p = data['p']
            P = data['P']
            F = data['F']
            PNum = data['PNum']
            Mnum = np.asarray(data['MPropMatrix'])
            MeanN = np.mean(Mnum,axis=0)
            MedianN = np.median(Mnum,axis=0)
            timetaken = data['timetaken']

        print("Plotting P,F",P,F)
        print("Timetaken:",timetaken)
        FList.append(F)
        PList.append(P)
        FinalMean.append(MeanN[-1])
        FinalMedian.append(MedianN[-1])
        Theory.append(min(P/(1-F),1))
        
        #Plotting
        plt.figure()
        for i in range(len(Mnum)):
            plt.loglog(np.arange(T),Mnum[i],linewidth=0.5)

        plt.loglog(np.arange(T),np.ones(T)*min((P)/(1-F),1),'blue',linewidth=5)

        plt.loglog(np.arange(T),MeanN,'k',linewidth=3,label="Mean")
        plt.loglog(np.arange(T),MedianN,'y',linewidth=3,label="Median")

        plt.legend(loc='lower right')

        plt.savefig(args.directory + "/MRatio_P_%0.3f_F_%0.3f.png"%(P,F))
        plt.close()
        

    except:
        print("Error with file")

F = np.asarray(FList)
N = np.asarray(PList)
Mean = np.asarray(FinalMean)
Median = np.asarray(FinalMedian)
Theory = np.asarray(Theory)

print("F",F)
print("N",N)

print("Starting mean heatmap")

#cp = ax.tricontourf(N.ravel(), F.ravel(), Mean.ravel(),10,cmap='coolwarm')

SortedFList = list(set(F))
SortedNList = list(set(N))

SortedFList.sort()
SortedNList.sort()

print("SortedF",SortedFList)
print("SortedN",SortedNList)

dF = (SortedFList[1] - SortedFList[0])/2
dN = (SortedNList[1] - SortedNList[0])/2

MeanMatrix = np.zeros((len(SortedFList),len(SortedNList)))
MedianMatrix = copy.deepcopy(MeanMatrix)
TheoryMatrix = copy.deepcopy(MeanMatrix)

for row in range(len(MeanMatrix)):
    for col in range(len(MeanMatrix[row])):
        targetN = SortedNList[col]
        targetF = SortedFList[row]

        for i in range(len(Mean)):
            if (abs(F[i] - targetF)<dF) and (abs(N[i]-targetN)<dN):
                MeanMatrix[row][col] = Mean[i]
                MedianMatrix[row][col] = Median[i]
                TheoryMatrix[row][col] = min(P/(1-targetF),1)
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
plt.close()

#####################################
#Median
fig=plt.figure(2)
ax = fig.add_subplot(1,1,1)
cp = ax.tricontourf(N.ravel(), F.ravel(), Median.ravel(),10,cmap='coolwarm')
ax.pcolor(np.asarray(SortedNList)-dN,np.asarray(SortedFList)-dF,np.array(MedianMatrix),cmap='coolwarm')

TheoryNList = np.linspace(min(N),max(N),100)
plt.plot(TheoryNList,1-TheoryNList,'k',linewidth=5)

ax.set_xticks(np.asarray(SortedNList).round(decimals=3))
plt.xticks(rotation=45)
ax.set_yticks(SortedFList)

ax.set_ylabel(r'mutant fitness $F$',fontsize=15)
ax.set_xlabel(r'patch ratio',fontsize=15)

cbar = fig.colorbar(cp)
cbar.set_label(label=r'Endstate Median')
cbar.ax.tick_params(labelsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)

plt.title("NodeNum: %d, Time: %d"%(nodenum,T),fontsize=20)


plt.tight_layout()
#plt.grid(True)

plt.savefig(str(args.directory) + '/Final_Median.png')
plt.close()


#####################################
#Make Theory Matrix
fig=plt.figure(2)
ax = fig.add_subplot(1,1,1)
cp = ax.tricontourf(N.ravel(), F.ravel(), Mean.ravel(),10,cmap='coolwarm')
ax.pcolor(np.asarray(SortedNList)-dN,np.asarray(SortedFList)-dF,np.array(TheoryMatrix),cmap='coolwarm')

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

plt.savefig(str(args.directory) + '/Theory_Mean.png')
plt.close()

#####################################
#Make Difference between theory and data
DiffMatrix = MeanMatrix - TheoryMatrix

fig=plt.figure(2)
ax = fig.add_subplot(1,1,1)
cp = ax.tricontourf(N.ravel(), F.ravel(), (Mean-Theory).ravel(),10,cmap='coolwarm')
ax.pcolor(np.asarray(SortedNList)-dN,np.asarray(SortedFList)-dF,np.array(DiffMatrix),cmap='coolwarm')

TheoryNList = np.linspace(min(N),max(N),100)
plt.plot(TheoryNList,1-TheoryNList,'k',linewidth=5)

ax.set_xticks(np.asarray(SortedNList).round(decimals=3))
plt.xticks(rotation=45)
ax.set_yticks(SortedFList)

ax.set_ylabel(r'mutant fitness $F$',fontsize=15)
ax.set_xlabel(r'patch ratio',fontsize=15)

cbar = fig.colorbar(cp)
cbar.set_label(label=r'Mean-Theory')
cbar.ax.tick_params(labelsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)

plt.title("NodeNum: %d, Time: %d"%(nodenum,T),fontsize=20)


plt.tight_layout()
#plt.grid(True)

plt.savefig(str(args.directory) + '/MeanMinusTheory_Mean.png')

