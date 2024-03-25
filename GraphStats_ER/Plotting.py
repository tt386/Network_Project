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

from colour import Color

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
parser.add_argument('-a','--all',type=int,help='Plot all of the sub-figures')
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


MeanList = []
MedianList = []
CList = []

MeanGraphSizeList = []
MeanDegreeList = []

AbsorbingStateProb = []

#MeanList as above but disregard cases where we reach the absorbing state/
MeanNoAbsorbList = []

#Mean of the mean cluster coefficients
MeanClusterCoeff = []

ComponentSizeList = np.zeros(1000)

ComponentNum = []

s1List = []
Clist = []

for d in dirlist:
    filelist = os.listdir(args.directory + "/" + d)
    for names in filelist:
        if names.endswith(".npz"):
            filename = names
            print("Found File!")

    with np.load(os.path.join(args.directory,d,filename)) as data:
        Repeats = data['Repeats']
        C = data["C"]
        nodenum = data['n']
        T = data['T']
        p = data['p']
        F = data['F']
        P = data['P']
        #MNumMatrix = data['MNumMatrix']
        GraphSizeList=data['GraphSizeList']
        #ZealotNumList=data['ZealotNumList']
        #deg_listList=data['deg_listList']
        #deg_cnt_listList=data["deg_cnt_listList"]
        MeanClusterCoeffList=data["MeanClusterCoeffList"]
        MeanDegree=data["MeanDegreeList"]
        Histogram = data["Histogram"]
        timetaken = data['timetaken']
        print("Time Taken:",timetaken)

    #Populate Component List:
    for i in Histogram:
        ComponentSizeList[int(i[0])-1] += i[1]


    Histx = Histogram[:,0]
    Histy = Histogram[:,1]

    ComponentNum.append(np.sum(Histy)/Repeats)

    Histy = Histy/Repeats#np.sum(Histy)

    s1List.append(Histy[0])
    Clist.append(C)

    def Prob(s,r):

        C = np.exp(-np.exp(7.21)*r**1.97)
        B = np.log(1/(1-C))#np.log(1+np.sqrt(C)) - np.log(1-C)
        A = C*np.exp(B)

        """
        A = 0.5*np.exp(-np.exp(7.21)*r**1.97)#(1-np.pi*r**2)**nodenum
        B = np.log((2+A+np.sqrt(2*A+A**2))/2)#np.log(1 + np.sqrt(A)) - np.log(1-A)
        """
        return A*np.exp(-B*(s))


    #HistTheory = Prob(Histx,radius)#A * np.exp(-B*(Histx-1))

    def NumTreesSize_s(N,s,C):

        """
        The number of trees of size s is T. Because of the large factorials
        we calculate log(T) and perform sums of log of the factorials. Then
        we convert back to T.
        """

        NList = np.arange(1,N +1)

        NsList = np.arange(1,N-s +1)

        sList = np.arange(1,s +1)

        
        term1 = 0#np.log(N)
        term2 = np.sum(np.log(NList))
        term3 = np.sum(np.log(NsList))
        term4 = np.sum(np.log(sList))
        term5 = (s-2)* np.log(s)
        term6 = (s-1) * np.log(C/N)
        term7 = (special.binom(s,2) - (s-1) + s*(N-s)) * np.log(1-C/N)

        tot = term1 + term2 - term3 - term4 + term5 + term6 + term7
        

        """
        term1 = np.log(2)
        term2 = (s-2)*np.log(s)
        term3 = (s-1)*np.log(C)
        term4 = -C*s
        term5 = -np.log(2-C)
        term6 = -np.sum(np.log(sList))

        tot = term1+term2+term3+term4+term5+term6
        """
        output = np.exp(tot)
        
        return output

    #HistTheory = nodenum*special.binom(nodenum,Histx) * Histx**(Histx-2) * (C/nodenum)**(Histx-1) * (1-C/nodenum)**(special.binom(Histx,2)-(Histx-1) * Histx*(nodenum-Histx))

    #HistTheory = nodenum * (Histx**(Histx-2)*C**(Histx-1) * np.exp(-C*Histx))/special.factorial(Histx) * np.exp(C*(Histx**2 + 3*Histx - 2)/(2*nodenum))

    HistTheory = []
    for s in Histx:
        HistTheory.append(NumTreesSize_s(nodenum, s,C))

    #HistTheory /= sum(HistTheory)

    plt.figure()
    plt.scatter(Histx,Histy)
    plt.plot(Histx,HistTheory)

    plt.xlabel("Component Size")
    plt.ylabel("Mean Number Per Graph")

    plt.ylim(min(Histy),nodenum)
    plt.xlim(min(Histx),max(Histx))

    plt.title("C: %0.5f"%(C))

    plt.savefig(args.directory + "/" + d + "/ComponentHistList.png")
    plt.savefig(args.directory + "/ComponentHistList_C_%0.5f.png"%(C))

    plt.yscale('log')
    plt.savefig(args.directory + "/LogComponentHistList_C_%0.5f.png"%(C))

    plt.close()

print(ComponentSizeList)

Clist,s1List,ComponentNum = zip(*sorted(zip(Clist,s1List,ComponentNum)))

s1List = np.asarray(s1List)
Clist = np.asarray(Clist)

"""
fitx = np.log(rlist[rlist<1e-2])
fity = np.log(abs(np.log(s1List[rlist<1e-2])))

fitx =np.delete(fitx,0)
fity = np.delete(fity,0)

print(len(fitx))
print(len(fity))


MULTFACTOR = 1000
slope, intercept, r, p, se = linregress(MULTFACTOR*fitx, MULTFACTOR*fity)

result = linregress(MULTFACTOR*fitx,MULTFACTOR*fity)
print(result.intercept,result.intercept_stderr)

intercept /= MULTFACTOR

print("For s1:",slope, intercept, r, p, se)

"""
plt.figure()
"""
plt.plot(rlist,abs(np.log(s1List)))

plt.plot(rlist,np.exp(intercept) * rlist**2)
"""

plt.scatter(Clist, s1List)
#plt.plot(rlist, np.exp(-np.exp(intercept)*rlist**slope))

plt.xlabel("C")
plt.ylabel("Proportion")

plt.savefig(args.directory + "/ProportionSize1.png")


plt.close()

"""
r = rlist[rlist<1e-2]
r =np.delete(r,0)

print("initial value is (log(r),logv) = ",np.log(r[0]),np.log(abs(np.log(s1List))[1]))
print("second value is (log(r),logv) = ",np.log(r[1]),np.log(abs(np.log(s1List))[2]))


plt.figure()
plt.scatter(rlist,abs(np.log(s1List)))
intercept = 7.36
plt.plot(r,np.exp(intercept) * r**2)
plt.yscale('log')
plt.xscale('log')
plt.savefig(args.directory + "/ProportionSize1_LOG.png"%(radius))

plt.close()


"""
"""
fitx = np.log(rlist[rlist<2e-2])
fity = np.log(abs(np.log(s1List[rlist<2e-2])))

fitx =np.delete(fitx,0)
fity = np.delete(fity,0)
"""


###################################################################


ComponentRatio = ComponentNum/nodenum

"""
cfitx = np.log(rlist[rlist<2e-2])
cfity = np.log(abs(np.log(ComponentRatio[rlist<2e-2])))

cfitx =np.delete(cfitx,0)
cfity = np.delete(cfity,0)

slope, intercept, r, p, se = linregress(cfitx, cfity)

print(slope, intercept, r, p, se)
"""


plt.figure()


"""
plt.scatter(np.pi*nodenum*rlist**2,abs(np.log(ComponentRatio)))

plt.plot(np.pi*nodenum*rlist**2, np.exp(intercept) * rlist**2)#np.exp(-np.exp(intercept)*rlist**slope))

plt.xlabel("Mean Neighbour Num")
plt.ylabel("abs(log(Component Ratio))")
"""
plt.scatter(Clist,ComponentNum)

#plt.plot(rlist, nodenum * np.exp(-np.exp(intercept) * rlist**2))

plt.xlabel("C")
plt.ylabel("Component Number")


#plt.yscale('log')
#plt.xscale('log')

plt.savefig(args.directory + "/MeanComponentNumber.png")
plt.yscale('log')
plt.xscale('log')
plt.savefig(args.directory + "/MeanComponentNumber_LOG.png")
plt.close()
