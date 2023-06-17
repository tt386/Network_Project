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

for d in dirlist:
    try:
        filelist = os.listdir(args.directory + "/" + d)
        for names in filelist:
            if names.endswith(".npz"):
                filename = names
                print("Found File!")

        with np.load(os.path.join(args.directory,d,filename)) as data:
            Repeats = data['Repeats']
            nodenum = data['n']
            T = data['T']
            C = data['r']
            p = data['p']
            F = data['F']
            P = data['P']
            MNumMatrix = data['MNumMatrix']
            GraphSizeList=data['GraphSizeList']
            ZealotNumList=data['ZealotNumList']
            #deg_listList=data['deg_listList']
            #deg_cnt_listList=data["deg_cnt_listList"]
            MeanClusterCoeffList=data["MeanClusterCoeffList"]
            MeanDegree=data["MeanDegreeList"]

            timetaken = data['timetaken']
            print("Time Taken:",timetaken)

        MeanGraphSizeList.append(np.mean(GraphSizeList))
        MeanDegreeList.append(np.mean(MeanDegree))


        #Create Mean List
        """
        MeanMNum = np.mean(MNumMatrix,axis = 0)
        MedianMNum = np.median(MNumMatrix,axis = 0)

        MeanList.append(MeanMNum/np.mean(GraphSizeList))
        MedianList.append(MedianMNum/np.median(GraphSizeList))
        """
        Ratio = np.divide(MNumMatrix,GraphSizeList[:,np.newaxis])

        if args.all:
            MeanList = []
            MedianList = []
            MeanNoAbsorbList = []

        MeanList.append(np.mean(Ratio,axis=0))
        MedianList.append(np.median(Ratio,axis=0))

        #Mean Cluster Coeff
        MeanClusterCoeff.append(np.mean(MeanClusterCoeffList))

        #Absorbing State
        AbsorbedNum = 0
        for i in Ratio:
            if i[-1] == 1:
                AbsorbedNum += 1
        AbsorbingStateProb.append(AbsorbedNum/len(Ratio))

        masked_matrix = np.where(Ratio == 1, np.nan,Ratio)
        MeanNoAbsorbList.append(np.nanmean(masked_matrix,axis = 0))

        CList.append(C)

        CList = [float(value) if isinstance(value, np.ndarray) else value for value in CList]
        MeanList = [list(sublist) if isinstance(sublist, np.ndarray) else sublist for sublist in MeanList]
        MedianList = [list(sublist) if isinstance(sublist, np.ndarray) else sublist for sublist in MedianList]

        #print(CList)
        #print(MeanList)

        #Plot each repeat for a C:
        print("Plot C = ",C)
        if args.all:
            print(args.all)

            #Figure for the number of mutants at each time point
            x = np.arange(len(MNumMatrix[0]))#MeanMNum))

            plt.figure()
            for i in range(len(MNumMatrix)):
                plt.plot(x,MNumMatrix[i]/GraphSizeList[i])

            plt.plot(x,MeanList[-1],color='black',linewidth=5)

            plt.plot(x,MeanNoAbsorbList[-1],color='orange',linewidth=5,label='No Absorb')

            Theory = P/(1-F)

            plt.legend(loc='upper left')

            plt.plot([min(x),max(x)],[Theory,Theory],'--r',linewidth=5)

            plt.title("C=%0.3f"%(C))
            plt.savefig(str(args.directory) +'/'+str(d) +'/AllRepeats.png' )
            plt.savefig(str(args.directory) +'/C_%0.3f_AllRepeats.png'%(C) )

            plt.close()


    except:
        print("Error with",d)

        

if not args.all:
    print("Finished all sub-plots")
    CList,MeanList,MedianList,MeanGraphSizeList,MeanDegreeList,AbsorbingStateProb,MeanNoAbsorbListi,MeanClusterCoeff = zip(*sorted(zip(CList, MeanList, MedianList,MeanGraphSizeList,MeanDegreeList,AbsorbingStateProb,MeanNoAbsorbList,MeanClusterCoeff)))

    if args.all:
        plt.figure()
        print("Plotting All Means")
        x = np.arange(len(MeanList[0]))

        def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
            c1=np.array(mpl.colors.to_rgb(c1))
            c2=np.array(mpl.colors.to_rgb(c2))
            return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

        c1='#1f77b4' #blue
        c2='red' #green

        for i in range(len(CList)):
            plt.plot(
                    x,
                    MeanList[i],
                    color=colorFader(c1,c2,i/len(CList)),label=str(CList[i]),
                    linewidth = 0.5,
                    alpha = 0.5)

        Theory = P/(1-F)

        plt.plot(
                [min(x),max(x)],
                [Theory,Theory],
                '--k',
                linewidth=5,
                label='Theory',
                alpha = 0.5)

        

        plt.savefig(str(args.directory) +'/AllMeans.png')
        plt.close()

        ##############################
        plt.figure()
        print("Plotting all Medians")
        x = np.arange(len(MedianList[0]))

        def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
            c1=np.array(mpl.colors.to_rgb(c1))
            c2=np.array(mpl.colors.to_rgb(c2))
            return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

        c1='#1f77b4' #blue
        c2='red' #green

        for i in range(len(CList)):
            plt.plot(
                    x,
                    MedianList[i],
                    color=colorFader(c1,c2,i/len(CList)),label=str(CList[i]),
                    linewidth = 0.5,
                    alpha = 0.5)

        Theory = P/(1-F)

        plt.plot(
                [min(x),max(x)],
                [Theory,Theory],
                '--k',
                linewidth=5,
                label='Theory',
                alpha = 0.5)

        plt.savefig(str(args.directory) +'/AllMedians.png')
        plt.close()


    ##############################
    Theory = P/(1-F)
    print("Plotting Ednstates")

    plt.figure()
    #Plot how the end-state mean evolves with C
    EndMean = []
    EndMedian = []

    EndNoAbsorbMean = []
    for i in range(len(MeanList)):
        EndMean.append(np.mean(MeanList[i][-int(T/10):]))
        EndMedian.append(np.mean(MedianList[i][-int(T/10):]))
        EndNoAbsorbMean.append(np.mean(MeanNoAbsorbList[i][-int(T/10):]))

    plt.plot(CList,EndMean, label='ER Mean Endstate')
    plt.plot(CList,EndMedian, label='ER Median Endstate')
    #plt.plot(CList,EndNoAbsorbMean, label='ER Mean Endstate No Absorb')


    plt.plot(
            [min(CList),max(CList)],
            [Theory,Theory],
            '--k',
            linewidth=5,
            label='Complete Theory',
            alpha = 0.5)

    plt.title("Fitness %0.3f, Zealot Ratio %0.3f"%(F,P))
    plt.xlabel("r")
    plt.ylabel("EndState Ratio of M")

    plt.legend(loc='upper right')

    plt.savefig(str(args.directory) +'/EndMeanWithC.png')
    plt.close()


    #Plot how Mean Degree Changes with C
    plt.figure()
    plt.plot(CList,MeanGraphSizeList)

    plt.title("Fitness %0.3f, Zealot Ratio %0.3f"%(F,P))
    plt.xlabel("r")
    plt.ylabel("Mean Graph Size")

    plt.savefig(str(args.directory) + '/MeanGraphSize.png')
    plt.close()



    #Plot how Mean degree changes with C
    plt.figure()
    plt.plot(CList,CList,'--k',linewidth=5)

    plt.plot(CList,MeanDegreeList)

    plt.title("Fitness %0.3f, Zealot Ratio %0.3f"%(F,P))
    plt.xlabel("r")
    plt.ylabel("Mean Degree")

    plt.grid(True)

    plt.savefig(str(args.directory) + '/MeanDegree.png')
    plt.close()



    ##############################
    plt.figure()
    plt.semilogy(CList,AbsorbingStateProb)

    plt.title("Fitness %0.3f, Zealot Ratio %0.3f"%(F,P))
    plt.xlabel("r")
    plt.ylabel("Prob of Absorbing State")

    plt.savefig(str(args.directory) + '/AbsorbingStateProb.png')
    plt.close()





    plt.figure()
    plt.plot(CList,MeanClusterCoeff)
    plt.title("Fitness %0.3f, Zealot Ratio %0.3f"%(F,P))
    plt.xlabel("r")
    plt.ylabel("Mean Clustering Coefficient")

    plt.grid("True")

    plt.savefig(str(args.directory) + '/MeanClusteringCoeff.png')
    plt.close()
