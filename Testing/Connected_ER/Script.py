from pylab import *
import networkx as nx

import random

import copy

import matplotlib.pyplot as plt

import time


nodes = 1000

p = 0.001

"""
#Regular for comparison
RandomGraph = nx.gnp_random_graph(nodes,p,seed=None,directed=False)
positions = nx.spring_layout(RandomGraph)

#####
fig = plt.figure()
ax = plt.subplot(111)
ax.set_title('Erdos-Renyi')

nx.draw(RandomGraph, positions)
plt.tight_layout()
plt.savefig("RandomGraph.png")
plt.close()
#####
"""
#Now create my version
EmptyGraph = nx.empty_graph(nodes)



nodelist = EmptyGraph.nodes()
print(nodelist)
EmptyGraph.add_edge(0,nodes-1)
for i in range(nodes-1):
    print(i)
    EmptyGraph.add_edge(i,i+1)

Edges = EmptyGraph.edges()
print(Edges)


positions = nx.spring_layout(EmptyGraph)
#####
fig = plt.figure()
ax = plt.subplot(111)
ax.set_title('Loop')

nx.draw(EmptyGraph, positions)
plt.tight_layout()
plt.savefig("Loop.png")
plt.close()
#####
    

#Add random edges

for i in range(nodes):
    for j in range(i+2,nodes):
        if (i,j) not in Edges:
            if random.uniform(0,1) < p:
                EmptyGraph.add_edge(i,j)

#####
fig = plt.figure()
ax = plt.subplot(111)
ax.set_title('My Erdos Renyi')

nx.draw(EmptyGraph, positions)
plt.tight_layout()
plt.savefig("MyErdos.png")
plt.close()
#####

print(EmptyGraph.edges())





#Regular for comparison
RandomGraph = nx.gnp_random_graph(nodes,p,seed=None,directed=False)
#positions = nx.spring_layout(RandomGraph)

#####
fig = plt.figure()
ax = plt.subplot(111)
ax.set_title('Erdos-Renyi')

nx.draw(RandomGraph, positions)
plt.tight_layout()
plt.savefig("RandomGraph.png")
plt.close()
#####
