from Params import *

from pylab import *
import networkx as nx

import random

#Create Random Graph
RandomGraph = nx.gnp_random_graph(n, p, seed=None, directed=False)

#Set the actual physical pixel positions of the nodes
positions = nx.spring_layout(RandomGraph)

#Set the values of the sites
States = []
for i in range(n):
    if random.uniform(0,1) < p:
        States.append(1)
    else:
        States.append(0)
nx.set_node_attributes(RandomGraph, States, "states")

nx.draw(RandomGraph,      #Graph
        positions,  #Defined positions
        node_color=['blue' if i==1 else 'red' for i in States]
        )

savefig(SaveDirName+"/Random.png")
close()

#Create the same for random geometric
GeoGraph = nx.random_geometric_graph(n, R, dim=2, pos=positions, p=2, seed=None)

nx.draw(GeoGraph,      #Graph
        positions,  #Defined positions
        node_color=['blue' if i==1 else 'red' for i in States]
        )

savefig(SaveDirName+"/Geometric.png")
close()


print(positions)
print(RandomGraph.nodes())
