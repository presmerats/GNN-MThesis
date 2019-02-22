# local python tests
from pprint import pprint
from igraph import *
import igraph.test




#igraph.test.run_tests()

# generation
g = Graph.Tree(127, 2)
g = Graph.GRG(100, 0.2)
g = Graph.Erdos_Renyi(100, 0.2)
g = Graph.Watts_Strogatz(1, 100, 4, 0.5 )
g = Graph.Barabasi(100 )
summary(g)
layout = g.layout("large")
layout = g.layout("kk")
plot(g, layout = layout)

# graph metrics
pprint(g.degree([2,3,6,99]))
pprint(g.edge_betweenness())
pprint(g.pagerank())
pprint(g.get_adjacency())
pprint(dir(g))