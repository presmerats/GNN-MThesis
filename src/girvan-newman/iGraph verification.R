library(igraph)


G = read_graph('verifi_edge_list',format=c("edgelist"))
nb = betweenness(G, v=V(G), directed = FALSE, normalized=TRUE)
b = betweenness(G, v=V(G), directed = FALSE, normalized=FALSE)
neb = edge_betweenness(G, directed=FALSE)

nb[1:10]
b[1:10]
neb[1:10]

# write to disk the normalized betweeness list
write(nb,file="igraph_r_betweenness.txt", ncolumns=1)
