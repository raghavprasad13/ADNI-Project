source("collective_influence_algorithm.R")

to_install <- c("optparse", "reticulate")
library(reticulate)
np <-import("numpy")

# Hard coded path - need to change to use path from args

mat <- np$load("/home/pranav/Graph_Alz/Preprocessed_Data/AD/AV45/002_S_5018~2012-11-15_16_29_51.0~I347148/adj_mat.npy")
adj_mat = (mat >=0.4)

pet_graph <- graph_from_adjacency_matrix( adj_mat )
pet_graph <- set_vertex_attr(pet_graph, "name", value=paste("n", as.vector(V(pet_graph)), sep=""))

# find influencers, remove them, and plot the results after each removal
g <- getInfluencers(pet_graph, d=2)
g

