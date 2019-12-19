import numpy as np
import networkx as nx

###############################
####### 3-GRAPHLETS ##########
###############################

graphlets_3 = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
graphlets_3[0].add_nodes_from(range(3))

graphlets_3[1].add_nodes_from(range(3))
graphlets_3[1].add_edge(0,1)

graphlets_3[2].add_nodes_from(range(3))
graphlets_3[2].add_edge(0,1)
graphlets_3[2].add_edge(1,2)

graphlets_3[3].add_nodes_from(range(3))
graphlets_3[3].add_edge(0,1)
graphlets_3[3].add_edge(1,2)
graphlets_3[3].add_edge(0,2)
    
def shortest_path_feature_map(G):    
    """
    G: networkx graph
    Returns: a feature vector f such that f_i = number of shortest path 
    in the G of length i.
    """
    all_paths = dict()
    sp_counts = dict()
    
    sp_lengths = dict(nx.shortest_path_length(G))
    nodes = G.nodes()
    for v1 in nodes:
        for v2 in nodes:
            if v2 in sp_lengths[v1]:
                length = sp_lengths[v1][v2]
                if length in sp_counts:
                    sp_counts[length] += 1
                else:
                    sp_counts[length] = 1

                if length not in all_paths:
                    all_paths[length] = len(all_paths)
                        
    feature_map = np.zeros(len(all_paths))
    for length in sp_counts:
        feature_map[all_paths[length]] = sp_counts[length]
    
    # The first entry is the number of 0-length shortest paths i.e the number of nodes,
    # the following entries double count the number of shortest paths (for every path i->j, 7
    # the path j->i is also counted), therefore divide them by two. 
    feature_map[1:] /= 2
    
    return feature_map


def graphlet_feature_map(G, n_samples=0, k=3):
    """
    G: networkx graph,
    n_samples: number of sampled graphlets in G, defaults to k*number of nodes,
    k: int for k-graphlets (only k=3 supported for now),
    Returns: a feature vector f such that f_i = number of k-graphlets G_i in G.
    """

    if k == 3:
        graphlets = graphlets_3
    else:
        raise Exception('{}-graphles kernel not implemented.'.format(k))
        
    if n_samples == 0:
        n_samples = G.number_of_nodes() * k
        
    feature_map = np.zeros((len(graphlets)))
    
    for _ in range(n_samples):
        sample = G.subgraph(np.random.choice(G.nodes(), k, replace=False))
        for i, graphlet in enumerate(graphlets):
            if nx.is_isomorphic(graphlet, sample):
                feature_map[i] += 1
                break
    
    return feature_map