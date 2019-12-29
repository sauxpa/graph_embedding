import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def remove_random_edges(G, k=1, in_place=True):
    """
    G: networkx graph
    k: int, number of edges to remove
    in_place: bool, controls whether the graph is copied of modified in-place
    """
    if not in_place:
        G_ = deepcopy(G)
    else:
        G_ = G

    edges = np.array(G_.edges)
    remove_idx = np.random.choice(len(edges), size=k)
    for edge in edges[remove_idx]:
        G_.remove_edge(edge[0], edge[1])

    return G_


def wasserstein_metric(alpha, beta):
    """Compute 2-Wasserstein distance between point clouds representing
    empirical measures by solving the linear sum assignment problem in bipartite graphs.
    alpha, beta: (n,d) arrays:
        n: number of points in the cloud,
        d: dimension of the cloud.
    """
    
    if len(alpha.shape) == 1:
        # cdist needs inputs of shape of length 2
        dist_matrix = cdist(alpha.reshape(-1,1), beta.reshape(-1,1), metric='euclidean')
    else:
        dist_matrix = cdist(alpha, beta, metric='euclidean')
        
    assignment = linear_sum_assignment(dist_matrix)
    return dist_matrix[assignment].sum() / dist_matrix.shape[0]
