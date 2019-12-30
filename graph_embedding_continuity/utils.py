import numpy as np
from copy import deepcopy
import scipy.sparse as sp
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


def normalize_sum_row_sparse(A):
    """
    Normalize a scipy.sparse.csr_matrix to have 
    each row sum to 1.
    """
    
    if not sp.isspmatrix_csr(A):
        raise ValueError('Input must be a sparse.scipy.csr_matrix.')
        
    row_sum = sp.csr_matrix(A.sum(axis=1))
    row_sum.data = 1/row_sum.data

    row_sum = row_sum.transpose()
    scaling_matrix = sp.diags(row_sum.toarray()[0])

    return scaling_matrix.dot(A)


def cnormalize(A, p=2):
    """
    Normalize a matrix A to have columns of unit L^p norm.
    """
    norms = np.linalg.norm(A, axis=0, ord=p)
    return A / norms