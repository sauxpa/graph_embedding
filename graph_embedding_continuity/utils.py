import numpy as np
import torch
from copy import deepcopy
import scipy
import scipy.sparse as sp
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
        alpha = alpha.reshape(-1,1)
        beta = beta.reshape(-1,1)
        
    if isinstance(alpha, torch.Tensor) and isinstance(beta, torch.Tensor):
        dist_matrix = torch.cdist(alpha, beta, p=2)
    else:
        dist_matrix = scipy.spatial.distance.cdist(alpha, beta, metric='euclidean')
        
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


def cnormalize(A, p=2, same_direction=False):
    """
    Normalize a matrix A to have columns of unit L^p norm.
    """
    A /= np.linalg.norm(A, axis=0, ord=p)
    if same_direction:
        orientation = np.sign(A[0])
        orientation[np.where(orientation == 0.0)] = 1.0
        A *= orientation

    return A


def expected_removal_loss(G,
                          emb_func,
                          cloud_metric=wasserstein_metric,
                          n_samples=-1,
                         ):
    """Expected distance between embeddings of the original graph and
    the graph with one edge removed uniformly at random.
    G: networkx graph,
    emb_func: function that takes G as an argument, built as a partial function of 
        an embedding function,
    cloud_metric: function to compute distance between point cloud embeddings; defaults to 
        built-in wasserstein metric,
    n_samples: if -1 (default), average across all possible edges removed; if not,
        try n_samples edges uniformly at random.
        
    Return: average distance.
    """

    if n_samples == -1:
        n_samples = G.number_of_edges()
        edges = G.edges()
    else:
        edges_idx = np.random.choice(range(G.number_of_edges()), size=2, replace=False)
        edges = list(G.edges())
        edges = [edges[idx] for idx in edges_idx]

    dists = np.empty(n_samples)

    emb = emb_func(G)
    
    for i, edge in enumerate(edges):
        G_removed = deepcopy(G)
        G_removed.remove_edge(*edge)
        emb_removed = emb_func(G_removed)
        dists[i] = cloud_metric(emb, emb_removed)
    
    return dists.mean()