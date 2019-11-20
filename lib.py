import numpy as np
import scipy
from copy import deepcopy

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

def eigenmap_embedding(L, k):
    """
    L : graph Laplacian
    k: embedding dimension
    return (N,k) array
    """

    # solve generalized eigenvalues problem with degree matrix
    D = scipy.sparse.diags(L.diagonal())
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(L, k=2,  M=D)

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    return eigenvectors
