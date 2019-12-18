import numpy as np
import scipy
from copy import deepcopy
import networkx as nx

from .deepwalk import DeepWalk

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

def eigenmap_embedding(G, k):
    """
    G : nx graph,
    k: embedding dimension.
    
    Returns (N,k) array.
    """
    
    # Graph Laplacian
    L = nx.laplacian_matrix(G)

    # solve generalized eigenvalues problem with degree matrix
    D = scipy.sparse.diags(L.diagonal())
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(L, k=k,  M=D)

    # eigenvalues are in random order coming out of eigs, 
    # make sure they are now sorted in increasing order.
    # Also make sure both eigenvalues and eigenvectors 
    # are real; they should be but there is usually
    # a small numerical residual imaginary part.
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:,idx])

    return eigenvectors, eigenvalues

def deepwalk_embedding(G, 
                       k, 
                       n_train=0,
                       walk_length=0,
                       window_size=0,
                       n_neg=0,
                       hidden_size=0,
                       use_cuda=False,
                      ):
    """
    G : nx graph,
    k: embedding dimension,
    n_train: number of training iterations of the deepwalk network,
    walk_length : number of hops in the graph random walk,
    window_size: radius of the node context.
    
    Returns (N,k) torch tensor
    """
    dw = DeepWalk(G, 
                  walk_length=walk_length, 
                  window_size=window_size,
                  embedding_size=k,
                  n_neg=n_neg,
                  hidden_size=hidden_size,
                  use_cuda=use_cuda,
                 )
    dw.train(n_train)
    
    emb_word = dw.model_word(dw.one_hot).data
    emb_context = dw.model_context(dw.one_hot).data
        
    return emb_word, emb_context