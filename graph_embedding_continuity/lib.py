import numpy as np
import scipy
import networkx as nx

from .deepwalk import DeepWalk
from .graph_kernels import shortest_path_feature_map, graphlet_feature_map


##############################
####### NODE EMBEDDING #######
##############################

def eigenmap_embedding(G, k):
    """
    G : nx graph,
    k: embedding dimension.
    
    Returns (N,k) array of (column) eigenvectors, (N) array of eigenvalues.
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
    
    Returns 2 (N,k) torch tensors
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


###############################
####### GRAPH EMBEDDING #######
###############################

def graph_kernel_embedding(Gs,
                           k=0,
                           kernel='',
                           config=dict(),
                           pad_symbol=0.0,
                          ):
    """
    Gs: list of n networkx graphs,
    k: int, select only the first k dimensions if k > 0,
    kernel: string, type of graph kernel to use (from graph_kernels.py),
    config: dict of kernel-specific parameters,
    pad_symbol: to pad output array to make it a matrix.
   
    Returns (n, d_max) array where d_max is the maximum embedding dimensions
    (some kernels such as shortest path kernel produce a varying number of
    dimensions), each vector being padded with pad_symbol.
    """
    if kernel == 'shortest_path':
        emb_map = shortest_path_feature_map
    elif kernel == 'graphlet':
        n_samples = config.get('n_samples', 0)
        k = config.get('k', 3)
        emb_map = lambda G: graphlet_feature_map(G,  n_samples=n_samples, k=k)
    else:
        raise Exception('{} kernel not implemented.'.format(kernel))
    
    emb = [emb_map(G) for G in Gs]
    # maximum dimension
    d_max = np.max([len(v) for v in emb])

    # pad with zeros
    emb_pad = np.zeros((len(Gs), d_max))
    for i, v in enumerate(emb):
        emb_pad[i, :len(v)] = v
        emb_pad[i, len(v):] = pad_symbol
    
    if k > 0:
        emb_pad = emb_pad[:, :k]
    
    return emb_pad.squeeze()