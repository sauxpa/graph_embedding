import numpy as np
import scipy.sparse as sp
import networkx as nx

from .deepwalk import DeepWalk, Node2Vec
from .graph_kernels import shortest_path_feature_map, graphlet_feature_map
from .rw_factorization import *
from .utils import cnormalize

##############################
####### NODE EMBEDDING #######
##############################

def eigenmap_embedding(G, 
                       k,
                       dtype='float64',
                       normalize=False,
                      ):
    """
    G : nx graph,
    k: embedding dimension,
    dtype: scipy.sparse.eigs only supports float dtype,
    normalize: if True, scale each vector by its L2 norm.
    
    Returns (N,k) array of (column) eigenvectors, (N) array of eigenvalues.
    """
    
    # Graph Laplacian
    L = nx.laplacian_matrix(G).astype(dtype)

    # solve generalized eigenvalues problem with degree matrix
    # extract smallest magnitude eigenvalues
    D = sp.diags(L.diagonal())
    eigenvalues, eigenvectors = sp.linalg.eigs(L, k=k,  M=D, which='SR')

    # eigenvalues are in random order coming out of eigs, 
    # make sure they are now sorted in increasing order.
    # Also make sure both eigenvalues and eigenvectors 
    # are real; they should be but there is usually
    # a small numerical residual imaginary part.
    idx = eigenvalues.argsort()
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:,idx])

    if normalize:
        # unit norm
        eigenvectors = cnormalize(eigenvectors)
        # same direction
        eigenvectors *= np.sign(eigenvectors[0])
    
    return eigenvectors, eigenvalues


def rw_factorization_embedding(G, 
                               k, 
                               p=1.0, 
                               q=1.0, 
                               dtype='float64', 
                               normalize=False,
                              ):
    """
    G : nx graph,
    k: embedding dimension,
    p, q: bias parameters in node2vec random walks.
    dtype: scipy.sparse.eigs only supports float dtype,
    normalize: if True, scale each vector by its L2 norm.
    
    Returns (N,k) array of (column) eigenvectors, (N) array of eigenvalues.
    """
    
    # transition matrix for the two-step biased Markov chain
    G_ = rw_2step_graph(G, p, q)
    P = rw_2step_transition_matrix(G_=G_).astype(dtype)
    
    # extract eigenvalue 1 (invariant measure) and the largest others.
    eigenvalues_, eigenvectors_ = sp.linalg.eigs(P.T, k=k, which='LR')

    # eigenvalues are in random order coming out of eigs, 
    # make sure they are now sorted in increasing order.
    # Also make sure both eigenvalues and eigenvectors 
    # are real; they should be but there is usually
    # a small numerical residual imaginary part.
    idx = eigenvalues_.argsort()[::-1]
    eigenvalues_ = np.real(eigenvalues_[idx])
    eigenvectors_ = np.real(eigenvectors_[:,idx])
    
    # Collapse the two-step chain.
    # For instance the first eigenvector corresponds to
    # the invariant measure of the walk, in this case :
    # P(W_t=y) = sum_{x} P(W_{t-1}=x, W_t=y)
    eigenvectors = np.empty((G.number_of_nodes(), k))
    for i, node in enumerate(G.nodes):
        idx = [j for j, node_ in enumerate(G_.nodes) if node_[1] == node]
        eigenvectors[i, :] = np.sum(eigenvectors_[idx, :], axis=0)
    
    if normalize:
        # unit norm
        eigenvectors = cnormalize(eigenvectors)
        # same direction
        eigenvectors *= np.sign(eigenvectors[0])
        
    return eigenvectors


def deepwalk_embedding(G, 
                       k, 
                       n_train=0,
                       walk_length=0,
                       window_size=0,
                       n_neg=0,
                       hidden_size=0,
                       use_cuda=False,
                       normalize=False,
                      ):
    """
    G : nx graph,
    k: embedding dimension,
    n_train: number of training iterations of the deepwalk network,
    walk_length : number of hops in the graph random walk,
    window_size: radius of the node context,
    normalize: if True, scale each vector by its L2 norm.
    
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
        
    if normalize:
        emb_word = cnormalize(emb_word)
        emb_context = cnormalize(emb_context)
        
    return emb_word, emb_context


def node2vec_embedding(G, 
                       k,
                       p=1.0,
                       q=1.0,
                       n_train=0,
                       walk_length=0,
                       window_size=0,
                       n_neg=0,
                       hidden_size=0,
                       use_cuda=False,
                       normalize=False,
                      ):
    """
    G : nx graph,
    k: embedding dimension,
    p, q: float, to control bfs/dfs in node2vec random walks,
    n_train: number of training iterations of the deepwalk network,
    walk_length : number of hops in the graph random walk,
    window_size: radius of the node context,
    normalize: if True, scale each vector by its L2 norm.
    
    Returns 2 (N,k) torch tensors
    """
    node2vec = Node2Vec(G, 
                        p=p,
                        q=q,
                        walk_length=walk_length, 
                        window_size=window_size,
                        embedding_size=k,
                        n_neg=n_neg,
                        hidden_size=hidden_size,
                        use_cuda=use_cuda,
                       )
    node2vec.train(n_train)
    
    emb_word = node2vec.model_word(node2vec.one_hot).data
    emb_context = node2vec.model_context(node2vec.one_hot).data
        
    if normalize:
        emb_word = cnormalize(emb_word)
        emb_context = cnormalize(emb_context)
        
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