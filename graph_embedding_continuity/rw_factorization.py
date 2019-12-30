import itertools
import networkx as nx

from .utils import normalize_sum_row_sparse

def rw_2step_graph(G, p=1.0, q=1.0):
    """From an undirected graph G, build the directed graph G_
    corresponding to the two-step Markov chain induced by the
    (p,q)-biased random walk à la node2vec.
    """
    nodes_ = [edge for edge in itertools.product(G.nodes, G.nodes) if edge in G.edges()]
    G_ = nx.DiGraph()
    G_.add_nodes_from(nodes_)
    for (u1, v1), (u2, v2) in itertools.product(nodes_, nodes_):
        if v1 == u2:
            # back to previous node
            if u1 == v2:
                G_.add_edge((u1,v1), (u2, v2), weight=1/p)
            else:
                spl = nx.shortest_path_length(G, u1, v2)
                if spl == 1:
                    G_.add_edge((u1,v1), (u2, v2), weight=1.0)
                else:
                    G_.add_edge((u1,v1), (u2, v2), weight=1/q)
    return G_


def rw_2step_transition_matrix(G=None, G_=None, p=1.0, q=1.0):
    """Transition matrix of the two-step (p,q)-biased
    random walk on G.
    If G_ is not provided, compute it from G.
    """
    if not G_:
        G_ = rw_2step_graph(G, p, q)
    return normalize_sum_row_sparse(nx.adjacency_matrix(G_))
