import networkx as nx

def generate_cycle(N):
    """
    N: number of nodes
    
    Returns unweighted, undirected networkx Graph 
    corresponding to a cycle 0->1->...->n-1->0.    
    """
    cycle = nx.Graph()
    cycle.add_nodes_from(range(N))
    cycle.add_edges_from([(i, (i+1) % N) for i in range(N)])
    return cycle

def generate_chain(N):
    """
    N: number of nodes
    
    Returns unweighted, undirected networkx Graph 
    corresponding to a chain 0->1->...->n-1.    
    """
    chain = nx.Graph()
    chain.add_nodes_from(range(N))
    chain.add_edges_from([(i, i+1) for i in range(N-1)])
    return chain
