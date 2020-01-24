# graph_embedding

Investigation of the continuity of the several graph embedding techniques (Eigenmap, DeepWalk, node2vec, graph kernels...as well as an original Random Walk Matrix Factorization method).

* **graph_embedding_continuity**: Python library for various graph embeddings (implementation from scratch using numpy, scipy, torch and networkx)

Two problems classes of problems are studied:
* **continuity**: remove an edge and compare the embedding before and after in several situations (complete graph, chain and cycle, preferential attachment, removal or weakening of an inter-community bridge).
* **commutativity**: if one removes a sequence of edges, does the order matter? For matrix-based embeddings (eigenmap, RWF) the answer is no (the final matrix that is factorized is the same in all cases). For neural network-based methods, it may indeed matter, due to the sequential nature of training.
