import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import scipy
import abc
from tqdm import tqdm

from .utils import wasserstein_metric

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    
class SkipgramModel(abc.ABC):
    """
    Base class for skipgram-based models such as Deepwalk or node2vec.
    """
    def __init__(self,
                 G,
                 walk_length=0,
                 window_size=0,
                 embedding_size=0,
                 n_neg=0,
                 hidden_size=0,
                 verbose=10,
                 use_cuda=False,
                 time_reg_strength=0.0,
                 cloud_metric_p=2,
                 prior_emb_word=torch.empty(0),
                 prior_emb_context=torch.empty(0),
                ):
        self.G = G
        self.walk_length = walk_length
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.n_neg = n_neg # how many fake contexts to sample per true context
        self.hidden_size = hidden_size
        
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        
        self.A = nx.adjacency_matrix(self.G)
        self.P = self.proba_matrix()

        self.n = len(self.G.nodes())
        self.nodes = torch.LongTensor(self.G.nodes()).to(self.device)
        self.one_hot = F.one_hot(self.nodes).type(torch.FloatTensor).to(self.device)
        self.frequencies = np.ones(self.n)
        
        self.model_word = Model(self.n, self.hidden_size, self.embedding_size).to(self.device)
        self.model_context = Model(self.n, self.hidden_size, self.embedding_size).to(self.device)
        self.optimizer_word = torch.optim.Adam(self.model_word.parameters(), lr=0.001)
        self.optimizer_context = torch.optim.Adam(self.model_context.parameters(), lr=0.001)

        # strength of the embedding time evolution penalty
        self.time_reg_strength = time_reg_strength
        # L^p metric used to regularize embeddings
        self.cloud_metric_p = cloud_metric_p
        # prior embedding for time regularization
        self.prior_emb_word = prior_emb_word
        self.prior_emb_context = prior_emb_context
        
        # when to print in the training loop
        self.verbose = verbose

    def sigmoid(self, x):
        return 1/(1+torch.exp(-x))

    def proba_matrix(self):
        n, m = self.A.shape
        diags = self.A.sum(axis=1).flatten()
        with scipy.errstate(divide='ignore'):
            diags_inv = 1.0 / diags
        D_inv = scipy.sparse.spdiags(diags_inv, [0], m, n)
        return D_inv.dot(self.A)

    def cloud_metric(self, x, y):
        """ Function to compute distance between point cloud embeddings.
        """
        return torch.norm(y-x, self.cloud_metric_p)
        
    @abc.abstractmethod
    def sample_random_walk(self, node):
        pass
        
    @property
    def do_time_reg(self):
        return len(self.prior_emb_word) and len(self.prior_emb_context)
    
    def skipgram(self):
        """
        Run skipgram on the sentence composed of the nodes in self.path.
        """
        losses = []
        for j, u in enumerate(self.path):
            self.frequencies[u] += 1
            u_emb = self.model_word.forward(self.one_hot[u])
            window = self.path[max(j-self.window_size, 0): min(j+self.window_size, self.walk_length-1)]
            for v in window:
                v_emb = self.model_context.forward(self.one_hot[v])
                losses.append(-torch.log(self.sigmoid(torch.dot(u_emb, v_emb))))
            
            neg_distr = self.frequencies ** 0.75
            neg_distr = neg_distr / np.sum(neg_distr)
            negative_samples = np.random.choice(self.nodes.cpu(), self.n_neg*self.window_size, p=neg_distr)
            
            for v in negative_samples:
                v_emb = self.model_context.forward(self.one_hot[v])
                losses.append(-torch.log(self.sigmoid(-torch.dot(u_emb, v_emb))))
            
        loss = torch.sum(torch.stack(losses))
        return loss
        
    def loss(self):
        """
        Loss manager:
            * compute skipgram loss from current self.path,
            * can add embedding time regularization.
        """
        
        skipgram_loss = self.skipgram()
        
        if self.do_time_reg:
            # add time regularization only if prior embeddings are available
            time_reg_loss = self.time_reg_strength * (
                self.cloud_metric(self.model_word(self.one_hot), self.prior_emb_word) +
                self.cloud_metric(self.model_context(self.one_hot), self.prior_emb_context)
            )
        else:
            time_reg_loss = torch.tensor(0.0).to(self.device)
            
        loss = skipgram_loss + time_reg_loss
        
        self.optimizer_word.zero_grad()
        self.optimizer_context.zero_grad()
        loss.backward()
        self.optimizer_word.step()
        self.optimizer_context.step()
        
        return skipgram_loss.item(), time_reg_loss.item()
        
    def train(self, walks_per_node):
        if self.do_time_reg:
            tdqm_dict_keys = ['skipgram loss', 'time dynamic loss']
            tdqm_dict = dict(zip(tdqm_dict_keys, [0.0, 0.0]))
            postfix = {'skipgram loss': 0.0, 'time dynamic loss': 0.0}
        else:
            tdqm_dict_keys = ['loss']
            tdqm_dict = dict(zip(tdqm_dict_keys, [0.0]))
            postfix = {'loss': 0.0}

        for i in range(walks_per_node):
            if self.do_time_reg:
                total_skipgram_loss = 0.0
                total_time_reg_loss = 0.0
            else:
                total_loss = 0.0

            with tqdm(total=self.n,
                      unit_scale=True,
                      postfix=postfix,
                      desc="Epoch : %i/%i" % (i+1, walks_per_node),
                      ncols=100,
                      disable=((i+1)%self.verbose != 0 and i != 0 and i != walks_per_node-1),
                     ) as pbar:

                nodes = self.nodes[torch.randperm(self.n)]
                for j, node in enumerate(nodes):
                    self.path = self.sample_random_walk(node)
                    skipgram_loss, time_reg_loss = self.loss()
                    
                    # logging
                    if self.do_time_reg:
                        total_skipgram_loss += skipgram_loss
                        total_time_reg_loss += time_reg_loss
                        
                        tdqm_dict['skipgram loss'] = total_skipgram_loss/(j+1)
                        tdqm_dict['time dynamic loss'] = total_time_reg_loss/(j+1)
                    else:
                        total_loss += skipgram_loss
                        tdqm_dict['loss'] = total_loss/(j+1)
                    
                    pbar.set_postfix(tdqm_dict)
                    pbar.update(1)

                    
class DeepWalk(SkipgramModel):
    def __init__(self,
                 G,
                 walk_length=0,
                 window_size=0,
                 embedding_size=0,
                 n_neg=0,
                 hidden_size=0,
                 verbose=10,
                 use_cuda=False,
                 time_reg_strength=0.0,
                 cloud_metric_p=2,
                 prior_emb_word=torch.empty(0),
                 prior_emb_context=torch.empty(0),
                ):
        
        super().__init__(G,
                         walk_length=walk_length,
                         window_size=window_size,
                         embedding_size=embedding_size,
                         n_neg=n_neg,
                         hidden_size=hidden_size,
                         verbose=verbose,
                         use_cuda=use_cuda,
                         time_reg_strength=time_reg_strength,
                         cloud_metric_p=cloud_metric_p,
                         prior_emb_word=prior_emb_word,
                         prior_emb_context=prior_emb_context,
                        )
        
    def sample_random_walk(self, node):
        """Simulate a graph random walk starting from node.
        """
        path = torch.LongTensor(size=(self.walk_length,)).to(self.device)
        path[0] = node
        for t in range(1, self.walk_length):
            node = int(np.random.choice(self.G.nodes, p=self.P[node, :].A.flatten()))
            path[t] = node
        return path
    
    
class Node2Vec(SkipgramModel):
    def __init__(self,
                 G,
                 p=1.0,
                 q=1.0,
                 walk_length=0,
                 window_size=0,
                 embedding_size=0,
                 n_neg=0,
                 hidden_size=0,
                 verbose=10,
                 use_cuda=False,
                 time_reg_strength=0.0,
                 cloud_metric_p=2,
                 prior_emb_word=torch.empty(0),
                 prior_emb_context=torch.empty(0),
                ):
        
        super().__init__(G,
                         walk_length=walk_length,
                         window_size=window_size,
                         embedding_size=embedding_size,
                         n_neg=n_neg,
                         hidden_size=hidden_size,
                         verbose=verbose,
                         use_cuda=use_cuda,
                         time_reg_strength=time_reg_strength,
                         cloud_metric_p=cloud_metric_p,
                         prior_emb_word=prior_emb_word,
                         prior_emb_context=prior_emb_context,
                        )
    
        # lower p --> breadth-first sampling
        self.p = p
        # lower q --> depth-first sampling
        self.q = q
    
    def sample_random_walk(self, node):
        """Simulate a graph random walk starting from node.
        """
        path = torch.LongTensor(size=(self.walk_length,)).to(self.device)
        node = node.item()
        path[0] = node
        previous = node
        for t in range(1, self.walk_length):
            neighbors = list(self.G.neighbors(node))
            weights = dict(zip(
                neighbors,
                [
                    nx.shortest_path_length(self.G, previous, neighbor) for neighbor in neighbors
                ]
            ))
            
            proba = self.P[node, :].A.flatten()
            
            proba_neighb = [
                1/self.p * proba[neighb] if weight == 0 else (1/self.q * proba[neighb] if weight == 2 else proba[neighb]) for neighb, weight in weights.items()
            ]
            proba_neighb = proba_neighb / np.sum(proba_neighb)
            
            node = int(np.random.choice(neighbors, p=proba_neighb))
            path[t] = node
            previous = node
        return path