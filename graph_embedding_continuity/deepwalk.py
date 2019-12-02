
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import scipy

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class DeepWalk():
    def __init__(self,
                 G,
                 walk_length=0,
                 window_size=0,
                 embedding_size=0,
                 n_neg=0,
                 hidden_size=0,
                 verbose=10,
                 use_cuda=False,
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

    def sample_random_walk(self, node):
        """Simulate a graph random walk starting from node.
        """
        path = torch.LongTensor(size=(self.walk_length,)).to(self.device)
        path[0] = node
        for t in range(1, self.walk_length):
            node = int(np.random.choice(self.G.nodes, p=self.P[node, :].A.flatten()))
            path[t] = node
        return path

    def skipgram(self, path):
        """Run skipgram on the sentence composed of the nodes in path.
        """
        losses = []
        for j, u in enumerate(path):
            self.frequencies[u] += 1
            u_emb = self.model_word.forward(self.one_hot[u])
            window = path[max(j-self.window_size, 0): min(j+self.window_size, self.walk_length-1)]
            for v in window:
                v_emb = self.model_context.forward(self.one_hot[v])
                losses.append(-torch.log(self.sigmoid(torch.dot(u_emb, v_emb))))
                del v_emb
            
            neg_distr = self.frequencies ** 0.75
            neg_distr = neg_distr / np.sum(neg_distr)
            negative_samples = np.random.choice(self.nodes.cpu(), self.n_neg*self.window_size, p=neg_distr)
            
            for v in negative_samples:
                v_emb = self.model_context.forward(self.one_hot[v])
                losses.append(-torch.log(self.sigmoid(-torch.dot(u_emb, v_emb))))
                del v_emb
            
            del u_emb

        loss = torch.sum(torch.stack(losses))
        
        self.optimizer_word.zero_grad()
        self.optimizer_context.zero_grad()
        loss.backward()
        self.optimizer_word.step()
        self.optimizer_context.step()

    def train(self, walks_per_node):
        print('Training...')
        for i in range(walks_per_node):
            nodes = self.nodes[torch.randperm(self.n)]
            for node in nodes:
                path = self.sample_random_walk(node)
                self.skipgram(path)
            if (i+1)%self.verbose == 0:
                print('{}/{}'.format(i+1, walks_per_node))
        print('...done!')
            