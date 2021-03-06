import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain



class PathEncoder(nn.Module):

    """Docstring for PathEncoder. """

    def __init__(self, ndim, num_layers = 2, device = "cuda"):
        """TODO: to be defined.

        :num_entities: TODO
        :num_rel: TODO
        :ndim: TODO

        """
        nn.Module.__init__(self)

        self.ndim = ndim
        self.rnn = nn.LSTM(input_size = 2*self.ndim, hidden_size=self.ndim,
                           num_layers = num_layers)
        self.device = device

    def forward(self, paths):
        """TODO: Docstring for forward.

        :paths: Padded list of [(e0, r0), (e1, r1), (e2, r2)]
        :entity_embeddings: TODO
        :rel_embeddings: TODO
        :returns: TODO

        """
        bsize = paths.shape[1]
        seqlen = paths.shape[0]
        output, (h, c) = self.rnn(paths)
        return h[-1,:,:]




class PathScorer(nn.Module):

    """Docstring for PathScorer. """

    def __init__(self, num_entities, num_rels, dim,num_layers=2, device="cuda" ):
        """TODO: to be defined.

        :num_entities: TODO
        :: TODO

        """
        nn.Module.__init__(self)

        self.num_entities = num_entities
        self.num_rels = num_rels
        self.dim = dim
        self.device = device
        self.path_encoder = PathEncoder(dim, num_layers, device)
        self.dummy_entity = num_entities
        self.entity_embeddings = nn.Embedding(num_entities+1, dim)
        self.dummy_rel = num_rels
        self.rel_embeddings = nn.Embedding(num_rels+1, dim)
    

    def pad_pack(self, paths, max_len = 3):
        padded = [ [(self.dummy_entity, self.dummy_rel)]*(max_len - len(p)) + p
                  for p in paths]
        embeddings = []
        for timestep in zip(*padded):
            entities, relations = zip(*timestep)
            entities = torch.LongTensor(entities).to(self.device)
            relations = torch.LongTensor(relations).to(self.device)
            embeddings.append(torch.cat([self.entity_embeddings(entities),
                               self.rel_embeddings(relations)], dim=1))
        # returns PathLen x NumPath x (2Dim) : concatenation of relation and
                              # entity
        return torch.stack(embeddings, dim = 0)

    def forward(self, paths, relation):
        num_paths = [len(p) for p in paths]
        all_paths = list(chain(*paths)) # concatenate all paths
        target_relation = self.rel_embeddings(torch.LongTensor([relation]).to(self.device))
        # 1 x dim
        packed_padded_paths = self.pad_pack(all_paths)
        path_embeddings = self.path_encoder(packed_padded_paths)
        # num_paths x dim
        scores = torch.sum(target_relation * path_embeddings, dim = 1)
        # num_paths,
        # paths to each endpoint
        split_paths = torch.split(scores, num_paths)
        total_score = torch.stack([torch.logsumexp(path_score, dim = 0) for
                                   path_score in split_paths])

        return torch.sigmoid(total_score)
 

        
