import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class PathEncoder(nn.Module):

    """Docstring for PathEncoder. """

    def __init__(self, ndim, device = "cpu"):
        """TODO: to be defined.

        :num_entities: TODO
        :num_rel: TODO
        :ndim: TODO

        """
        nn.Module.__init__(self)

        self.ndim = ndim
        self.map_rel = nn.Linear(self.ndim, self.ndim, bias = False)
        self.map_ent = nn.Linear(self.ndim, self.ndim, bias = False)
        self.map_hidden = nn.Linear(self.ndim, self.ndim, bias = True)
        self.device = torch.device(device)

    def forward(self, paths):
        """TODO: Docstring for forward.

        :paths: Padded list of [(e0, r0), (e1, r1), (e2, r2)]
        :entity_embeddings: TODO
        :rel_embeddings: TODO
        :returns: TODO

        """
        bsize = paths[0][0].shape[0]
        rembedding = torch.zeros((bsize, self.ndim))
        for e, r in paths:
            rembedding = F.relu(self.map_rel(r) + self.map_ent(e) +
                                self.map_hidden(rembedding))
        return rembedding




class PathScorer(nn.Module):

    """Docstring for PathScorer. """

    def __init__(self, num_entities, num_rels, dim, device="cpu" ):
        """TODO: to be defined.

        :num_entities: TODO
        :: TODO

        """
        nn.Module.__init__(self)

        self.num_entities = num_entities
        self.num_rels = num_rels
        self.dim = dim
        self.device = torch.device(device)
        self.path_encoder = PathEncoder(dim, device)
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
            entities = torch.LongTensor(entities, device = self.device)
            relations = torch.LongTensor(relations, device = self.device)
            embeddings.append((self.entity_embeddings(entities),
                               self.rel_embeddings(relations)))
        return embeddings

    def forward(self, paths, relation):
        target_relation = self.rel_embeddings(torch.LongTensor([relation],
                                                               device=self.device))
        # 1 x dim
        packed_padded_paths = self.pad_pack(paths)
        path_embeddings = self.path_encoder(packed_padded_paths)
        # num_paths x dim
        scores = torch.sum(target_relation * path_embeddings, dim = 1)
        # num_paths,
        total_score = torch.logsumexp(scores, dim = 0)
        return torch.sigmoid(total_score)
 

        
