import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np
"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, feat_out, gcn=False, cuda=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.feat_out = feat_out
        self.cuda = cuda
        self.gcn = gcn

    def regu(self, matrix):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i,j]==0.0025:
                    matrix[i,j] = 0
        return matrix

    def forward(self, nodes, to_neighs, num_sample = None):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        row_num = 15
        _set = set
        # iteration = 1
        if not num_sample is None:
            _sample = random.sample

            #Randomly sample ten elements a time from to_neighs and add them into sample_neighs
            samp_neighs = [_set(_sample(to_neigh,
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        #mask1 is the pre-weighted function of geographic neighbors
        mask1 = Variable(torch.zeros(len(samp_neighs), len(nodes)))
        for i in range(len(nodes)):
            grid_no = i
            neighs = samp_neighs[i]
            for neigh in neighs:
                if abs(grid_no-neigh) in {row_num, 1}:
                    mask1[grid_no, neigh] = 1
                if abs(grid_no-neigh) in {row_num+1, row_num-1}:
                    mask1[grid_no, neigh] = np.sqrt(2)
        if self.cuda:
            mask1 = mask1.cuda()
        num_neigh1 = mask1.sum(1, keepdim=True) + 0.00001
        mask1 = mask1.div(num_neigh1)

        # mask2 is the pre-weighted function of semantic neighbors
        mask2 = self.feat_out
        if self.cuda:
            mask2 = mask2.cuda()
        num_neigh2 = mask2.sum(1, keepdim=True) + 0.00001
        mask2 = mask2.div(num_neigh2)
        mask2 = self.regu(mask2)

        if self.cuda:
                embed_matrix = torch.Tensor(self.features(torch.LongTensor(nodes).cuda()))
        else:
                embed_matrix = torch.Tensor(self.features(torch.LongTensor(nodes)))
        # print "Aggregator.features.weight.sum = ", torch.sum(
        #     torch.FloatTensor(torch.Tensor(self.features(torch.LongTensor(nodes))))).item()

        to_feats1 = mask1.mm(embed_matrix)
        to_feats2 = mask2.mm(embed_matrix)
        to_feats = torch.cat([to_feats1, to_feats2], dim=1)

        return to_feats

