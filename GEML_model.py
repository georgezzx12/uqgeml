import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import pandas as pd
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from Encoders import Encoder
from Aggregators import MeanAggregator

"""
Supervised Grid_Embedding model.
"""


class GEML(nn.Module):

    def __init__(self, enc, rnn, m_size, embed_dim):
        super(GEML, self).__init__()
        self.enc = enc
        self.rnn = rnn
        self.MSE = nn.MSELoss()
        self.m_size = m_size
        self.embed_dim = embed_dim
        self.w_in = 0.25
        self.w_out = 0.25
        self.w_all = 0.5
        self.tran_Matrix = nn.Parameter(torch.FloatTensor(self.embed_dim, self.embed_dim))
        init.xavier_uniform_(self.tran_Matrix)
        self.tran_Matrix_in = nn.Parameter(torch.FloatTensor(1, self.embed_dim))
        init.xavier_uniform_(self.tran_Matrix_in)
        self.tran_Matrix_out = nn.Parameter(torch.FloatTensor(1, self.embed_dim))
        init.xavier_uniform_(self.tran_Matrix_out)
        self.hn = torch.FloatTensor(1, self.m_size, self.embed_dim)
        init.xavier_uniform_(self.hn)
        self.cn = torch.FloatTensor(1, self.m_size, self.embed_dim)
        init.xavier_uniform_(self.cn)

    def forward(self, nodes):
        embeds = self.enc(nodes).t()
        # print "self.tran_Matrix =", self.tran_Matrix
        # print "self.hn =", self.hn
        # print "self.cn =", self.cn
        inputs = embeds.reshape(1, self.m_size, self.embed_dim)
        output, (hn, cn) = self.rnn(inputs, (self.hn, self.cn))
        self.hn = nn.Parameter(hn)
        self.cn = nn.Parameter(cn)
        output = output.reshape(self.m_size, self.embed_dim)
        od_matrix = torch.FloatTensor(output.mm(self.tran_Matrix).mm(output.t()))
        od_in = torch.div(torch.FloatTensor(output.mm(self.tran_Matrix_in.t())), self.m_size)
        od_out = torch.div(torch.FloatTensor(output.mm(self.tran_Matrix_out.t())), self.m_size)
        return od_matrix, od_out, od_in

    def loss(self, nodes, ground_truth):
        od_matrix, od_out, od_in = self.forward(nodes)
        ground_truth = torch.FloatTensor(ground_truth)
        gt_out = torch.div(ground_truth.sum(1, keepdim=True), self.m_size)
        gt_in = torch.div(ground_truth.sum(0, keepdim=True).t(), self.m_size)
        loss_in = torch.mul(self.MSE(od_in, gt_in), self.w_in)
        loss_out = torch.mul(self.MSE(od_out, gt_out), self.w_out)
        loss_all = torch.mul(self.MSE(od_matrix, ground_truth), self.w_all)
        loss = loss_in + loss_out + loss_all
        return loss

def load_beijng_periodicity(df, start_index, m_size):
    feat_out = df[start_index: start_index + m_size].reset_index(drop=True)

    feat_in = feat_out.T.reset_index(drop=True)

    feat_o = feat_in.apply(lambda x: x.sum(), axis=1).reset_index(drop=True)

    feat_i = feat_out.apply(lambda x: x.sum(), axis=1).reset_index(drop=True)

    frames = [feat_out, feat_in, feat_o, feat_i]
    feat_data = pd.concat(frames, ignore_index=True, axis=1).reset_index(drop=True)

    feat_data = np.array(feat_data)
    feat_out = np.array(feat_out)

    return feat_data, feat_out

def MSE(pred, gt):
    diff = abs(pred - gt)
    size = gt.shape[0] * gt.shape[1]
    mse = torch.div(torch.sum(diff.mul(diff)), size)
    return mse

def SMAPE(pre, actul):
    sum = pre + actul + 1
    diff = abs(pre - actul)
    size = pre.shape[0] * pre.shape[1]
    smape = torch.mul(torch.div(torch.sum(torch.div(diff, sum)), size), 2)
    return smape

def run_beijing():
    month = 8
    if month == 11:
        row_num = 15
        col_num = 15
        m_size = 225
        scale = 400
        train_day = 22
        train_size = m_size * 24 * 23
        test_size = m_size * 24 * 8
        model_name = "November_"
        path = "../Datasets/Chengdu/11_Didi_Matrix.csv"
        path3 = "../Datasets/Chengdu/11_Zero_PreResult.csv"
        path4 = "../Datasets/Chengdu/11_HisAve_PreResult.csv"
    elif month ==8:
        row_num = 20
        col_num = 20
        m_size = 400
        scale = 220
        train_day = 23
        train_size = m_size * 24 * 24
        test_size = m_size * 24 * 8
        model_name = "August_"
        path = "../Datasets/Beijing/8_Matrix.csv"
        path3 = "../Datasets/Beijing/8_Zero_PreResult.csv"
        path4 = "../Datasets/Beijing/8_HisAve_PreResult.csv"

    learning_rate = 0.0001
    epoch_no = 2
    model_no =40

    batch_no = 24
    step_size = 3
    df = pd.read_csv(path, header=None)
    train_df = df[:train_size]
    test_df = df[(df.shape[0]-test_size):].reset_index(drop=True)

    geo_neighbors = defaultdict(set)

    # Formulate the neighbor set for each grid
    for i in range(0, m_size):

        grid_no = i
        gn_grids = []
        # gn_grid2dis = {}
        if i == 0:
            gn_grids = [i + 1, i + col_num, i + col_num + 1]
            # print "0::::[i + 1, i + col_num, i + col_num + 1]"

        elif i == col_num - 1:
            gn_grids = [i - 1, i + col_num - 1, i + col_num]
            # print "2::::[i - 1, i + col_num - 1, i + col_num]"

        elif i == row_num * col_num - col_num:
            gn_grids = [i - col_num, i - col_num + 1, i + 1]
            # print "6::::[i - col_num, i - col_num + 1, i + 1]"

        elif i == row_num * col_num - 1:
            gn_grids = [i - col_num - 1, i - col_num, i - 1]
            # print "8::::[i - col_num - 1, i - col_num, i - 1]"

        elif i in range(1, col_num - 1):
            gn_grids = [i - 1, i + 1, i + col_num - 1, i + col_num, i + col_num + 1]
            # print "1::::[i - 1, i + 1, i + col_num - 1, i + col_num, i + col_num + 1]"

        elif i in range(row_num * col_num - col_num + 1, row_num * col_num - 1):
            gn_grids = [i - col_num - 1, i - col_num, i - col_num + 1, i - 1, i + 1]
            # print "7::::[i - col_num - 1, i - col_num, i - col_num + 1, i - 1, i + 1]"

        elif i in range(0 + col_num, row_num * col_num - col_num, col_num):
            gn_grids = [i - col_num, i - col_num + 1, i + 1, i + col_num, i + col_num + 1]
            # print "3::::[i - col_num, i - col_num + 1, i + 1, i + col_num, i + col_num + 1]"

        elif i in range(col_num - 1 + col_num, row_num * col_num - 1, col_num):
            gn_grids = [i - col_num - 1, i - col_num, i - 1, i + col_num - 1, i + col_num]
            # print "5::::[i - col_num - 1, i - col_num, i - 1, i + col_num - 1, i + col_num]"

        else:
            gn_grids = [i - col_num - 1, i - col_num, i - col_num + 1, i - 1, i + 1, i + col_num - 1, i + col_num,
                        i + col_num + 1]
            # print "4::::[i - col_num - 1, i - col_num, i - col_num + 1, i - 1, i + 1, i + col_num - 1, i + col_num, i + col_num + 1]"

        # print "gn_grids=" + "\n", gn_grids
        for k in range(len(gn_grids)):
            geo_neighbors[grid_no].add(gn_grids[k])
        # print "geo_neighbors[", grid_no, "]=" , geo_neighbors[grid_no]
        # print "\n"

    batch_nodes = list(range(m_size))
    feature_dim = m_size*2 + 2
    embed_dim = 128
    feat_data = torch.FloatTensor(m_size, feature_dim)
    feat_out = torch.zeros(m_size, m_size)
    # print "Before iteration feat_out.sum = ", torch.sum(feat_out).item()
    features = nn.Embedding(m_size, feature_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # print "Before iteration features.weight.sum = ", torch.sum(torch.FloatTensor(torch.Tensor(features(torch.LongTensor(batch_nodes))))).item()
    # hidden_feat = torch.FloatTensor(2, 400, 128)

    agg1 = MeanAggregator(features, feat_out, gcn=False, cuda=True)
    enc1 = Encoder(features, feature_dim, embed_dim, geo_neighbors, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), feat_out, gcn=False, cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, embed_dim, geo_neighbors, agg2,
                   base_model=enc1, gcn=False, cuda=False)
    rnn = nn.LSTM(embed_dim, embed_dim, 1)

    geml = GEML(enc2, rnn, m_size, embed_dim)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, geml.parameters()), lr=learning_rate)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, geml.parameters()), lr=learning_rate, momentum=0.8, weight_decay=1e-5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, geml.parameters()), lr=learning_rate)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, geml.parameters()), lr=learning_rate, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8)

    # Training Process
    times = []
    path1 = "../../Result/GEML/" + model_name + str(model_no) + ".csv"
    path2 = "../../Model/GEML/" + model_name + str(model_no) + ".pt"
    print model_name + str(model_no)
    fo = open(path1, "w")
    fo.write(
       "###################################################################" + "\n")
    fo.write("w_in=" + str(geml.w_in) + "; w_out=" + str(geml.w_out) + "; w_all=" + str(geml.w_all) + "\n")
    fo.write("Learning_rate=" + str(learning_rate) + "\n")
    fo.close()
    for epoch in range(epoch_no):
        # print"\n"
        print "#############################################################################################################"
        print "Training Process: epoch=", epoch
        batches = random.sample(range(0, 24), batch_no)  # Randomly choose batch_no sequences from the sequences of 24 hours.
        print "Training iterations = ", batches
        str_batches = map(lambda x: str(x), batches)
        fo = open(path1, "a")
        fo.write(
           "#############################################################################################################" + "\n")
        fo.write("Training Process: epoch=" + str(epoch) + "\n" + "batches= " + ",".join(str_batches) + "\n")
        start_time = time.time()
        for batch in batches:
            loss = torch.zeros(1)
            l1_regularization = torch.zeros(1)
            for n in range(train_day):  # train_day-12. Load one Matrix at the same time slot with iteration in the following day
                start_index = (batch + n * 24) * m_size
                feat_data, feat_out = load_beijng_periodicity(train_df, start_index, m_size)
                feat_out = torch.FloatTensor(feat_out)
                agg1.feat_out = feat_out # send this matrix into aggregator to calculate the mask2, the pre-weight of embedding
                agg2.feat_out = feat_out
                # print "Before loss feat_out.sum = ", torch.sum(feat_out).item()
                gt_start = (batch + (n + 1) * 24) * m_size
                ground_truth = torch.div(
                    torch.FloatTensor(np.array(train_df[gt_start:gt_start + m_size].reset_index(drop=True))), scale)
                feat_data = torch.div(torch.FloatTensor(feat_data), scale)
                # print "Before loss feat_data.sum = ", torch.sum(feat_data).item()
                features.weight = nn.Parameter(feat_data, requires_grad=False)
                # print "Before loss features.weight.sum = ", torch.sum(
                #     torch.FloatTensor(torch.Tensor(features(torch.LongTensor(batch_nodes))))).item()
                # print "torch.isnan(features.weight)=", torch.isnan(features.weight).any()
                optimizer.zero_grad()
                loss_one = geml.loss(batch_nodes, Variable(torch.FloatTensor(ground_truth[np.array(batch_nodes)])))
                loss += loss_one
                # print "loss_one= ", loss_one
            param_count = 0
            for param in geml.parameters():
                l1_regularization += torch.mean(torch.abs(param))
                param_count += 1
            l1_regularization = torch.div(l1_regularization, param_count)
            loss = loss + l1_regularization
            print "loss=", loss.item()
            fo.write("loss=" + str(loss.item()) + "\n")
            # print "#############################################################################################"
            loss.backward(retain_graph=True)
            optimizer.step()
            # scheduler.step()

            # loss = torch.zeros(1)
            # l1_regularization = torch.zeros(1)
            # for n in range(train_day-12, train_day):  # Load one Matrix at the same time slot with iteration in the following day
            #     start_index = (batch + n * 24) * m_size
            #     feat_data, feat_out = load_beijng_periodicity(train_df, start_index, m_size)
            #     feat_out = torch.FloatTensor(feat_out)
            #     agg1.feat_out = feat_out
            #     agg2.feat_out = feat_out
            #     # print "Before loss feat_out.sum = ", torch.sum(feat_out).item()
            #     gt_start = (batch + (n + 1) * 24) * m_size
            #     # ground_truth = torch.div(
            #     #     torch.FloatTensor(np.array(train_df[gt_start:gt_start + m_size].reset_index(drop=True))), scale)
            #     ground_truth = DealZero(torch.div(
            #         torch.FloatTensor(np.array(train_df[gt_start:gt_start + m_size].reset_index(drop=True))), scale))
            #     # feat_data = torch.div(torch.FloatTensor(feat_data), scale)
            #     feat_data = DealZero(torch.div(torch.FloatTensor(feat_data), scale))
            #     features.weight = nn.Parameter(feat_data, requires_grad=False)
            #     # print "torch.isnan(features.weight)=", torch.isnan(features.weight).any()
            #
            #     optimizer.zero_grad()
            #     loss_one = geml.loss(batch_nodes, Variable(torch.FloatTensor(ground_truth[np.array(batch_nodes)])))
            #     loss += loss_one
            #     # print "loss_one= ", loss_one
            #     # print "torch.isnan(hn)=", torch.isnan(hn).any()
            #     # print "torch.isnan(cn)=", torch.isnan(cn).any()
            #
            # param_count = 0
            # for param in geml.parameters():
            #     l1_regularization += torch.mean(torch.abs(param))
            #     param_count += 1
            # l1_regularization = torch.div(l1_regularization, param_count)
            # loss = loss + l1_regularization
            # print "loss=", loss.item()
            # fo.write("loss=" + str(loss.item()) + "\n")
            # # print "#############################################################################################"
            # loss.backward(retain_graph=True)
            # optimizer.step()
            # # scheduler.step()

        end_time = time.time()
        times.append(end_time - start_time)
        # print "#############################################################################################################"

        # print"times=", times
    torch.save(geml.state_dict(), path2)
    print "Total time of training:", np.sum(times)
    # print"times=", times
    print "Average time of training:", np.mean(times)
    fo.write(
       "Total time of training:" + str(np.sum(times)) + "\n" + "Average time of training:" + str(
           np.mean(times)) + "\n")
    # Testing Process
    result = []
    ground = []
    for iteration in range(24):
        fo.write(
           "#############################################################################################################" + "\n")
        fo.write("Testing iteration =" + str(iteration) + "\n")
        print "#############################################################################################################"
        print "Testing iteration =", iteration
        for m in range(7):  # Load one Matrix (that is, a graph) in every iteration
            start_index = (iteration + m * 24) * m_size
            feat_data, feat_out = load_beijng_periodicity(test_df, start_index, m_size)
            feat_out = torch.FloatTensor(feat_out)
            agg1.feat_out = feat_out
            agg2.feat_out = feat_out
            gt_start = (iteration + (m + 1) * 24) * m_size
            ground_truth = test_df[gt_start:(gt_start + m_size)].reset_index(drop=True)
            feat_data = torch.div(torch.FloatTensor(feat_data), scale)
            features.weight = nn.Parameter(feat_data, requires_grad=False)
            od_matrix, od_out, od_in = geml.forward(batch_nodes)
            od_matrix = torch.mul(od_matrix, scale)
            torch.nn.functional.relu(od_matrix, inplace=True)
            sum1 = torch.sum(od_matrix).item()
            sum2 = ground_truth.sum().sum()
            print "od_matrix.sum=", sum1
            print "ground_truth.sum=", sum2
            # print("od_matrix=", od_matrix)
            result.append(od_matrix)
            ground.append(ground_truth)
            # print "result = ", result
            print "Testing Process: m=", m
            fo.write("od_matrix.sum=" + str(sum1) + "\n" + "ground_truth.sum=" + str(sum2) + "\n")

    od_matrices = torch.cat(result)
    ground_truth = torch.FloatTensor(np.array(pd.concat(ground)))
    size = ground_truth.shape[0] * ground_truth.shape[1]
    mse = MSE(od_matrices, ground_truth)
    rmse = torch.sqrt(mse)
    smape = SMAPE(od_matrices, ground_truth)
    print "#############################################################################################################"
    print "test_mse=", mse.item()
    print "test_rmse=", rmse.item()
    print "test_smape=", smape.item()
    print "ground_truth.size=", size
    df1 = pd.DataFrame(od_matrices.detach().numpy())
    df1.to_csv("../Datasets/Beijing/8_GEML_PredResult_3.csv", header=False, index=False)
    print("Save GEML PredResult!")


if __name__ == "__main__":
    run_beijing()
