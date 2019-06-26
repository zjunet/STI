# -*- coding: utf-8 -*-

import torch
import numpy as np
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import *
import time
import dataprocess
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--data_file', type=str, default='data/data.npy', help='path of input file')
parser.add_argument('-n', '--social_network', type=str, default='data/network.pkl', help='path of network file')
parser.add_argument('-o', '--output_file', type=str, default='data/imputed_data.npy', help='path of output file')
parser.add_argument('-m', '--missing_marker', type=float, default=-1, help='marker of missing elements, default value is -1')
parser.add_argument('-b', '--batch_size', type=int, default=256, help='the number of samples in each batch, default value is 256')
parser.add_argument('-e', '--num_epoch', type=int, default=200, help='number of epoch, default value is 200')
parser.add_argument('-s', '--hidden_size', type=int, default=32, help='size of hidden feature in LSTM, default value is 32')
parser.add_argument('-k', '--dim_memory', type=int, default=32, help='dimension of memory matrix, default value is 32')
parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
parser.add_argument('-d', '--dropout', type=float, default=0.8, help='the dropout rate of output layers, default value is 0.8')
parser.add_argument('-r', '--decoder_learning_ratio', type=float, default=5, help='ratio between the learning rate of decoder and encoder, default value is 10')
parser.add_argument('-w', '--weight_decay', type=float, default=0)
parser.add_argument('--log', action='store_true', help='print log information, you can see the train loss in each epoch')

args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(0)
# Device configuration
torch.set_default_tensor_type('torch.FloatTensor')

# Hyper-parameters
hidden_size = args.hidden_size
batch_size = args.batch_size
K = args.dim_memory
num_epochs = args.num_epoch

learning_rate = args.learning_rate
weight_decay = args.weight_decay
dropout = args.dropout
decoder_learning_ratio = args.decoder_learning_ratio

input_data = np.load(args.data_file)
input_size = input_data.shape[2]
S = input_data.shape[1]
L = [[1. * s * k / S / K + (1 - 1. * k / K) * (1 - 1. * s / S) for k in range(1, K + 1)] for s in range(1, S + 1)]
L = th.from_numpy(np.array(L))


class DataSet(torch.utils.data.Dataset):
    def __init__(self):
        super(DataSet, self).__init__()

        self.input_data, self.input_time, self.input_mask, self.input_interval, self.input_length, \
           self.output_mask_all, self.origin_u, self.neighbor_length, self.neighbor_interval, self.neighbor_time, self.neighbor_data, self.lower, self.upper\
            = dataprocess.load_data(np.load(args.data_file), pickle.load(open(args.social_network, 'rb')), args.missing_marker)
        self.input_interval = np.expand_dims(self.input_interval, axis=2)
        self.neighbor_interval = np.expand_dims(self.neighbor_interval, axis=3)
        self.mask_in = (self.output_mask_all == 1).astype(dtype = np.float32)

        self.mask_out = (self.output_mask_all == 2).astype(dtype = np.float32)
        self.mask_all = (self.output_mask_all != 0).astype(dtype = np.float32)

    def __getitem__(self, index):
        return self.input_data[index], self.input_mask[index], self.input_interval[index], self.input_length[index],\
               self.origin_u[index], self.mask_in[index], self.mask_out[index], self.mask_all[index], self.neighbor_data[index],\
               self.neighbor_interval[index], self.neighbor_length[index]

    def __len__(self):
        return len(self.input_data)

train_dataset = DataSet()
test_dataset = DataSet()
print('load successfully')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)

# print(input_data.shape[1])
encoder = MemoryEncoder(input_size, hidden_size, K)
neighbor_encoder = MemoryEncoder(input_size, hidden_size, K)
decoder = DecoderRNN(input_size, hidden_size, 6, K)

if th.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# optimier
encoder_optimizer = torch.optim.Adam(encoder.parameters() , lr=learning_rate)
neighbor_encoder_optimizer = torch.optim.Adam(neighbor_encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

if th.cuda.is_available():
    encoder = encoder.cuda()
    neighbor_encoder = neighbor_encoder.cuda()
    decoder = decoder.cuda()

# Train the model
# total_step = len(train_loader)
curve = []
curve_train = []
best_performance = [10000, 10000]
for epoch in range(num_epochs):
    loss_all, num_all = 0, 0
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    for i, (i_data, i_mask, i_interval, i_length, u_all, m_in, m_out, m_all, n_input, n_inter, n_len) in enumerate(train_loader):
        input = i_data
        m_all = m_all.unsqueeze(2)
        m_in = m_in.unsqueeze(2)
        # print u_all.size()
        if th.cuda.is_available():
            input = input.cuda()
            i_length = i_length.cuda()
            i_interval = i_interval.cuda()
            i_mask = i_mask.cuda()
            u_all = u_all.cuda()
            m_in = m_in.cuda()
            n_input = n_input.cuda()
            n_inter = n_inter.cuda()
            n_len = n_len.cuda()

        start = time.time()
        loss, num = train_batch(input, i_length, i_interval,i_mask, u_all, m_in, n_input, n_inter, n_len,
                encoder.train(), neighbor_encoder.train(), decoder.train(), encoder_optimizer, neighbor_encoder_optimizer, decoder_optimizer)
        loss_all += loss
        num_all += num
    if args.log:
        print('train epoch {} mse:'.format(epoch), loss_all * (train_dataset.upper - train_dataset.lower) *1./num_all/train_dataset.input_data.shape[2])
    curve_train.append(loss_all)

impute_data_all = []
for i, (i_data, i_mask, i_interval, i_length, u_all, m_in, m_out, m_all, n_input, n_inter, n_len) in enumerate(test_loader):
    input = i_data
    m_out = m_out.unsqueeze(2)
    m_in = m_in.unsqueeze(2)
    m_all = m_all.unsqueeze(2)
    if th.cuda.is_available():
        input = input.cuda()
        i_length = i_length.cuda()
        i_interval = i_interval.cuda()
        i_mask = i_mask.cuda()
        m_out = m_out.cuda()
        m_in = m_in.cuda()
        m_all = m_all.cuda()
        u_all = u_all.cuda()

    imputed_data = impute_batch(i_data, i_length, i_interval, i_mask, u_all, m_all, n_input, n_inter, n_len,
            encoder.eval(), neighbor_encoder.eval(), decoder.eval())

    impute_data_all.append(imputed_data)

# print(impute_data_all[0], type(impute_data_all[0]))
impute_data_all = np.concatenate(impute_data_all, axis=0)
impute_data_all = impute_data_all * (test_dataset.upper - test_dataset.lower) + test_dataset.lower
np.save(args.output_file, impute_data_all[:,::-1])
print('finish, imputed data is dump in {}'.format(args.output_file))
