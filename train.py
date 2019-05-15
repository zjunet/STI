# -*- coding: utf-8 -*-

import torch

import torch.nn as nn
import torch.autograd as autograd
import numpy as np

import torchvision
from torch.nn.utils import clip_grad_norm
import torchvision.transforms as transforms
import torch.nn.functional as F
from social_memory import *
import random
import time
import sys

from collections import Counter

load_dir = ''

torch.cuda.set_device(0)
# Device configuration
torch.set_default_tensor_type('torch.FloatTensor')

# Hyper-parameters
embed_size = 2
hidden_size = 16
num_layers = 2
batch_size = 256
K = 16
num_epochs = 3000

learning_rate = 1e-3
weight_decay = 0
dropout = 0.8
decoder_learning_ratio = 5
vec_size = 8
input_data = np.load(load_dir + 'input_data.npy')
idx = list(range(len(input_data)))
random.shuffle(idx)
train_idx = idx
test_idx = idx


input_size = input_data.shape[2]
S = input_data.shape[1]
L = [[1. * s * k / S / K + (1 - 1. * k / K) * (1 - 1. * s / S) for k in range(1, K + 1)] for s in range(1, S + 1)]
L = th.from_numpy(np.array(L))
# L = 1

# Inherit Dataset
class DataSet(torch.utils.data.Dataset):
    def __init__(self, load_dir, idx):
        super(DataSet, self).__init__()
        self.input_data = np.load(load_dir + 'input_data.npy')[idx].astype(dtype = np.float32)
        self.input_mask = np.load(load_dir + 'input_mask.npy')[idx].astype(dtype = np.float32)
        self.input_interval = np.load(load_dir + 'input_interval.npy')[idx].astype(dtype = np.float32)
        self.input_interval = np.expand_dims(self.input_interval, axis=2)
        self.input_length = np.load(load_dir + 'input_length.npy')[idx].astype(dtype = np.float32)
        self.u = np.load(load_dir + 'u.npy')[idx].astype(dtype = np.float32)
        self.neighbor_length = np.load(load_dir + 'neighbor_length.npy')[idx].astype(dtype = np.float32)
        self.neighbor_interval = np.load(load_dir + 'neighbor_interval.npy')[idx].astype(dtype = np.float32)
        self.neighbor_interval = np.expand_dims(self.neighbor_interval, axis=3).astype(dtype = np.float32)
        self.neighbor_data = np.load(load_dir + 'neighbor_data.npy')[idx].astype(dtype = np.float32)
        self.neighbor_origin = np.load(load_dir + 'neighbor_origin.npy')[idx].astype(dtype = np.float32)
        self.neighbor_mask = np.load(load_dir + 'neighbor_mask.npy')[idx].astype(dtype = np.float32)

        self.output_mask_all = np.load(load_dir + 'output_mask_all.npy')[idx]

        self.mask_in = (self.output_mask_all == 1).astype(dtype = np.float32)
        self.mask_out = (self.output_mask_all == 2).astype(dtype = np.float32)
        self.mask_all = (self.output_mask_all !=0).astype(dtype = np.float32)

    def __getitem__(self, index):
        return self.input_data[index], self.input_mask[index], self.input_interval[index], self.input_length[index],\
               self.u[index], self.mask_in[index], self.mask_out[index], self.mask_all[index], self.neighbor_data[index],\
               self.neighbor_interval[index], self.neighbor_length[index],self.neighbor_origin[index], self.neighbor_mask[index]

    def __len__(self):
        return len(self.input_data)

train_dataset = DataSet(load_dir, train_idx)
# Dataset
print('load successfully')
test_loader = torch.utils.data.DataLoader(DataSet(load_dir, test_idx), batch_size = batch_size)

# print(input_data.shape[1])
encoder = MemoryEncoder(input_size, embed_size ,hidden_size, K)
neighbor_encoder = MemoryEncoder(input_size, embed_size ,hidden_size, K)
decoder = DecoderRNN(input_size, embed_size, hidden_size, 6, K)

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

    for i, (i_data, i_mask, i_interval, i_length, u_all, m_in, m_out, m_all, n_input, n_inter, n_len, n_data, n_mask) in enumerate(train_loader):
        input = i_data
        m_all = m_all.unsqueeze(2)
        m_in = m_in.unsqueeze(2)
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
            n_data = n_data.cuda()
            n_mask = n_mask.cuda()

        start = time.time()
        loss, num = train_batch(input, i_length, i_interval,i_mask, u_all, m_in, n_input, n_inter, n_len, n_data, n_mask,
                encoder.train(), neighbor_encoder.train(), decoder.train(), encoder_optimizer, neighbor_encoder_optimizer, decoder_optimizer)
        loss_all += loss
        num_all += num
    # print(time_since(start, 1.*(i+1)/len(train_loader)))
    print('train: epoch{}'.format(epoch), loss_all*1./num_all)
    torch.save(encoder.state_dict(), load_dir + 'encoder.pkl')
    torch.save(neighbor_encoder.state_dict(), load_dir + 'neighbor_encoder.pkl')
    torch.save(decoder.state_dict(), load_dir + 'decoder.pkl')

    last_five = []
    if epoch % 200 == 0:
        loss_all, loss_abs_all,num_all = 0, 0, 0
        for i, (i_data, i_mask, i_interval, i_length, u_all, m_in, m_out, m_all, n_input, n_inter, n_len, n_data, n_mask) in enumerate(test_loader):
            input = i_data
            m_out = m_out.unsqueeze(2)
            m_in = m_in.unsqueeze(2)
            if th.cuda.is_available():
                input = input.cuda()
                i_length = i_length.cuda()
                i_interval = i_interval.cuda()
                i_mask = i_mask.cuda()
                m_out = m_out.cuda()
                m_in = m_in.cuda()
                u_all = u_all.cuda()
                n_input = n_input.cuda()
                n_inter = n_inter.cuda()
                n_len = n_len.cuda()
                n_data = n_data.cuda()
                n_mask = n_mask.cuda()

            start = time.time()
            loss, loss_abs, num = test_batch(input, i_length, i_interval, i_mask, u_all, m_out, m_in, n_input, n_inter, n_len, n_data, n_mask,
                    encoder.eval(), neighbor_encoder.eval(), decoder.eval(),  encoder_optimizer, neighbor_encoder_optimizer, decoder_optimizer)
            loss_all += loss
            loss_abs_all += loss_abs
            num_all += num

        print('evaluate: epoch {} rmse:{}, mae:{}'.format(epoch, np.sqrt(loss_all*1./num_all), loss_abs_all*1./num_all))

