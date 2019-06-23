import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V
from torch.distributions import Bernoulli
import random
# th.set_default_tensor_type('torch.cuda.FloatTensor')
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class TLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0,
                 dropout_method='pytorch', jit=False):
        super(TLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.c2cs = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))
        if th.cuda.is_available():
            self.mask = self.mask.cuda()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, t, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        # print h.size(), c.size(), 'f@ck'
        # print h.size(), c.size(), x.size()
        h = h.view(h.size(0), self.hidden_size)
        c = c.view(c.size(0), self.hidden_size)
        x = x.view(x.size(0), self.input_size)

        cs = self.c2cs(c).tanh()
        cs_o = cs / t
        # cs_o = cs / th.log(math.e + t)
        c_T = c - cs
        c_star = c_T + cs_o
        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)  # x: batch * input_size -> batch * hidden_size h: hidden_size -> hidden_size

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c_star, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
                c_t.data.set_(th.mul(c_t, self.mask).data)
                c_t.data *= 1.0/(1.0 - self.dropout)

        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                    h_t.data.set_(th.mul(h_t, self.mask).data)
                    h_t.data *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(h_t.size(0), -1)
        c_t = c_t.view(c_t.size(0), -1)
        return h_t, (h_t, c_t)

class MemoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, K, L=1, bias=True, dropout=0.0,
                 dropout_method='pytorch', jit=False):
        super(MemoryEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.K = K
        self.L = L
        self.dropout = dropout
        self.attn = nn.Linear(hidden_size, K, bias=False)

        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method
        self.lstm_cell = TLSTMCell(self.input_size, self.hidden_size, self.bias, self.dropout, self.dropout_method)

    def forward(self, x, interval, mask, hidden=None):
        if hidden == None:
            h0 = th.zeros(x.size(0), self.hidden_size)
            c0 = th.zeros(x.size(0), self.hidden_size)
            if th.cuda.is_available():
                h0, c0 = h0.cuda(), c0.cuda()
            hidden = (h0, c0)
            h0_ = th.zeros(x.size(0), self.hidden_size)
            c0_ = th.zeros(x.size(0), self.hidden_size)
            if th.cuda.is_available():
                h0_, c0_ = h0_.cuda(), c0_.cuda()

        hs = th.zeros(x.size(0), x.size(1), self.hidden_size)
        cs = th.zeros(x.size(0), x.size(1), self.hidden_size)
        if th.cuda.is_available():
            hs, cs = hs.cuda(), cs.cuda()

        for i in range(x.size(1)):
            hs[:, i, :], hidden = self.lstm_cell(x[:, i, :], interval[:, i], hidden)
            cs[:, i, :] = hidden[1]

        attn = self.attn(hs) * self.L # B * S * K
        attn = F.softmax(attn, dim=1) * mask.unsqueeze(2)
        attn = attn / th.sum(attn)
        context = attn.transpose(1,2).bmm(hs) # B * K * H

        return hs, (hs, cs), context


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, K, neighbor_num = 8, dropout=0.8):
        super(DecoderRNN, self).__init__()

        # Keep for reference
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.K = K
        self.select_attn = nn.Linear(hidden_size, K, bias=False)
        self.neighbor_attn = nn.Linear(hidden_size, K * 8, bias=False)
        self.lstm_cell = th.nn.LSTMCell(self.input_size, hidden_size)

        self.net = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),nn.Linear(hidden_size, input_size))

    def forward(self, input_seq, last_context, last_hidden, encoder_matrix, neighbor_matrix):
        # Get current hidden state from input word and last hidden state
        # print('[decoder] last_hidden', last_hidden.size())

        hidden = self.lstm_cell(input_seq,last_hidden)
        rnn_output = hidden[0]

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average

        beta = F.softmax(self.select_attn(rnn_output), dim=1).unsqueeze(2) # B * k B * K * h
        context = th.sum(beta * encoder_matrix, dim=1)

        neighbor_beta = F.softmax(self.neighbor_attn(rnn_output), dim=1).unsqueeze(2) # B * K * H
        # print neighbor_beta.shape, neighbor_matrix.shape

        neighbor_matrix = neighbor_matrix.view(neighbor_matrix.size(0), neighbor_matrix.size(1) * neighbor_matrix.size(2),
                                               neighbor_matrix.size(3))
        neighbor_context = th.sum(neighbor_beta * neighbor_matrix, dim = 1)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x H -> B x H

        concat_input = th.cat((rnn_output, context, neighbor_context), 1)
        output = self.net(concat_input)
        return output, context, hidden



def train_batch(input_batches, input_lengths, input_interval, input_mask, target_batches, target_mask,
                neighbor_input, neighbor_interval, neighbor_length,
               encoder, neighbor_encoder, decoder, encoder_optimizer, neighbor_encoder_optimizer, decoder_optimizer, clip=10, teacher_ratio = 0.8):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    batch_size = input_batches.size(0)

    tmp_neighbor_input = neighbor_input.view(neighbor_input.size(0) * neighbor_input.size(1), neighbor_input.size(2), \
                                             neighbor_input.size(3))
    tmp_neighbor_interval = neighbor_interval.view(neighbor_interval.size(0) * neighbor_interval.size(1), neighbor_interval.size(2),\
                                             neighbor_interval.size(3))
    tmp_neighbor_length = neighbor_length.view(neighbor_length.size(0) * neighbor_length.size(1), neighbor_length.size(2))

        # Run words through encoder
    encoder_outputs, encoder_hidden, encoder_context = encoder(input_batches, input_interval, input_mask, None)
    neighbor_encoder_outputs, neighbor_encoder_hidden, neighbor_encoder_context = neighbor_encoder(tmp_neighbor_input, tmp_neighbor_interval, tmp_neighbor_length, None)
    neighbor_encoder_context = neighbor_encoder_context.view(neighbor_input.size(0), neighbor_input.size(1), neighbor_encoder_context.size(1),neighbor_encoder_context.size(2))

    # Prepare input and output variables
    decoder_context = encoder_outputs[-1]
    input_lengths = input_lengths.unsqueeze(2)

    decoder_h = th.sum(encoder_hidden[0] * input_lengths, dim=1)
    decoder_c = th.sum(encoder_hidden[1] * input_lengths, dim=1)

    if th.cuda.is_available():
        decoder_h = th.zeros(decoder_h.size(0),decoder_h.size(1)).cuda()
        decoder_c = th.zeros(decoder_c.size(0),decoder_c.size(1)).cuda()
    decoder_hidden = (decoder_h, decoder_c)

    max_length = input_batches.size(1)
    all_decoder_outputs = V(th.zeros(batch_size, input_batches.size(1), input_batches.size(2)))
    # print all_decoder_outputs.size()
    if th.cuda.is_available():
        all_decoder_outputs = all_decoder_outputs.cuda()
        # print(type(all_decoder_outputs))
    # Move new Variables to CUDA
    decoder_input = th.zeros(input_batches.size(0), input_batches.size(2)).float()
    if th.cuda.is_available():
        decoder_input = decoder_input.cuda()

    # teacher forcing

    random_mask = Bernoulli(th.zeros(target_mask.size(0),target_mask.size(1),1)+0.9)
    random_sample = random_mask.sample()

    if th.cuda.is_available():
        random_sample = random_sample.cuda()
    for t in range(max_length):
        mask = target_mask[:, t] * random_sample[:,t]
        # print(mask)
        # import sys; sys.exit()
        decoder_output, decoder_context, decoder_hidden = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_context, neighbor_encoder_context
        )

        all_decoder_outputs[:,t] = decoder_output
        decoder_input = target_batches[:, t] * mask \
                            + (1 - mask) * decoder_output
    # target_mask = target_mask * (1 - random_sample)
    if th.sum(target_mask) == 0:
        return 0, 1
    loss = th.sum((all_decoder_outputs - target_batches) * (all_decoder_outputs - target_batches) * target_mask)
    loss.backward()
    ec = th.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    nc = th.nn.utils.clip_grad_norm_(neighbor_encoder.parameters(), clip)
    dc = th.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    neighbor_encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.cpu().detach().numpy(), th.sum(target_mask).cpu().detach().numpy() * target_batches.size(2)

def test_batch(input_batches, input_lengths, input_interval, input_mask, target_batches, target_mask, test_mask,
               neighbor_input, neighbor_interval, neighbor_length,
               encoder, neighbor_encoder, decoder):

    # Zero gradients of both optimizers
    batch_size = input_batches.size(0)

    tmp_neighbor_input = neighbor_input.view(neighbor_input.size(0) * neighbor_input.size(1), neighbor_input.size(2), \
                                             neighbor_input.size(3))
    tmp_neighbor_interval = neighbor_interval.view(neighbor_interval.size(0) * neighbor_interval.size(1), neighbor_interval.size(2),\
                                             neighbor_interval.size(3))
    tmp_neighbor_length = neighbor_length.view(neighbor_length.size(0) * neighbor_length.size(1), neighbor_length.size(2))

        # Run words through encoder
    encoder_outputs, encoder_hidden, encoder_context = encoder(input_batches, input_interval, input_mask, None)
    neighbor_encoder_outputs, neighbor_encoder_hidden, neighbor_encoder_context = neighbor_encoder(tmp_neighbor_input, tmp_neighbor_interval, tmp_neighbor_length, None)
    neighbor_encoder_context = neighbor_encoder_context.view(neighbor_input.size(0), neighbor_input.size(1), neighbor_encoder_context.size(1),neighbor_encoder_context.size(2))

#     print('decoder_input', decoder_input.size())
    decoder_context = encoder_outputs[-1]
    # print input_lengths.size(), encoder_outputs.size()
    input_lengths = input_lengths.unsqueeze(2)

    decoder_h = th.sum(encoder_hidden[0] * input_lengths, dim=1)
    decoder_c = th.sum(encoder_hidden[1] * input_lengths, dim=1)
    decoder_hidden = (decoder_h, decoder_c)

    max_length = input_batches.size(1)
    all_decoder_outputs = V(th.zeros(batch_size, input_batches.size(1), input_batches.size(2)))
    if th.cuda.is_available():
        all_decoder_outputs = all_decoder_outputs.cuda()
    decoder_input = th.zeros(input_batches.size(0), input_batches.size(2)).float()
    if th.cuda.is_available():
        decoder_input = decoder_input.cuda()
    # Run through decoder one time step at a time
    for t in range(max_length):
        decoder_output, decoder_context, decoder_hidden = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_context, neighbor_encoder_context, neighbor_data[:,:,t], neighbor_mask[:,:,t]
        )
        all_decoder_outputs[:,t] = decoder_output
        decoder_input = target_batches[:, t] * test_mask[:, t] \
                                + (1 - test_mask[:, t]) * decoder_output

    loss = th.sum((all_decoder_outputs - target_batches) * (all_decoder_outputs - target_batches) * target_mask)
    loss_abs = th.sum(th.abs(all_decoder_outputs - target_batches) * target_mask)
    # loss.backward()

    # print all_decoder_outputs.cpu().detach().numpy()[0,:5], target_batches.cpu().detach().numpy()[0,:5],'\r'
    # Clip gradient norm

    return loss.cpu().detach().numpy(), loss_abs.cpu().detach().numpy(), th.sum(target_mask).cpu().detach().numpy() * target_batches.size(2)


def impute_batch(input_batches, input_lengths, input_interval, input_mask, target_batches, test_mask,
               neighbor_input, neighbor_interval, neighbor_length,
               encoder, neighbor_encoder, decoder, clip=50):
    # Zero gradients of both optimizers
    batch_size = input_batches.size(0)
    tmp_neighbor_input = neighbor_input.view(neighbor_input.size(0) * neighbor_input.size(1), neighbor_input.size(2), \
                                             neighbor_input.size(3))
    tmp_neighbor_interval = neighbor_interval.view(neighbor_interval.size(0) * neighbor_interval.size(1), neighbor_interval.size(2),\
                                             neighbor_interval.size(3))
    tmp_neighbor_length = neighbor_length.view(neighbor_length.size(0) * neighbor_length.size(1), neighbor_length.size(2))

    # Run words through encoder
    encoder_outputs, encoder_hidden, encoder_context = encoder(input_batches, input_interval, input_mask, None)
    neighbor_encoder_outputs, neighbor_encoder_hidden, neighbor_encoder_context = neighbor_encoder(tmp_neighbor_input, tmp_neighbor_interval, tmp_neighbor_length, None)
    neighbor_encoder_context = neighbor_encoder_context.view(neighbor_input.size(0), neighbor_input.size(1), neighbor_encoder_context.size(1),neighbor_encoder_context.size(2))

    decoder_context = encoder_outputs[-1]
    # print input_lengths.size(), encoder_outputs.size()
    input_lengths = input_lengths.unsqueeze(2)

    decoder_h = th.sum(encoder_hidden[0] * input_lengths, dim=1)
    decoder_c = th.sum(encoder_hidden[1] * input_lengths, dim=1)
    decoder_hidden = (decoder_h, decoder_c)

    max_length = input_batches.size(1)
    all_decoder_outputs = V(th.zeros(batch_size, input_batches.size(1), input_batches.size(2)))
    if th.cuda.is_available():
        all_decoder_outputs = all_decoder_outputs.cuda()
    decoder_input = th.zeros(input_batches.size(0), input_batches.size(2)).float()
    if th.cuda.is_available():
        decoder_input = decoder_input.cuda()
    # Run through decoder one time step at a time
    for t in range(max_length):
        decoder_output, decoder_context, decoder_hidden = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_context, neighbor_encoder_context
        )
        all_decoder_outputs[:,t] = decoder_output
        decoder_input = target_batches[:, t] * test_mask[:, t] \
                                + (1 - test_mask[:, t]) * decoder_output
        all_decoder_outputs[:,t] = decoder_input

    return all_decoder_outputs.cpu().detach().numpy()

