import numpy as np
import copy
import random

def load_data(data, social_network, missing_label):
    input_data_list = []
    input_time_list = []
    input_mask_list = []
    input_interval_list = []
    input_length_list = []

    output_mask_all_list = []
    origin_u = []

    max_length = data.shape[1]
    channel = data.shape[2]
    upper = np.max(data)

    data_tmp = copy.deepcopy(data)
    data_tmp[data_tmp == missing_label] = upper
    lower = np.min(data_tmp)
    del data_tmp

    for u in data:

        idx_u = np.arange(len(u))
        # print idx_u.shape
        u[u!=missing_label] = (u[u!=missing_label] - lower) / (upper-lower)

        valid_data = u[u[:,0] != missing_label]
        valid_real_idx = idx_u[u[:,0]!=missing_label]

        input_idx = valid_real_idx
        input_data = np.zeros((max_length, channel))
        input_mask = np.zeros(max_length)
        input_length = np.zeros(max_length)
        input_interval = np.zeros(max_length) + 1
        input_time = np.zeros(max_length)
        output_mask_all = np.zeros(max_length)

        input_data[:len(input_idx)] = valid_data
        input_time[:len(input_idx)] = valid_real_idx
        input_mask[:len(input_idx)] = 1
        input_length[min(len(input_idx)-1, max_length - 1)] = 1

        input_real_idx = valid_real_idx
        output_mask_all[input_real_idx.astype(dtype=int)] = 1

        input_interval[0] = 1
        input_interval[1:len(input_idx)] = input_real_idx[1:] - input_real_idx[:-1]

        input_data_list.append(input_data)
        input_time_list.append(input_time)
        input_mask_list.append(input_mask)
        input_interval_list.append(input_interval)
        input_length_list.append(input_length)

        origin_u.append(list(reversed(u)))
        output_mask_all_list.append(list(reversed(output_mask_all)))

    input_data_list = np.array(input_data_list).astype(dtype = np.float32)
    input_time_list = np.array(input_time_list).astype(dtype = np.float32)
    input_mask_list = np.array(input_mask_list).astype(dtype = np.float32)
    input_interval_list = np.array(input_interval_list).astype(dtype = np.float32)
    input_length_list = np.array(input_length_list).astype(dtype = np.float32)
    output_mask_all_list = np.array(output_mask_all_list).astype(dtype = np.float32)
    origin_u = np.array(origin_u).astype(dtype = np.float32)

    max_num = 8

    neighbor_length = np.zeros((input_length_list.shape[0], max_num, input_length_list.shape[1])).astype(dtype = np.float32)
    neighbor_interval = np.zeros((input_interval_list.shape[0], max_num, input_interval_list.shape[1])).astype(dtype = np.float32)
    neighbor_time = np.zeros((input_interval_list.shape[0], max_num, input_interval_list.shape[1])).astype(dtype = np.float32)
    neighbor_data = np.zeros((input_data_list.shape[0], max_num, input_data_list.shape[1], input_data_list.shape[2])).astype(dtype = np.float32)

    for i, neighbors in enumerate(social_network):
        for j in range(max_num):
            m = random.randint(0, len(neighbors))
            m = (neighbors + [i])[m]
            neighbor_length[i][j] = input_length_list[m]
            neighbor_interval[i][j] = input_interval_list[m]
            neighbor_time[i][j] = input_time_list[m]
            neighbor_data[i][j] = input_data_list[m]

    return input_data_list, input_time_list, input_mask_list, input_interval_list, input_length_list, \
           output_mask_all_list, origin_u, neighbor_length, neighbor_interval, neighbor_time, neighbor_data, lower, upper




