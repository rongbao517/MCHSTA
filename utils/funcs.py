import torch
import numpy as np
from data import MyDataLoader, MyDataLoader2
import pandas as pd
from .config import DATA_PATHS

def load_all_adj(args,device):


    if DATA_PATHS[args.datasets[0]]['adj'][-3:] == 'npy':
        adj_pems04 = np.load(DATA_PATHS[args.datasets[0]]['adj'], allow_pickle=True).astype(np.int64)
    elif DATA_PATHS[args.datasets[0]]['adj'][-3:]=='csv':
        adj_pems04 = get_adjacency_matrix(distance_df_filename=DATA_PATHS[args.datasets[0]]['adj'],
                                   num_of_vertices=DATA_PATHS[args.datasets[0]]['num_of_vertices'])

    if DATA_PATHS[args.datasets[1]]['adj'][-3:] == 'npy':
        adj_pems07 = np.load(DATA_PATHS[args.datasets[1]]['adj'], allow_pickle=True).astype(np.int64)
    elif DATA_PATHS[args.datasets[1]]['adj'][-3:]=='csv':
        adj_pems07 = get_adjacency_matrix(distance_df_filename=DATA_PATHS[args.datasets[1]]['adj'],
                                   num_of_vertices=DATA_PATHS[args.datasets[1]]['num_of_vertices'])

    if DATA_PATHS[args.datasets[2]]['adj'][-3:] == 'npy':
        adj_pems08 = np.load(DATA_PATHS[args.datasets[2]]['adj'], allow_pickle=True).astype(np.int64)
    elif DATA_PATHS[args.datasets[2]]['adj'][-3:] == 'csv':
        adj_pems08 = get_adjacency_matrix(distance_df_filename=DATA_PATHS[args.datasets[2]]['adj'],
                                          num_of_vertices=DATA_PATHS[args.datasets[2]]['num_of_vertices'])


    return torch.tensor(adj_pems04).to(device), torch.tensor(adj_pems07).to(device), torch.tensor(adj_pems08).to(device)


def load_time(args, feat_dir):
    if feat_dir[-3:]=='npy':
        file_data = np.load(feat_dir)
    elif feat_dir[-3:]=='npz':
        if 'PEMS' in feat_dir:
            file_data = np.load(feat_dir)['data']
        else:
            file_data = np.load(feat_dir)['arr_0']
    # data = file_data['data']
    if len(file_data.shape) ==3:
        data = file_data[:,:,0]
    else:
        data = file_data
    t_len,N_node = data.shape

    alltime = np.arange(t_len).astype(np.float64)
    timeInterval = DATA_PATHS[args.dataset]['Interval']
    alltime = (alltime%(60*24*7/timeInterval))/(60*24*7/timeInterval)
    alltime = alltime.reshape(alltime.shape[0],-1)
    time_len = data.shape[0]
    seq_len = args.seq_len
    pre_len = args.pre_len
    split_ratio = args.split_ratio
    if args.split_ratio < 1:
        train_size = int(time_len * split_ratio)
        val_size = int(time_len * (1 - split_ratio) / 3)
    else:
        train_size = int(split_ratio * 24 * 60 / DATA_PATHS[args.dataset]['Interval'])
        val_size = int((time_len - train_size) / 3)

    train_data = alltime[:train_size]
    val_data = alltime[train_size:train_size + val_size]
    test_data = alltime[train_size + val_size:time_len]
    # if args.split_ratio < 1:
    #     if args.labelrate != 100:
    #         train_data = train_data[args.start:args.start + args.new_train_size]

    train_X, val_X, test_X = list(), list(), list()

    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i: i + seq_len]))
    for i in range(len(val_data) - seq_len - pre_len):
        val_X.append(np.array(val_data[i: i + seq_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i: i + seq_len]))

    train_X = np.array(train_X)
    val_X = np.array(val_X)
    test_X = np.array(test_X)

    print(f"time_train {train_X.shape}  time_val {val_X.shape}  time_test {test_X.shape}")

    return train_X, val_X, test_X


def load_data(args, scaler=None, visualize=False, distribution=False):

    time = False
    feat_dir = DATA_PATHS[key]['feat']
    adj_dir = DATA_PATHS[key]['adj']
    num_of_vertices = DATA_PATHS[key]['num_of_vertices']
    # feat_dir = DATA_PATHS[args.dataset]['feat']
    # adj_dir = DATA_PATHS[args.dataset]['adj']
    # num_of_vertices = DATA_PATHS[args.dataset]['num_of_vertices']

    train_X, train_Y, val_X, val_Y, test_X, test_Y, max_speed, scaler = load_graphdata_channel(args, feat_dir, time,
                                                                                               scaler,
                                                                                               visualize=visualize)
    time_train, time_val, time_test = load_time(args,feat_dir)

    print("***",train_X.shape)
    print("***",train_Y.shape)
    print('***',time_train.shape)
    if adj_dir[-3:] == 'npy':
        adj = np.load(adj_dir, allow_pickle=True).astype(np.int64)
    elif adj_dir[-3:]=='csv':
        # adj = pd.read_csv(adj_dir).values
        adj = get_adjacency_matrix(distance_df_filename=adj_dir, num_of_vertices=num_of_vertices)

    train_dataloader = MyDataLoader2(torch.FloatTensor(train_X), torch.FloatTensor(train_Y), torch.FloatTensor(time_train),
                                    batch_size=args.batch_size)
    val_dataloader = MyDataLoader2(torch.FloatTensor(val_X), torch.FloatTensor(val_Y),torch.FloatTensor(time_val), batch_size=args.batch_size)
    test_dataloader = MyDataLoader2(torch.FloatTensor(test_X), torch.FloatTensor(test_Y),torch.FloatTensor(time_test), batch_size=args.batch_size)


    return train_dataloader, val_dataloader, test_dataloader, torch.tensor(adj), max_speed, scaler


def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    type_: str, {connectivity, distance}
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A

def load_distribution(feat_dir):
    file_data = np.load(feat_dir)
    data = file_data['data']
    where_are_nans = np.isnan(data)
    data[where_are_nans] = 0
    where_are_nans = (data != data)
    data[where_are_nans] = 0
    data = data[:, :, 0]  # flow only
    data = np.array(data)

    return data
###########################################
def load_graphdata_channel(args, feat_dir, time, scaler=None, visualize=False):

    if feat_dir[-3:]=='npy':
        file_data = np.load(feat_dir)
    elif feat_dir[-3:]=='npz':
        if 'PEMS' in feat_dir:
            file_data = np.load(feat_dir)['data']
        else:
            file_data = np.load(feat_dir)['arr_0']

    if len(file_data.shape) ==3:
        data = file_data[:,:,0]
    else:
        data = file_data
    print(data.shape)
    where_are_nans = np.isnan(data)
    data[where_are_nans] = 0
    where_are_nans = (data != data)
    data[where_are_nans] = 0

    if time:
        num_data, num_sensor = data.shape
        data = np.expand_dims(data, axis=-1)
        data = data.tolist()

        for i in range(num_data):
            time = (i % 288.0) / 288.0
            for j in range(num_sensor):
                data[i][j].append(time)

        data = np.array(data)

    max_val = np.max(data)
    time_len = data.shape[0]
    seq_len = args.seq_len
    pre_len = args.pre_len
    split_ratio = args.split_ratio
    if args.split_ratio<1:
        train_size = int(time_len * split_ratio)
        val_size = int(time_len * (1 - split_ratio) / 3)
    else:

        train_size = int(args.split_ratio * 24 * 60 / DATA_PATHS[key]['Interval'])
        val_size = int((time_len - train_size) / 3)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:time_len]

    if args.split_ratio < 1:
        if args.labelrate != 100:
            import random
            new_train_size = int(train_size * args.labelrate / 100)
            start = random.randint(0, train_size - new_train_size - 1)
            train_data = train_data[start:start+new_train_size]
            args.start = start
            args.new_train_size =new_train_size

    train_X, train_Y, val_X, val_Y, test_X, test_Y = list(), list(), list(), list(), list(), list()

    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i: i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(val_data) - seq_len - pre_len):
        val_X.append(np.array(val_data[i: i + seq_len]))
        val_Y.append(np.array(val_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i: i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))

    if visualize:
        test_X = test_X[-288:]
        test_Y = test_Y[-288:]

    if args.labelrate != 0:
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
    val_X = np.array(val_X)
    val_Y = np.array(val_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    TXTPATH = args.save_path / 'detail_log.txt'

    print(f"train {train_X.shape}--{train_Y.shape}  val {val_X.shape}--{val_Y.shape}  test {test_X.shape}--{test_Y.shape}")
    with open(TXTPATH, 'a') as f:
        f.write(f"train {train_X.shape}--{train_Y.shape}  val {val_X.shape}--{val_Y.shape}  test {test_X.shape}--{test_Y.shape}")



    return train_X, train_Y, val_X, val_Y, test_X, test_Y, max_val, scaler

def masked_loss(y_pred, y_true):
    mask_true = (y_true > 0.01).float()
    mask = mask_true
    mask /= mask.mean()
    mae_loss = torch.abs(y_pred - y_true)
    mse_loss = torch.square(y_pred - y_true)
    mape_loss = mae_loss / y_true
    mae_loss = mae_loss * mask
    mse_loss = mse_loss * mask
    mape_loss = mape_loss * mask
    mae_loss[mae_loss != mae_loss] = 0
    mse_loss[mse_loss != mse_loss] = 0
    mape_loss[mape_loss != mape_loss] = 0

    return mae_loss.mean(), torch.sqrt(mse_loss.mean()), mape_loss.mean()

def masked_loss2(y_pred, y_true):
    # print(y_pred)
    # print(y_true)
    mask_true = (y_true > 0.01).astype(np.float32)
    # mask_pred = (y_pred > 0.01).float()
    # mask = torch.mul(mask_true, mask_pred)
    mask = mask_true
    mask /= mask.mean()
    mae_loss = np.abs(y_pred - y_true)
    mse_loss = np.square(y_pred - y_true)
    mape_loss = mae_loss / y_true
    mae_loss = mae_loss * mask
    mse_loss = mse_loss * mask
    mape_loss = mape_loss * mask
    mae_loss[mae_loss != mae_loss] = 0
    mse_loss[mse_loss != mse_loss] = 0
    mape_loss[mape_loss != mape_loss] = 0

    return mae_loss.mean(), np.sqrt(mse_loss.mean()), mape_loss.mean()

