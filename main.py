import argparse
import torch
import copy
import time
import os
import numpy as np
import torch.optim as optim
from utils.funcs import load_data, load_all_adj
from utils.funcs import masked_loss,masked_loss2
from utils.vec import generate_vector
from pathlib import Path

from design import MDTGAT,Domain_classifier_DG,DomainDiscriminator

from visualization import result_visualization


def increment_path(path, exist_ok=False, sep="", mkdir=False):

    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)


    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

# python main.py --split_ratio 1 --save_path "less1_Design_MDTGAT" --device
def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='pems-bay', help='dataset')
    parser.add_argument('--datasets', type=list, default=['pems-bay','shenzhen','chengdu'], help='datasets')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--division_seed', type=int, default=0, help='division_seed')
    parser.add_argument('--model', type=str, default='DASTNet', help='model')
    parser.add_argument('--labelrate', type=float, default=10, help='percent')
    parser.add_argument('--patience', type=int, default=200, help='patience')
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--vec_dim", type=int, default=64)
    parser.add_argument("--enc_dim", type=int, default=64)
    parser.add_argument("--walk_length", "--wl", type=int, default=8)
    parser.add_argument("--num_walks", type=int, default=200)
    parser.add_argument("--theta", type=float, default=1)
    parser.add_argument("--p", type=float, default=1)
    parser.add_argument("--q", type=float, default=1)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument('--device', type=str, default='2', help='CUDA Device')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pre_len", type=int, default=3)
    parser.add_argument("--split_ratio", type=float, default=0.7,help='<1 is ratio, >=1 is day')
    parser.add_argument("--base_split_ratio", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument('--val', action='store_true', default=False, help='eval')
    parser.add_argument('--test', action='store_true', default=False, help='test')
    parser.add_argument('--train', action='store_true', default=False, help='train')
    parser.add_argument('--etype', type=str, default="gin", choices=["gin"], help='feature type')
    parser.add_argument('--save_path', type=str, default="MC_HSTA", help='feature type')
    parser.add_argument('--exist_ok', action='store_true', default=False, help='train')
    return parser.parse_args()


def train(dur, model, optimizer, total_step, start_step):
    t0 = time.time()
    train_mae, val_mae, train_rmse, val_rmse, train_acc = list(), list(), list(), list(), list()
    train_correct = 0

    model.train()
    if type == 'pretrain':
        domain_classifier.train()

    for i, (feat, label,t) in enumerate(train_dataloader.get_iterator()):
        Reverse = False
        # print(i)
        if i > 0:
            if train_acc[-1] > (com_num/all_num):
                Reverse = True
        p = float(i + start_step) / total_step
        constant = 2. / (1. + np.exp(-10 * p)) - 1
        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        t = torch.FloatTensor(t).to(device)
        if torch.sum(label) <= 0.001:
            continue

        optimizer.zero_grad()

        if args.model not in ['DCRNN', 'STGCN', 'HA']:
            if type == 'pretrain':
                pred, shared_pems04_feat, shared_pems07_feat, shared_pems08_feat = model(vec_pems04, vec_pems07,
                                                                                         vec_pems08, feat,t, False)
            elif type == 'fine-tune':
                pred = model(vec_pems04, vec_pems07, vec_pems08, feat, t, False)

            if type == 'pretrain':
                pems04_pred = domain_classifier(shared_pems04_feat, constant, Reverse)
                pems07_pred = domain_classifier(shared_pems07_feat, constant, Reverse)
                pems08_pred = domain_classifier(shared_pems08_feat, constant, Reverse)

                pems04_label = 0 * torch.ones(pems04_pred.shape[0]).long().to(device)
                pems07_label = 1 * torch.ones(pems07_pred.shape[0]).long().to(device)
                pems08_label = 2 * torch.ones(pems08_pred.shape[0]).long().to(device)

                pems04_pred_label = pems04_pred.max(1, keepdim=True)[1]
                pems04_correct = pems04_pred_label.eq(pems04_label.view_as(pems04_pred_label)).sum()
                pems07_pred_label = pems07_pred.max(1, keepdim=True)[1]
                pems07_correct = pems07_pred_label.eq(pems07_label.view_as(pems07_pred_label)).sum()
                pems08_pred_label = pems08_pred.max(1, keepdim=True)[1]
                pems08_correct = pems08_pred_label.eq(pems08_label.view_as(pems08_pred_label)).sum()

                pems04_loss = domain_criterion(pems04_pred, pems04_label)
                pems07_loss = domain_criterion(pems07_pred, pems07_label)
                pems08_loss = domain_criterion(pems08_pred, pems08_label)

                domain_loss = pems04_loss + pems07_loss + pems08_loss

        if type == 'pretrain':
            train_correct = pems04_correct + pems07_correct + pems08_correct
        mae_train, rmse_train, mape_train = masked_loss(pred, label)
        if type == 'pretrain':
            loss = mae_train + args.beta * (args.theta * domain_loss)
        elif type == 'fine-tune':
            loss = mae_train

        loss.backward()
        optimizer.step()

        train_mae.append(mae_train.item())
        train_rmse.append(rmse_train.item())

        if type == 'pretrain':
            train_acc.append(train_correct.item() / all_num)
        elif type == 'fine-tune':
            train_acc.append(0)
    if type == 'pretrain':
        domain_classifier.eval()
    model.eval()

    for i, (feat, label,t) in enumerate(val_dataloader.get_iterator()):
        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        t = torch.FloatTensor(t).to(device)
        if torch.sum(label) <= 0.001:
            continue
        pred = model(vec_pems04, vec_pems07, vec_pems08, feat,t, True)
        mae_val, rmse_val, mape_val = masked_loss(pred, label)
        val_mae.append(mae_val.item())
        val_rmse.append(rmse_val.item())

    test_mae, test_rmse, test_mape = test()
    dur.append(time.time() - t0)
    return np.mean(train_mae), np.mean(train_rmse), np.mean(val_mae), np.mean(
        val_rmse), test_mae, test_rmse, test_mape, np.mean(train_acc)


def test():
    if type == 'pretrain':
        domain_classifier.eval()
    model.eval()

    test_mape, test_rmse, test_mae = list(), list(), list()

    for i, (feat, label,t) in enumerate(test_dataloader.get_iterator()):
        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        t = torch.FloatTensor(t).to(device)
        if torch.sum(label) <= 0.001:
            continue

        pred = model(vec_pems04, vec_pems07, vec_pems08, feat, t, True)

        mae_test, rmse_test, mape_test = masked_loss(pred, label)

        test_mae.append(mae_test.item())
        test_rmse.append(rmse_test.item())
        test_mape.append(mape_test.item())

    test_rmse = np.mean(test_rmse)
    test_mae = np.mean(test_mae)
    test_mape = np.mean(test_mape)

    return test_mae, test_rmse, test_mape


def test_result_save():
    if type == 'pretrain':
        domain_classifier.eval()
    model.eval()

    test_mape, test_rmse, test_mae = list(), list(), list()
    YS,YS_pred =list(), list()
    with torch.no_grad():
        for i, (feat, label,t) in enumerate(test_dataloader.get_iterator()):
            feat = torch.FloatTensor(feat).to(device)
            label = torch.FloatTensor(label).to(device)
            t = torch.FloatTensor(t).to(device)

            if torch.sum(label) <= 0.001:
                continue

            pred = model(vec_pems04, vec_pems07, vec_pems08, feat, t, True)
            mae_test, rmse_test, mape_test = masked_loss(pred, label)
            YS_pred.append(pred.cpu().numpy())
            YS.append(label.cpu().numpy())

            test_mae.append(mae_test.item())
            test_rmse.append(rmse_test.item())
            test_mape.append(mape_test.item())

        YS_pred = np.vstack(YS_pred)
        YS =  np.vstack(YS)
    np.save(str(args.save_path/'prediction.npy'), YS_pred)
    np.save(str(args.save_path/'groundtruth.npy'), YS)
    print(f'save result YS {YS.shape} YS_pred {YS_pred.shape}')
    with open(TXTPATH, 'a') as f:
        try:
            for i in range(YS.shape[1]):
                mae_test, rmse_test, mape_test = masked_loss2(YS_pred[:, i, :], YS[:, i, :])
                test_rmse = np.mean(rmse_test)
                test_mae = np.mean(mae_test)
                test_mape = np.mean(mape_test)
                print(f'{i}/{YS.shape[1]} mae: {test_mae: .3f}, rmse: {test_rmse: .3f}, mape: {test_mape * 100: .3f}')
                f.write(f'{i}/{YS.shape[1]} mae: {test_mae: .3f}, rmse: {test_rmse: .3f}, mape: {test_mape * 100: .3f}')
        except:
            pass
    test_rmse = np.mean(test_rmse)
    test_mae = np.mean(test_mae)
    test_mape = np.mean(test_mape)

    return test_mae, test_rmse, test_mape


def model_train(args, model, optimizer):
    dur = []
    epoch = 1
    best = 999999999999999
    acc = list()

    step_per_epoch = train_dataloader.get_num_batch()
    total_step = 200 * step_per_epoch
    with open(TXTPATH, 'a') as f:
        f.write(f'\n\n{type} -- dataset {dataset}')

    while epoch <= args.epoch:
        start_step = epoch * step_per_epoch
        if type == 'fine-tune' and epoch > 1000:
            args.val = True
        mae_train, rmse_train, mae_val, rmse_val, mae_test, rmse_test, mape_test, train_acc = train(dur, model,
                                                                                                    optimizer,
                                                                                                    total_step,
                                                                                                    start_step)
        print(
            f'Epoch {epoch}/{args.epoch} | acc_train: {train_acc: .4f} | mae_train: {mae_train: .4f} | rmse_train: {rmse_train: .4f} | mae_val: {mae_val: .4f} | rmse_val: {rmse_val: .4f} | mae_test: {mae_test: .4f} | rmse_test: {rmse_test: .4f} | mape_test: {mape_test: .4f} | Time(s) {dur[-1]: .4f}')
        with open(TXTPATH, 'a') as f:
            f.write(
                f'\nEpoch {epoch}/{args.epoch} | acc_train: {train_acc: .4f} | mae_train: {mae_train: .4f} | rmse_train: {rmse_train: .4f} | mae_val: {mae_val: .4f} | rmse_val: {rmse_val: .4f} | mae_test: {mae_test: .4f} | rmse_test: {rmse_test: .4f} | mape_test: {mape_test: .4f} | Time(s) {dur[-1]: .4f}')
        epoch += 1
        acc.append(train_acc)
        if mae_val <= best:
            if type == 'fine-tune':  # and mae_val > 0.001:
                best = mae_val
                state = dict([('model', copy.deepcopy(model.state_dict())),
                              ('optim', copy.deepcopy(optimizer.state_dict())),
                              ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
                cnt = 0
            elif type == 'pretrain':
                best = mae_val
                state = dict([('model', copy.deepcopy(model.state_dict())),
                              ('optim', copy.deepcopy(optimizer.state_dict())),
                              ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
                cnt = 0

        else:
            cnt += 1
        # print(cnt)
        if cnt == args.patience or epoch > args.epoch:
            print(f'Stop!!')
            print(f'Avg acc: {np.mean(acc)}')
            break
    print("Optimization Finished!")
    return state


args = arg_parse(argparse.ArgumentParser())

SAVEPATH = Path('./save/'+args.save_path)
args.save_path = increment_path(SAVEPATH, exist_ok=args.exist_ok)
import shutil
import sys
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
currentPython = sys.argv[0]
shutil.copy2(currentPython, args.save_path)
shutil.copy2('design.py', args.save_path)


device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

TXTPATH = args.save_path/'detail_log.txt'
print(f'save path: {args.save_path}')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.labelrate > 100:
    args.labelrate = 100

SPLIT_RADIO = args.split_ratio

adj_pems04, adj_pems07, adj_pems08 = load_all_adj(args,device)

print(f'adj_pems04{adj_pems04.shape}--adj_pems07{adj_pems07.shape}--adj_pems08{adj_pems08.shape}')
with open(TXTPATH, 'a') as f:
    f.write(f'adj_pems04{adj_pems04.shape}--adj_pems07{adj_pems07.shape}--adj_pems08{adj_pems08.shape}')


all_num = adj_pems04.shape[0]+adj_pems07.shape[0]+adj_pems08.shape[0]
vec_pems04 = vec_pems07 = vec_pems08 = None, None, None

cur_dir = os.getcwd()
if cur_dir[-2:] == 'sh':
    cur_dir = cur_dir[:-2]

pems04_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', f'node2vec_{args.datasets[0]}_{args.vec_dim}_vecdim9.pkl')
pems07_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', f'node2vec_{args.datasets[1]}_{args.vec_dim}_vecdim9.pkl')
pems08_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', f'node2vec_{args.datasets[2]}_{args.vec_dim}_vecdim9.pkl')

DATASET = args.dataset
with open(TXTPATH, 'a') as f:
    f.write(f'emb_pems04 {pems04_emb_path} \nemb_pems07 {pems07_emb_path}\nemb_pems08{pems08_emb_path}\n')

if os.path.exists(pems04_emb_path):
    print(f'Loading {args.datasets[0]} embedding...')
    vec_pems04 = torch.load(pems04_emb_path, map_location='cpu')
    vec_pems04 = vec_pems04.to(device)
else:
    print(f'Generating {args.datasets[0]} embedding...')
    args.dataset = args.datasets[0]
    vec_pems04, _ = generate_vector(args)
    vec_pems04 = vec_pems04.to(device)
    print(f'Saving {args.datasets[0]} embedding...')
    torch.save(vec_pems04.cpu(), pems04_emb_path)

if os.path.exists(pems07_emb_path):
    print(f'Loading {args.datasets[0]} embedding...')
    vec_pems07 = torch.load(pems07_emb_path, map_location='cpu')
    vec_pems07 = vec_pems07.to(device)
else:
    print(f'Generating {args.datasets[0]} embedding...')
    args.dataset = args.datasets[1]
    vec_pems07, _ = generate_vector(args)
    vec_pems07 = vec_pems07.to(device)
    print(f'Saving {args.datasets[0]} embedding...')
    torch.save(vec_pems07.cpu(), pems07_emb_path)

if os.path.exists(pems08_emb_path):
    print(f'Loading {args.datasets[0]} embedding...')
    vec_pems08 = torch.load(pems08_emb_path, map_location='cpu')
    vec_pems08 = vec_pems08.to(device)
else:
    print(f'Generating {args.datasets[0]} embedding...')
    args.dataset = args.datasets[2]
    vec_pems08, _ = generate_vector(args)
    vec_pems08 = vec_pems08.to(device)
    print(f'Saving {args.datasets[0]} embedding...')
    torch.save(vec_pems08.cpu(), pems08_emb_path)
print(f'Successfully load embeddings, {args.datasets[0]}: {vec_pems04.shape}, '
      f'{args.datasets[1]}: {vec_pems07.shape}, {args.datasets[2]}: {vec_pems08.shape}')

with open(TXTPATH, 'a') as f:
    f.write(f'Successfully load embeddings, {args.datasets[0]}: {vec_pems04.shape},'
            f' {args.datasets[1]}: {vec_pems07.shape}, {args.datasets[2]}: {vec_pems08.shape}')

domain_criterion = torch.nn.NLLLoss()

domain_classifier = DomainDiscriminator(args.enc_dim, class_num=3,device=device).to(device)

state, g = None, None


batch_seen = 0
cur_dir = os.getcwd()
if cur_dir[-2:] == 'sh':
    cur_dir = cur_dir[:-2]
assert args.model in ["DASTNet"]

bak_epoch = args.epoch
bak_val = args.val
bak_test = args.test
type = 'pretrain'
args.dataset = DATASET
cnt = 0
pretrain_model_path = os.path.join('{}'.format(cur_dir), 'pretrained', f'transfer_models_{args.dataset}_prelen_{args.pre_len}_'
                                                                       f'new_model{cnt:03}_{args.model}_epoch_{args.epoch}.pkl')
if os.path.exists(pretrain_model_path):
    for n in range(2, 9999):
        cnt = cnt+1
        pretrain_model_path = os.path.join('{}'.format(cur_dir), 'pretrained',
                                           f'transfer_models_{args.dataset}_prelen_{args.pre_len}_'
                                           f'new_model{cnt:03}_{args.model}_epoch_{args.epoch}.pkl')
        if not os.path.exists(pretrain_model_path):  #
            break

with open(TXTPATH, 'a') as f:
    f.write(f'pretrain_model_path {pretrain_model_path} \n')

args.split_ratio = args.base_split_ratio
if os.path.exists(pretrain_model_path):
    print(f'Loading pretrained model at {pretrain_model_path}')
    state = torch.load(pretrain_model_path, map_location='cpu')
    dataset_bak = args.dataset
    dataset = args.dataset
    com_num = 0
else:
    print(f'No existing pretrained model at {pretrain_model_path}')
    args.val = args.test = False
    dataset_bak = args.dataset
    labelrate_bak = args.labelrate
    args.labelrate = 100
    dataset_count = 0

    for dataset in [item for item in args.datasets if item not in [DATASET]]:
        dataset_count = dataset_count + 1

        print(
            f'\n\n****************************************************************************************************************')
        print(f'dataset: {dataset}, model: {args.model}, pre_len: {args.pre_len}, labelrate: {args.labelrate}')
        print(
            f'****************************************************************************************************************\n\n')

        if args.datasets.index(dataset) == 0:
            g = vec_pems04
        elif args.datasets.index(dataset) == 1:
            g = vec_pems07
        elif args.datasets.index(dataset) == 2:
            g = vec_pems08
        com_num=g.shape[0]
        args.dataset = dataset
        train_dataloader, val_dataloader, test_dataloader, adj, max_speed, scaler = load_data2(args)
        model = MDTGAT(input_dim=args.vec_dim,hidden_dim=args.hidden_dim,seq_len = args.seq_len, out_len=args.pre_len,
                       encode_dim=args.enc_dim, datasets=args.datasets,
                       dataset=args.dataset, ft_dataset=dataset_bak, adj_pems04=adj_pems04, adj_pems07=adj_pems07,
                       adj_pems08=adj_pems08,batch_size=args.batch_size,device=device).to(device)
        optimizer = optim.Adam([{'params': model.parameters()},
                                {'params': domain_classifier.parameters()}], lr=args.learning_rate)

        if dataset_count != 1:
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optim'])

        state = model_train(args, model, optimizer)

    print(f'Saving model to {pretrain_model_path} ...')
    torch.save(state, pretrain_model_path)
    torch.save(state, args.save_path/f'savee_poch_{args.epoch}.pkl')
    args.dataset = dataset_bak
    dataset = args.dataset
    args.labelrate = labelrate_bak
    args.val = bak_val
    args.test = bak_test

type = 'fine-tune'
args.split_ratio = SPLIT_RADIO
args.epoch = 50

print(f'\n\n******************************fine-tune****************************************************')
print(
    f'dataset: {args.dataset}, model: {args.model}, pre_len: {args.pre_len}, labelrate: {args.labelrate}, seed: {args.division_seed}')
print(f'*******************************************************************************************\n\n')

if args.datasets.index(dataset) == 0:
    g = vec_pems04
elif args.datasets.index(dataset) == 1:
    g = vec_pems07
elif args.datasets.index(dataset) == 2:
    g = vec_pems08

train_dataloader, val_dataloader, test_dataloader, adj, max_speed, scaler = load_data2(args)
model = MDTGAT(input_dim=args.vec_dim,hidden_dim=args.hidden_dim, seq_len = args.seq_len, out_len=args.pre_len,
               encode_dim=args.enc_dim,datasets=args.datasets,
               dataset=args.dataset, ft_dataset=dataset_bak, adj_pems04=adj_pems04, adj_pems07=adj_pems07,
               adj_pems08=adj_pems08,batch_size=args.batch_size,device=device).to(device)
optimizer = optim.Adam([{'params': model.parameters()},
                        {'params': domain_classifier.parameters()}], lr=args.learning_rate)

model.load_state_dict(state['model'])
optimizer.load_state_dict(state['optim'])

if args.labelrate != 0:
    test_state = model_train(args, model, optimizer)
    model.load_state_dict(test_state['model'])
    optimizer.load_state_dict(test_state['optim'])

test_mae, test_rmse, test_mape = test_result_save()
print(f'mae: {test_mae: .3f}, rmse: {test_rmse: .3f}, mape: {test_mape * 100: .3f}\n\n')
with open(TXTPATH, 'a') as f:
    f.write(f'\nmae: {test_mae: .3f}, rmse: {test_rmse: .3f}, mape: {test_mape * 100: .3f}\n\n')
result_visualization(args)