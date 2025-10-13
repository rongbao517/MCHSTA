import networkx as nx
import numpy as np
import torch
from multiprocessing import cpu_count
from .funcs import get_adjacency_matrix
import pandas as pd
from .config import DATA_PATHS
def generate_vector(args):
    adj_dir = DATA_PATHS[args.dataset]['adj']
    if adj_dir[-3:] == 'npy':
        adj = np.load(adj_dir, allow_pickle=True).astype(np.int64)
    elif adj_dir[-3:]=='csv':
        adj = get_adjacency_matrix(distance_df_filename=adj_dir, num_of_vertices=num_of_vertices)
    graph = nx.DiGraph(adj)
    spl = dict(nx.all_pairs_shortest_path_length(graph))
    from .node2vec import Node2Vec as node2vec
    n2v = node2vec(G=graph, distance=spl, emb_size=args.vec_dim, length_walk=args.walk_length,
                   num_walks=args.num_walks, window_size=5, batch=4, p=args.p, q=args.q,
                   workers=(int(cpu_count() / 2)))
    n2v = n2v.train()

    gfeat = []
    for i in range(len(adj)):
        nodeivec = n2v.wv.get_vector(str(i))
        gfeat.append(nodeivec)
    g = torch.tensor(np.array(gfeat))

    return g, gfeat

