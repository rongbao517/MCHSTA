import torch
import torch.nn as nn
import torch.nn.functional as F
# from tgcn import TGCN
# from torch_geometric.nn import GATConv,SAGEConv,GINConv
# from torch_geometric.nn.dense import dense_gat_conv
# from torch_geometric.utils import dense_to_sparse
# from torch_sparse import coalesce
# from model import Extractor_N2V
# from torch_geometric.nn.pool import SAGPooling
# from torch.nn import GRU, LSTM
import math
# from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# from torch_geometric.nn.pool.select.topk import topk
# from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from block import TransformerLayer


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class Grad(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.constant
        return grad_output, None

    def grad(x, constant):
        return Grad.apply(x, constant)


class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_dim, class_num,device):
        super(DomainDiscriminator, self).__init__()

        self.adversarial_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, class_num)).to(device)

    def forward(self, embed, alpha, if_reverse):
        if if_reverse:
            embed = GradReverse.grad_reverse(embed, alpha)
        else:
            embed = Grad.grad(embed, alpha)

        out = self.adversarial_mlp(embed)

        return F.log_softmax(out, dim=-1)

class Domain_classifier_DG(nn.Module):

    def __init__(self, num_class, encode_dim):
        super(Domain_classifier_DG, self).__init__()

        self.num_class = num_class
        self.encode_dim = encode_dim

        self.fc1 = nn.Linear(self.encode_dim, 16)
        self.fc2 = nn.Linear(16, num_class)

    def forward(self, input, constant, Reverse):
        if Reverse:
            input = GradReverse.grad_reverse(input, constant)
        else:
            input = Grad.grad(input, constant)
        logits = torch.tanh(self.fc1(input))
        logits = self.fc2(logits)
        logits = F.log_softmax(logits, 1)

        return logits


def dense_to_sparse(adj_matrix):
    N = adj_matrix.size(0)
    row, col = torch.nonzero(adj_matrix, as_tuple=False).t()

    return [row,col]

class space_enconder(nn.Module):
    def __init__(self,in_dim,Hidden_dim,out_dim,N_node,head_num=8,dropout=0.5,pool_ratio=0.5):
        super(space_enconder,self).__init__()

        self.dropout = dropout
        self.pooling_ratio = pool_ratio
        self.GCN = GCN(in_dim,Hidden_dim,in_dim,dropout)
        self.BN = nn.BatchNorm1d(Hidden_dim)
        self.Relu = nn.ReLU()
        self.trans = TransformerLayer(Hidden_dim, head_num)

    def forward(self,x,adj,data=None):
        adj1 = adj.to(torch.float)
        x = self.GCN(adj1, x)
        x = self.BN(self.Relu(x))
        out = self.trans(x.unsqueeze(0)).squeeze(0)
        return F.log_softmax(out,dim=1)

class GCNLayer(nn.Module):
    def __init__(self,input_features,output_features,bias=False):
        super(GCNLayer,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features,output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std,std)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)

    def forward(self,adj,x):
        support = torch.mm(x,self.weights)
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output+self.bias
        return output

class GCN(nn.Module):
    def __init__(self,input_size,hidden_size,num_class,dropout,bias=False):
        super(GCN,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_class = num_class
        self.gcn1 = GCNLayer(input_size,hidden_size,bias=bias)
        self.gcn2 = GCNLayer(hidden_size,num_class,bias=bias)
        self.dropout = dropout
    def forward(self,adj,x):
        x = F.relu(self.gcn1(adj,x))
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gcn2(adj,x)
        return F.log_softmax(x,dim=1)


class time_encoder(nn.Module):
    def __init__(self, sen_len,en_dim,device,head_num=8):
        super(time_encoder,self).__init__()
        self.device = device
        self.W = nn.Parameter(torch.ones([sen_len,sen_len]).to(self.device))
        self.l = nn.Linear(sen_len,sen_len)
        self.out_layer = TransformerLayer(sen_len, head_num)


    def forward(self,t,data):
        in_ = torch.einsum("bt,tn->bn", t, self.W)
        t_feature = self.l(in_)
        t_feature = t_feature.unsqueeze(2).expand_as(data)
        out = data + t_feature
        out = self.out_layer(out.permute(0,2,1)).permute(0,2,1)
        return out

class fuse_enconder(nn.Module):
    def __init__(self,en_dim,seq_len,head_num=8):
        super(fuse_enconder,self).__init__()
        self.en_dim = en_dim
        self.seq_len = seq_len
        self.in_en = nn.Linear(seq_len, en_dim)
        self.att = TransformerLayer(seq_len, head_num)
        self.out_en = nn.Linear(en_dim, seq_len)

    def forward(self, s_e,t_e):
        B, t_len,N_num = t_e.shape
        t_e1 = t_e
        t_e = self.in_en(t_e.permute(0,2,1))
        attention = (s_e.permute(1,0))@s_e
        attention = F.softmax(attention/(s_e.shape[1]**0.5),dim=-1)
        out = torch.einsum("pq,bnp->bnq", attention, t_e)
        out = self.out_en(out)
        out = self.att(out).permute(0,2,1)

        return out + t_e1

class decoder(nn.Module):
    def __init__(self,se_len,out_len):
        super(decoder,self).__init__()
        self.out_layer = nn.Linear(se_len,out_len)

    def forward(self,indata):
        out = self.out_layer(indata.permute(0,2,1))
        return out.permute(0,2,1)


class Gatefuse(nn.Module):
    def __init__(self, enc_dim):
        super().__init__()
        self.a = nn.Linear(enc_dim,enc_dim)
        self.b = nn.Linear(enc_dim,enc_dim)
        self.z = nn.Linear(enc_dim,enc_dim)
        self.bn = nn.BatchNorm1d(enc_dim)
        self.relu = nn.PReLU()
        self.out = nn.BatchNorm1d(enc_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self,f1,f2):
        w = self.z(f1+f2)
        w = self.dropout(w)
        x = w*(self.a(f1))+(1-w)*(self.b(f2))
        out = self.bn(self.relu(x))

        return self.out(out)

class CrossAttention(nn.Module):
    def __init__(self, info_dim1,info_dim2,out_dim, context_dim=None, heads=8, dim_head=64, dropout=0.3,ATTENTION_MODE = "math"):
        super().__init__()
        inner_dim = info_dim1 * heads

        self.ATTENTION_MODE = ATTENTION_MODE
        self.scale = dim_head**-0.5
        self.heads = heads
        self.attn_drop = nn.Dropout(dropout)

        self.to_q = nn.Linear(info_dim1, inner_dim, bias=False)
        self.to_k = nn.Linear(info_dim2, inner_dim, bias=False)
        self.to_v = nn.Linear(info_dim2, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim), nn.Dropout(dropout)
        )

    def forward(self, x, text, mask=None):
        # B, N, T
        B, N, F = x.shape
        q = self.to_q(x)
        # text = default(text, x)
        k = self.to_k(text)
        v = self.to_v(text)

        q, k, v = map(
            lambda t: t.reshape(B,self.heads,N, F), (q, k, v)
        )  # B H L D
        if self.ATTENTION_MODE == "flash":
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = x.reshape(B,N, self.heads*F)

        elif self.ATTENTION_MODE == "math":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, self.heads*F)
        else:
            raise NotImplemented
        return self.to_out(x).squeeze(0)


class MDTGAT(nn.Module):
    def __init__(self,input_dim,hidden_dim,seq_len,out_len,encode_dim,datasets,dataset, ft_dataset,adj_pems04, adj_pems07, adj_pems08,batch_size,device,head_num=8):
        super(MDTGAT,self).__init__()
        self.dataset = dataset
        self.datasets = datasets
        self.finetune_dataset = ft_dataset
        self.pems04_adj = adj_pems04
        self.pems07_adj = adj_pems07
        self.pems08_adj = adj_pems08
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_len
        self.encode_dim = encode_dim
        self.device = device

        self.pems04_featExtractor = space_enconder(input_dim, hidden_dim, encode_dim,adj_pems04.shape[0]).to(device)
        self.pems07_featExtractor = space_enconder(input_dim, hidden_dim, encode_dim,adj_pems07.shape[0]).to(device)
        self.pems08_featExtractor = space_enconder(input_dim, hidden_dim, encode_dim,adj_pems08.shape[0]).to(device)
        self.shared_pems04_featExtractor = space_enconder(input_dim, hidden_dim, encode_dim,adj_pems04.shape[0]).to(device)
        self.shared_pems07_featExtractor = space_enconder(input_dim, hidden_dim, encode_dim,adj_pems07.shape[0]).to(device)
        self.shared_pems08_featExtractor = space_enconder(input_dim, hidden_dim, encode_dim,adj_pems08.shape[0]).to(device)

        self.pems04_trans = TransformerLayer(seq_len, head_num)
        self.pems07_trans = TransformerLayer(seq_len, head_num)
        self.pems08_trans = TransformerLayer(seq_len, head_num)

        self.time_encoder = time_encoder(seq_len,encode_dim,device).to(device)
        self.fuse_encoder = fuse_enconder(encode_dim, seq_len)
        self.fuseattention = CrossAttention(encode_dim,encode_dim,encode_dim)
        self.deconder = decoder(seq_len,out_len)

    def forward(self, vec_pems04, vec_pems07,vec_pems08, feat, t, eval=False):
        t = t.squeeze()

        if self.dataset != self.finetune_dataset:
            if not eval:
                shared_pems04_feat = self.shared_pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)+vec_pems04
                shared_pems07_feat = self.shared_pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)+vec_pems07
                shared_pems08_feat = self.shared_pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)+vec_pems08
            else:
                if self.dataset == self.datasets[0]:
                    shared_pems04_feat = self.shared_pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)+vec_pems04
                elif self.dataset == self.datasets[1]:
                    shared_pems07_feat = self.shared_pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)+vec_pems07
                elif self.dataset == self.datasets[2]:
                    shared_pems08_feat = self.shared_pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)+vec_pems08

            if self.dataset == self.datasets[0]:
                feat = self.pems04_trans(feat.permute(0,2,1)).permute(0,2,1)
                pred = self.time_encoder(t, feat)
                pred = self.fuse_encoder(shared_pems04_feat,pred)
                pred = self.deconder(pred)
            elif self.dataset == self.datasets[1]:
                feat = self.pems07_trans(feat.permute(0, 2, 1)).permute(0, 2, 1)
                pred = self.time_encoder(t, feat)
                pred = self.fuse_encoder(shared_pems07_feat, pred)
                pred = self.deconder(pred)
            elif self.dataset == self.datasets[2]:
                feat = self.pems08_trans(feat.permute(0, 2, 1)).permute(0, 2, 1)
                pred = self.time_encoder(t, feat)
                pred = self.fuse_encoder(shared_pems08_feat, pred)
                pred = self.deconder(pred)

            if not eval:
                return pred, shared_pems04_feat, shared_pems07_feat, shared_pems08_feat
            else:
                return pred
        else:
            if self.dataset == self.datasets[0]:
                shared_pems04_feat = self.shared_pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)+vec_pems04
                pems04_feat = self.pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)+vec_pems04
                feat = self.pems04_trans(feat.permute(0, 2, 1)).permute(0, 2, 1)
                pred = self.time_encoder(t, feat)
                pred = self.fuse_encoder(pems04_feat+shared_pems04_feat, pred)
                pred = self.deconder(pred)
            elif self.dataset == self.datasets[1]:
                shared_pems07_feat = self.shared_pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)+vec_pems07
                pems07_feat = self.pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)+vec_pems07
                feat = self.pems07_trans(feat.permute(0, 2, 1)).permute(0, 2, 1)
                pred = self.time_encoder(t, feat)
                pred = self.fuse_encoder(pems07_feat+shared_pems07_feat, pred)
                pred = self.deconder(pred)
            elif self.dataset == self.datasets[2]:
                shared_pems08_feat = self.shared_pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)+vec_pems08
                pems08_feat = self.pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)+vec_pems08
                feat = self.pems08_trans(feat.permute(0, 2, 1)).permute(0, 2, 1)
                pred = self.time_encoder(t, feat)
                pred = self.fuse_encoder(pems08_feat+shared_pems08_feat, pred)
                pred = self.deconder(pred)

            return pred

