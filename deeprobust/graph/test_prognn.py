'''
    If you would like to reproduce the performance of the paper,
    please refer to https://github.com/ChandlerBang/Pro-GNN
'''
import argparse
import os

import numpy as np
import torch
from gcn import GCN

from prognn import ProGNN
from GNN import GCNII
from Dataset import Dataset
from utils import preprocess
from node_embedding_attack import NodeEmbeddingAttack
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
import networkx as nx

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.02,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='meta',
        choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=10, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=1.0, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=1.0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.0001, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')
parser.add_argument('--n_cluster', type=int, default=8, help='number of cluster')
parser.add_argument('--z_dim', type=int, default=16)
parser.add_argument('--nlayers', type=int, default=7)
parser.add_argument('--ntest', type=int, default=10)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

os.environ['CUDA_VISIBLE_DEVICES']='0'

# print(args)

print(device)
ARI = []
ACC = []
NMI = []
FMI = []
NAME = []


for cishu in ['Quake_Smart-seq2_Diaphragm','Quake_10x_Bladder','Quake_Smart-seq2_Heart','Wang_Lung','Romanov','Quake_Smart-seq2_Limb_Muscle']:


    data = Dataset(root='/dataset/', name=name, setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train  = idx_train + idx_val


    args.n_cluster = labels.max().item() + 1
    # args.z_dim = labels.max().item() + 1
    args.z_dim = 16
    print("====================" + name +"=========================================")


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    model = GCNII(nfeat=features.shape[1],
                      nlayers=args.nlayers,
                      nhidden=args.hidden,
                      nclass=args.z_dim,
                      dropout=0,
                      lamda=0.3,
                      alpha=0.2,
                      variant=False).to(device)

    if args.attack == 'no':
        perturbed_adj = adj
    # add_node = 1000
    model_embeding = NodeEmbeddingAttack()
    if method_my == "add":
        model_embeding.attack(adj, attack_type="add", n_candidates=add_node)
    if method_my == "add_by_remove":
        model_embeding.attack(adj, attack_type="add_by_remove", n_candidates=add_node)
    if method_my == "remove":
        model_embeding.attack(adj, attack_type="remove")
    perturbed_adj = model_embeding.modified_adj
    perturbed_adj = adj



    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)



    prognn = ProGNN(model, args, device)
    prognn.fit(features, perturbed_adj, labels, idx_train)
    acc_test_final = 0.0
    loss_test_final = 0.0
    ari_test_final =0.0
    nmi_test_final = 0.0
    fmi_test_final =0.0

    print("======test====================")
    for i in range(args.ntest):
        acc_max = 0
        ari_max = 0
        nmi_max = 0
        fmi_max = 0
        acc_test ,loss_test,ari_test,nmi_test,fmi_test,output,y_pred = prognn.test(features, labels, idx_test)
        if acc_test >= acc_max:
            acc_max = acc_test
            ari_max = ari_test
            nmi_max = nmi_test
            fmi_max = fmi_test

        acc_test_final += acc_test
        loss_test_final += loss_test
        ari_test_final += ari_test
        nmi_test_final += nmi_test
        fmi_test_final += fmi_test

    acc = acc_test_final/args.ntest
    ari = ari_test_final / args.ntest
    nmi = nmi_test_final / args.ntest
    fmi = fmi_test_final / args.ntest

    print("=========================" + name + "============================================")
    print(acc)
    print(ari)
    print(nmi)
    print(fmi)

    NAME.append(name)
    ACC.append(acc)
    ARI.append(ari)
    NMI.append(nmi)
    FMI.append(fmi)

    print("\tTest set results:",

          "loss= {:.4f}".format(loss_test_final/args.ntest),
          "accuracy= {:.4f}".format(acc_test_final/args.ntest),
          "ari= {:.4f}".format(ari_test_final / args.ntest),
          "nmi= {:.4f}".format(nmi_test_final / args.ntest),
          "fmi= {:.4f}".format(fmi_test_final / args.ntest))

# #
dict = {'name': NAME, 'acc':ACC,'ari': ARI, 'nmi': NMI, 'fmi': FMI}
df = pd.DataFrame(dict)
df.to_csv('result_2.csv')

df = pd.read_csv('result_2.csv')

print(df)




