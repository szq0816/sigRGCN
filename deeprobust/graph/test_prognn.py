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
#
# # Here the random seed is to split the train/val/test data,
# # we need to set the random seed to be the same as that when you generate the perturbed graph
# # data = Dataset(root='/data/', name=args.dataset, setting='nettack', seed=15)
# # Or we can just use setting='prognn' to get the spli
#
# print(device)
# ARI = []
# ACC = []
# NMI = []
# FMI = []
# NAME = []
#
#
# # for name in ['Adam']:
# #     fi_name = name + '-accuracy.npy'
# #     x = np.load(fi_name)
# #     fi_name = name + '-ari.npy'
# #     y = np.load(fi_name)
# #     fi_name = name + '-nmi.npy'
# #     z = np.load(fi_name)
# #     fi_name = name + '-fmi.npy'
# #     c = np.load(fi_name)
# #
# #     print("=============="+name + "===========================")
# #     print("accuracy")
# #     print(x)
# #     print("ari")
# #     print(y)
# #     print("nmi")
# #     print(z)
# #     print("fmi")
# #     print(c)
# #     print("======================================================")
#
# # for name in ['Quake_Smart-seq2_Diaphragm','Quake_10x_Bladder','Klein','Quake_10x_Limb_Muscle','Plasschaert','Quake_Smart-seq2_Heart','Wang_Lung',
# #                  'Muraro','Adam','Bach','Chen','Pollen','Quake_10x_Spleen','Quake_10x_Trachea','Quake_Smart-seq2_Lung','Quake_Smart-seq2_Trachea',
# #                  'Romanov','Tosches_turtle','Young','Quake_Smart-seq2_Limb_Muscle']:
#
for cishu in ['4']:
#     name = 'Wang_Lung'
#     method_my = "remove"
#     # name = 'Quake_10x_Bladder'
#     add_node = 1000
# # for name in ['Muraro']:
#     if name == "Quake_10x_Bladder":
#         args.lr = 0.1
#         args.lr_adj = 0.01
#     if name == "Quake_Smart-seq2_Diaphragm":
#         args.lr = 0.01
#         args.lr_adj = 0.001
#     if name == "Wang_Lung":
#         args.lr = 0.01
#         args.lr_adj = 0.0001
#     if name == "Quake_10x_Limb_Muscle":
#         args.lr = 0.01
#         args.lr_adj = 0.0001
#     if name == "Romanov":
#         args.lr = 0.1
#         args.lr_adj = 0.001
# #
#     data = Dataset(root='/dataset/', name=name, setting='prognn')
#     print("okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
#     adj, features, labels = data.adj, data.features, data.labels
#     idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
#     idx_train  = idx_train + idx_val
# #
# #     # adj_my = adj.A
# #     # idx_my = []
# #
# #     # x,y =  np.where(adj_my !=0 )
# #     # node_idx = np.random.rand(0, len(x), 30)
# #
# #     # for i in range(0,1500,10):
# #     #     turtle = (x[i], y[i])
# #     #     idx_my.append(turtle)
# #
# #     # G = nx.Graph()
# #     # G.add_edges_from(idx_my)
# #     # nx.draw(G, with_labels=False, edge_color='r', node_color='b', node_size=10)
# #     # # nx.draw(G, with_labels=True)
# #     # plt.show()
# #
# #     # idx_train = idx_train + idx_val + idx_test
# #     # idx_test = idx_train + idx_val + idx_test
# #
# #     # adata_view = data.adata
# #     # outdir =  os.getcwd()+'/out_original_bladder'
# #     # embed = 'UMAP'
# #     # adata_view.obsm['latent'] = data.features
# #     # adata_view.obs['celltype'] = data.labels
# #     # sc.pp.neighbors(adata_view, n_neighbors=30, use_rep='latent')
# #     # if not os.path.exists(outdir):
# #     #     os.makedirs(outdir)
# #     # sc.set_figure_params(dpi=80, figsize=(6, 6), fontsize=10)
# #     # if outdir:
# #     #     sc.settings.figdir = outdir
# #     #     save = '.svg'
# #     # else:
# #     #    save = None
# #     # if embed == 'UMAP':
# #     #    sc.tl.umap(adata_view, min_dist=0.1)
# #     #    color = [c for c in ['celltype', 'kmeans', 'leiden', 'cell_type'] if c in adata_view.obs]
# #     #    sc.pl.umap(adata_view, color=color, save=save, show=False, wspace=0.4, ncols=4)
# #     # elif embed == 'tSNE':
# #     #    sc.tl.tsne(adata_view, use_rep='latent')
# #     #    color = [c for c in ['celltype', 'kmeans', 'leiden', 'cell_type'] if c in adata_view.obs]
# #     #    sc.pl.tsne(adata_view, color=color, save=save, show=False, wspace=0.4, ncols=4)
#
    # idx_train = np.array(idx_train)
    # idx_test = np.array(idx_test)
    #
    # args.n_cluster = labels.max().item() + 1
    # # args.z_dim = labels.max().item() + 1
    # args.z_dim = 16
    #
    #
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("====================" + name +"=========================================")


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model = GCN(nfeat=features.shape[1],
    #             nhid=args.hidden,
    #             nclass= args.z_dim,
    #             dropout=args.dropout, device=device)

    # model = GCN(nfeat=features.shape[1],
    #             nhid=args.hidden,
    #             nlayers=args.nlayers,
    #             nclass=args.z_dim,
    #             dropout=args.dropout,
    #             lamda=0.3,
    #             alpha=0.2,
    #             variant=False,
    #             device=device)
#     model = GCNII(nfeat=features.shape[1],
#                       nlayers=args.nlayers,
#                       nhidden=args.hidden,
#                       nclass=args.z_dim,
#                       dropout=0,
#                       lamda=0.3,
#                       alpha=0.2,
#                       variant=False).to(device)
#
#     if args.attack == 'no':
#         perturbed_adj = adj
#     # add_node = 1000
#     model_embeding = NodeEmbeddingAttack()
#     if method_my == "add":
#         model_embeding.attack(adj, attack_type="add", n_candidates=add_node)
#     if method_my == "add_by_remove":
#         model_embeding.attack(adj, attack_type="add_by_remove", n_candidates=add_node)
#     if method_my == "remove":
#         model_embeding.attack(adj, attack_type="remove")
#     perturbed_adj = model_embeding.modified_adj
#     # perturbed_adj = adj
#
#
#
#     perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)
#
#
#
#     prognn = ProGNN(model, args, device)
#     prognn.fit(features, perturbed_adj, labels, idx_train)
#     acc_test_final = 0.0
#     loss_test_final = 0.0
#     ari_test_final =0.0
#     nmi_test_final = 0.0
#     fmi_test_final =0.0
#
#     print("======test====================")
#     for i in range(args.ntest):
#         acc_max = 0
#         ari_max = 0
#         nmi_max = 0
#         fmi_max = 0
#         acc_test ,loss_test,ari_test,nmi_test,fmi_test,output,y_pred = prognn.test(features, labels, idx_test)
#         if acc_test >= acc_max:
#             acc_max = acc_test
#             ari_max = ari_test
#             nmi_max = nmi_test
#             fmi_max = fmi_test
#
#         acc_test_final += acc_test
#         loss_test_final += loss_test
#         ari_test_final += ari_test
#         nmi_test_final += nmi_test
#         fmi_test_final += fmi_test
# #
# #     # adata_view = data.adata
# #     # outdir = os.getcwd() + '/out_final_baldder'
# #     # embed = 'UMAP'
# #     # adata_view.obsm['latent'] = output.detach().cpu().numpy()
# #     # # adata_view.obs['celltype'] = labels.cpu().numpy()
# #     # adata_view.obs['celltype'] = y_pred
# #     # sc.pp.neighbors(adata_view, n_neighbors=30, use_rep='latent')
# #     # if not os.path.exists(outdir):
# #     #     os.makedirs(outdir)
# #     #     sc.set_figure_params(dpi=80, figsize=(6, 6), fontsize=10)
# #     # if outdir:
# #     #     sc.settings.figdir = outdir
# #     #     save = '.svg'
# #     # else:
# #     #     save = None
# #     # if embed == 'UMAP':
# #     #     sc.tl.umap(adata_view, min_dist=0.1)
# #     #     color = [c for c in ['celltype', 'kmeans', 'leiden', 'cell_type'] if c in adata_view.obs]
# #     #     sc.pl.umap(adata_view, color=color, save=save, show=False, wspace=0.4, ncols=4)
# #     # elif embed == 'tSNE':
# #     #     sc.tl.tsne(adata_view, use_rep='latent')
# #     #     color = [c for c in ['celltype', 'kmeans', 'leiden', 'cell_type'] if c in adata_view.obs]
# #     #     sc.pl.tsne(adata_view, color=color, save=save, show=False, wspace=0.4, ncols=4)

    import scipy.io as scio
    path = os.getcwd() + '/data_cluster_Wang_Lung.mat'
    matdata = scio.loadmat(path)
    X = matdata['V'].T
    labels = matdata['project_labs']
    # X = output.detach().cpu()
    # X  = features
    # labels = labels
    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(X)
    # # x_min, x_max = Y.min(0), Y.max(0)
    # # X_norm = (Y - x_min) / (x_max - x_min)

    area=(20*np.random.rand(len(Y)))**2
    color = []
    name = 'Wang_Lung'
    if name == 'Quake_10x_Bladder':
       for i in np.arange(len(labels)):
           if labels[i] == 1 :
               color.append('darkgreen')
           if labels[i] == 2:
               color.append('darkorchid')
           if labels[i] == 3 :
               color.append('darkgoldenrod')
           if labels[i] == 4 :
               color.append('darkred')

    if name == 'Quake_Smart-seq2_Diaphragm':
        for i in np.arange(len(labels)):
            if labels[i] == 1 :
                color.append('darkgreen')
            if labels[i] == 2:
                color.append('darkorchid')
            if labels[i] == 3 :
                color.append('darkgoldenrod')
            if labels[i] == 4 :
                color.append('darkred')
            if labels[i] == 5 :
                color.append('royalblue')
    if name == 'Romanov':
            for i in np.arange(len(labels)):
                if labels[i] == 0:
                    color.append('red')
                if labels[i] == 1:
                    color.append('darkgreen')
                if labels[i] == 2:
                    color.append('darkorchid')
                if labels[i] == 3:
                    color.append('darkgoldenrod')
                if labels[i] == 4:
                    color.append('darkred')
                if labels[i] == 5:
                    color.append('royalblue')
                if labels[i] == 6:
                    color.append('yellow')
                if labels[i] == 7:
                    color.append('SpringGreen')
    if name == 'Wang_Lung':
        for i in np.arange(len(labels)):
            if labels[i] == 1 :
                color.append('darkgreen')
            if labels[i] == 2:
                color.append('darkorchid')
            if labels[i] == 3 :
                color.append('darkgoldenrod')
            if labels[i] == 4 :
                color.append('darkred')
            if labels[i] == 0 :
                color.append('royalblue')

        #cornflowerblue

    # if add_node==1000:
    #     ad_ = "1000"
    # if add_node==1500:
    #     ad_ = "1500"
    # if add_node==2000:
    #     ad_ = "2000"
    # if add_node==2500:
    #     ad_ = "2500"
    # if add_node == 10000:
    #     ad_ = "10000"
    # if add_node == 100000:
    #     ad_ = "100000"
    # if add_node == 50000:
    #     ad_ = "50000"
    # if add_node==5000:
    #     ad_ = "5000"
    # if add_node==20000:
    #     ad_ = "20000"

    plt.scatter(Y[:, 0], Y[:, 1], c=color, s=4, alpha=None,marker='o',edgecolors=None)
    filename = name + '_rssNMF' +'.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')


    plt.show()



#     acc = acc_test_final/args.ntest
#     ari = ari_test_final / args.ntest
#     nmi = nmi_test_final / args.ntest
#     fmi = fmi_test_final / args.ntest
#     # if alpha == 0.0001:
#     #     name_ = '0.0001'
#     # if alpha == 0.0002:
#     #     name_ = '0.0002'
#     # if alpha == 0.0003:
#     #     name_ = '0.0003'
#     # if alpha == 0.0004:
#     #     name_ = '0.0004'
#     # if alpha == 0.0005:
#     #     name_ = '0.0005'
#     # if alpha == 0.0007:
#     #     name_ = '0.0007'
#     print("=========================" + name + "============================================")
#     print(acc)
#     print(ari)
#     print(nmi)
#     print(fmi)
#
# #
# #     # acc = acc_max
# #     # ari = ari_max
# #     # nmi = nmi_max
# #     # fmi = fmi_max
# #
#     NAME.append(name)
#     ACC.append(acc)
#     ARI.append(ari)
#     NMI.append(nmi)
#     FMI.append(fmi)
# #
# #     print("\tTest set results:",
# #
# #           "loss= {:.4f}".format(loss_test_final/args.ntest),
# #           "accuracy= {:.4f}".format(acc_test_final/args.ntest),
# #           "ari= {:.4f}".format(ari_test_final / args.ntest),
# #           "nmi= {:.4f}".format(nmi_test_final / args.ntest),
# #           "fmi= {:.4f}".format(fmi_test_final / args.ntest))
# # #     fi_name = name + '-accuracy'
# # #     np.save(fi_name,acc_test_final/args.ntest)
# # #     fi_name = name + '-ari'
# # #     np.save(fi_name, ari_test_final / args.ntest)
# # #     fi_name = name + '-nmi'
# # #     np.save(fi_name, nmi_test_final / args.ntest)
# # #     fi_name = name + '-fmi'
# # #     np.save(fi_name, fmi_test_final / args.ntest)
# # #
# dict = {'name': NAME, 'acc':ACC,'ari': ARI, 'nmi': NMI, 'fmi': FMI}
# df = pd.DataFrame(dict)
# df.to_csv('result_2.csv')
#
# df = pd.read_csv('result_2.csv')
#
# print(df)
#
# #
#
#
