import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans

from utils import accuracy
from pgd import PGD, prox_operators
import warnings
from torch.nn import Parameter
import scipy.sparse as sp
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics


class ProGNN:
    """ ProGNN (Properties Graph Neural Network). See more details in Graph Structure Learning for Robust Graph Neural Networks, KDD 2020, https://arxiv.org/abs/2005.10203.

    Parameters
    ----------
    model:
        model: The backbone GNN model in ProGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
    See details in https://github.com/ChandlerBang/Pro-GNN.

    """

    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)
        self.mu = Parameter(torch.Tensor(args.n_cluster, args.z_dim))
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.n_cluster = args.n_cluster

    def fit(self, features, adj, labels, idx_train, **kwargs):
        """Train Pro-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
        args = self.args

        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        estimator = EstimateAdj(adj, symmetric=args.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=args.lr_adj)

        self.optimizer_l1 = PGD(estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=args.lr_adj, alphas=[args.alpha])

        # warnings.warn("If you find the nuclear proximal operator runs too slow on Pubmed, you can  uncomment line 67-71 and use prox_nuclear_cuda to perform the proximal on gpu.")
        # if args.dataset == "pubmed":
        #     self.optimizer_nuclear = PGD(estimator.parameters(),
        #               proxs=[prox_operators.prox_nuclear_cuda],
        #               lr=args.lr_adj, alphas=[args.beta])
        # else:
        # warnings.warn("If you find the nuclear proximal operator runs too slow, you can modify line 77 to use prox_operators.prox_nuclear_cuda instead of prox_operators.prox_nuclear to perform the proximal on GPU. See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        self.optimizer_nuclear = PGD(estimator.parameters(),
                  proxs=[prox_operators.prox_nuclear],
                  lr=args.lr_adj, alphas=[args.beta])

        self.train_gcn_output(1, features, estimator.estimated_adj,
                       labels, idx_train)
        # Train model
        t_total = time.time()
        print('lr', self.optimizer.param_groups[0]['lr'])

        for epoch in range(args.epochs):
            # if epoch!=0 and epoch % 15 == 0:
            #     self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * 0.1
            #     self.optimizer_adj.param_groups[0]['lr'] = self.optimizer_adj.param_groups[0]['lr'] * 0.1
            #     self.optimizer_l1.param_groups[0]['lr'] = self.optimizer_l1.param_groups[0]['lr'] *0.1
            #     self.optimizer_nuclear.param_groups[0]['lr'] = self.optimizer_nuclear.param_groups[0]['lr'] *0.1
            #     print('lr', self.optimizer.param_groups[0]['lr'])
            if args.only_gcn:
                self.train_gcn(epoch, features, estimator.estimated_adj,
                        labels, idx_train)
            else:

                for i in range(int(args.outer_steps)):
                    self.train_adj(epoch, features, adj, labels,
                            idx_train)

                for i in range(int(args.inner_steps)):
                    self.train_gcn(epoch, features, estimator.estimated_adj,
                            labels, idx_train)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

    def train_gcn(self, epoch, features, adj, labels, idx_train):
        args = self.args
        estimator = self.estimator
        adj = estimator.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj).to(self.device)
        # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        q = self.soft_assign(output[idx_train])
        p = self.target_distribution(q).data
        self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        loss_train = self.cluster_loss(p, q)
        acc_train = self.cluster_acc(self.y_pred, labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        # self.model.eval()
        # output = self.model(features, adj)
        #
        # # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # # acc_val = accuracy(output[idx_val], labels[idx_val])
        #
        # q = self.soft_assign(output[idx_val])
        # p = self.target_distribution(q).data
        # self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        # loss_val = self.cluster_loss(p, q)
        # acc_val = self.cluster_acc(self.y_pred, labels[idx_val])

        if acc_train > self.best_val_acc:
            self.best_val_acc = acc_train
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_train < self.best_val_loss:
            self.best_val_loss = loss_train
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      # 'loss_val: {:.4f}'.format(loss_val.item()),
                      # 'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))

    def accuracy_1(self,output, labels):
        """Return accuracy of output compared to labels.

        Parameters
        ----------
        output : torch.Tensor
            output from model
        labels : torch.Tensor or numpy.array
            node labels

        Returns
        -------
        float
            accuracy
        """
        if not hasattr(labels, '__len__'):
            labels = [labels]
        if type(labels) is not torch.Tensor:
            labels = torch.LongTensor(labels)
        if type(output) is not torch.Tensor:
            output = torch.LongTensor(output)


        preds = output.type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def train_gcn_output(self, epoch, features, adj, labels, idx_train):
        args = self.args
        estimator = self.estimator
        adj = estimator.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj).to(self.device)
        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(args.n_cluster, n_init=20)
        self.y_pred = kmeans.fit_predict(output.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))



    def train_adj(self, epoch, features, adj, labels, idx_train):
        estimator = self.estimator
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        normalized_adj = estimator.normalize()

        if args.lambda_:
            loss_smooth_feat_1 = self.feature_smoothing(estimator.estimated_adj, features)
            loss_smooth_feat_2 = self.trace_loss(estimator.estimated_adj,  self.n_cluster) ** 2
        else:
            loss_smooth_feat = 0 * loss_l1

        output = self.model(features, normalized_adj)
        # loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        # acc_train = accuracy(output[idx_train], labels[idx_train])

        q = self.soft_assign(output[idx_train])
        p = self.target_distribution(q).data
        self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        loss_gcn = self.cluster_loss(p, q)
        acc_train = self.cluster_acc(self.y_pred, labels[idx_train])

        loss_symmetric = torch.norm(estimator.estimated_adj \
                        - estimator.estimated_adj.t(), p="fro")

        loss_diffiential =  loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat_1 + args.phi * loss_symmetric + args.lambda_ * loss_smooth_feat_2
        # loss_diffiential = loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat_1 + args.phi * loss_symmetric
        # loss_diffiential = loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat_2 + args.phi * loss_symmetric
        loss_diffiential.backward()

        self.optimizer_adj.step()
        loss_nuclear =  0 * loss_fro
        if args.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                    + args.gamma * loss_gcn \
                    + args.alpha * loss_l1 \
                    + args.beta * loss_nuclear \
                    + args.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        # self.model.eval()
        # normalized_adj = estimator.normalize()
        # output = self.model(features, normalized_adj)
        #
        # # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # # acc_val = accuracy(output[idx_val], labels[idx_val])
        #
        # q = self.soft_assign(output[idx_val])
        # p = self.target_distribution(q).data
        # self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        # loss_val = self.cluster_loss(p, q)
        # acc_val = self.cluster_acc(self.y_pred, labels[idx_val])

        print('Epoch: {:04d}'.format(epoch+1),
              'acc_train: {:.4f}'.format(acc_train.item()),
              # 'loss_val: {:.4f}'.format(loss_val.item()),
              # 'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


        if acc_train > self.best_val_acc:
            self.best_val_acc = acc_train
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_gcn < self.best_val_loss:
            self.best_val_loss = loss_gcn
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj-adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))


    def test(self, features, labels, idx_test):
        """Evaluate the performance of ProGNN on test set
        """

        self.model.eval()
        adj = self.best_graph
        if self.best_graph is None:
            adj = self.estimator.normalize()
        output = self.model(features, adj)
        # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        # acc_test = accuracy(output[idx_test], labels[idx_test])

        q = self.soft_assign(output[idx_test])
        p = self.target_distribution(q).data
        self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        loss_test = self.cluster_loss(p, q)
        acc_test = self.cluster_acc(self.y_pred, labels[idx_test])
        ari_test,nmi_test,fmi_test=self.cluster_ari_nmi_fmi(self.y_pred, labels[idx_test])

        q_view = self.soft_assign(output)
        self.y_pred_view = torch.argmax(q_view, dim=1).data.cpu().numpy()

        np.save("test_acc", acc_test.item())
        return acc_test.item(),loss_test.item(),ari_test,nmi_test,fmi_test,output,self.y_pred_view

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat

    def trace_loss(self,adj, k):
        adj = torch.clamp(adj, 0, 1)
        adj = torch.round(adj)
        rowsum = adj.sum(axis=1).detach().cpu().numpy()
        d = torch.zeros(adj.shape).numpy()
        row, col = np.diag_indices_from(d)
        d[row, col] = rowsum
        l = d - adj.detach().cpu().numpy()
        e_vals, e_vecs = np.linalg.eig(l)
        sorted_indices = np.argsort(e_vals)
        q = torch.tensor(e_vecs[:, sorted_indices[0:k:]].astype(np.float32)).to("cuda")
        m = torch.mm(torch.t(q), adj)
        m = torch.mm(m, q)
        return torch.trace(m)

    def soft_assign(self, z):
        self.mu = self.mu.to(self.device)
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)*100
        return (p.t() / p.sum(1)).t()

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return self.gamma*kldloss

    def cluster_acc(self,y_pred, y_true):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """

        if type(y_pred) is not torch.Tensor:
            y_pred = torch.LongTensor(y_pred)
        if type(y_true) is not torch.Tensor:
            y_true = torch.LongTensor(y_pred)

        y_pred = y_pred.type_as(y_true)

        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()


        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        # from sklearn.utils.linear_assignment_ import linear_assignment
        from scipy.optimize import linear_sum_assignment as linear_assignment
        ind = linear_assignment(w.max() - w)
        my_acc = sum([w[i, j] for i, j in enumerate(ind[1])]) * 1.0 / y_pred.size

        return my_acc

    def cluster_ari_nmi_fmi(self,y_pred, y_true):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """

        if type(y_pred) is not torch.Tensor:
            y_pred = torch.LongTensor(y_pred)
        if type(y_true) is not torch.Tensor:
            y_true = torch.LongTensor(y_pred)

        y_pred = y_pred.type_as(y_true)

        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()

        ari_test = metrics.adjusted_rand_score(y_true, y_pred)
        nmi_test = metrics.normalized_mutual_info_score(y_true, y_pred)
        fmi_test = metrics.cluster.fowlkes_mallows_score(y_true, y_pred)


        return ari_test,nmi_test,fmi_test

    def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num):
        f_adj = np.matmul(z, np.transpose(z))
        cosine = f_adj
        cosine = cosine.reshape([-1, ])
        pos_num = round(upper_threshold * len(cosine))
        neg_num = round((1 - lower_treshold) * len(cosine))

        pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
        neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

        return np.array(pos_inds), np.array(neg_inds)



class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())/2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx


