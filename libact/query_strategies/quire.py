""" Active Learning by QUerying Informative and Representative Examples (QUIRE)

This module contains a class that implements an active learning algorithm
(query strategy): QUIRE

"""

import bisect

import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel,\
    rbf_kernel

from libact.base.interfaces import QueryStrategy
from libact.base.dataset import Dataset, DatasetBiome

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class QUIRE(QueryStrategy):

    """Querying Informative and Representative Examples (QUIRE)

    Query the most informative and representative examples where the metrics
    measuring and combining are done using min-max approach.

    Parameters
    ----------
    lambda: float, optional (default=1.0)
        A regularization parameter used in the regularization learning
        framework.

    kernel : {'linear', 'poly', 'rbf', callable}, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', or a callable.
        If a callable is given it is used to pre-compute the kernel matrix
        from data matrices; that matrix should be an array of shape
        ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default=1.)
        Kernel coefficient for 'rbf', 'poly'.

    coef0 : float, optional (default=1.)
        Independent term in kernel function.
        It is only significant in 'poly'.


    Attributes
    ----------

    Examples
    --------
    Here is an example of declaring a QUIRE query_strategy object:

    .. code-block:: python

       from libact.query_strategies import QUIRE

       qs = QUIRE(
                dataset, # Dataset object
            )

    References
    ----------
    .. [1] S.-J. Huang, R. Jin, and Z.-H. Zhou. Active learning by querying
           informative and representative examples.
    """

    def __init__(self, dataset, args, **kwargs):
        super(QUIRE, self).__init__(dataset, args, **kwargs)
        self.Uindex = self.dataset.get_unlabeled_entries()[1]
        self.Lindex = np.where(self.dataset.get_labeled_mask())[0].tolist()
        # self.Uindex = [
        #     idx for idx, _ in self.dataset.get_unlabeled_entries()
        # ]
        # self.Lindex = [
        #     idx for idx in range(len(self.dataset)) if idx not in self.Uindex
        # ]
        
        # get embeddings
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )

        self.T = kwargs.pop('T', None)
        if self.T is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'T'"
            )

        dataset = self.dataset

        self.W = [1]
        self.queried_hist_ = []

        self.raw_rw = 0

        all_Xs, _ = dataset.get_entries()
        all_ids = np.arange(len(all_Xs))
        unlabeled_dataset = self._handler(all_Xs, all_ids, transform=self._args['transform'])

        loader_te = DataLoader(unlabeled_dataset,
                            shuffle=False, **self.loader_te_args)
        self.model.eval()

        embeds = torch.tensor([])
        with torch.no_grad():
            for batch_idxs, (x,y) in enumerate(loader_te):
                x = torch.tensor(x).to(self.device)
                out, e1 = self.model(x)
                embeds = torch.cat((embeds, e1.cpu()))

        self.lmbda = kwargs.pop('lambda', 1.)
        self.y = self.dataset.get_entries()[1]
        X = embeds
        self.kernel = kwargs.pop('kernel', 'rbf')
        if self.kernel == 'rbf':
            self.K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'poly':
            self.K = polynomial_kernel(X=X,
                                       Y=X,
                                       coef0=kwargs.pop('coef0', 1),
                                       degree=kwargs.pop('degree', 3),
                                       gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'linear':
            self.K = linear_kernel(X=X, Y=X)
        elif hasattr(self.kernel, '__call__'):
            self.K = self.kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError

        if not isinstance(self.K, np.ndarray):
            raise TypeError('K should be an ndarray')
        if self.K.shape != (len(X), len(X)):
            raise ValueError(
                'kernel should have size (%d, %d)' % (len(X), len(X)))
        self.L = np.linalg.inv(self.K + self.lmbda * np.eye(len(X)))

    def calc_reward_fn(self):
        """Calculate the reward value"""
        reward = self.raw_rw
        reward /= len(self.dataset)
        reward /= self.T

        return reward

    def update(self, entry_id, label):       
        self.Lindex = list(set(self.Lindex + list(entry_id)))
        self.Lindex.sort()
        # self.Uindex.remove(entry_id)
        self.Uindex = np.delete(self.Uindex, np.array([np.where(self.Uindex == id) for id in entry_id]))
        self.y[entry_id] = label

        self.queried_hist_.append(entry_id)

        self.model.eval()
        with torch.no_grad():
            ex = torch.tensor(self.dataset.data[self.queried_hist_[-1]][0]).to(self.device)
            self.raw_rw += np.sum(
                self.W[-1] * (
                    self.model(ex)[0].max(1)[1].cpu().numpy() == self.dataset.data[self.queried_hist_[-1]][1]))

    def make_query(self,num=1):
        L = self.L
        Lindex = self.Lindex
        Uindex = self.Uindex
        Uindex = Uindex.tolist()
        query_index = -1
        min_eva = np.inf
        y_labeled = np.array([label for label in self.y if label is not None])
        det_Laa = np.linalg.det(L[np.ix_(Uindex, Uindex)])
        # efficient computation of inv(Laa)
        M3 = np.dot(self.K[np.ix_(Uindex, Lindex)],
                    np.linalg.inv(self.lmbda * np.eye(len(Lindex))))
        M2 = np.dot(M3, self.K[np.ix_(Lindex, Uindex)])
        M1 = self.lmbda * np.eye(len(Uindex)) + self.K[np.ix_(Uindex, Uindex)]
        inv_Laa = M1 - M2
        iList = list(range(len(Uindex)))
        if len(iList) == 1:
            return Uindex[0]
        
        U = list()
        for i, each_index in enumerate(Uindex):
            # go through all unlabeled instances and compute their evaluation
            # values one by one
            Uindex_r = Uindex[:]
            Uindex_r.remove(each_index)
            iList_r = iList[:]
            iList_r.remove(i)
            inv_Luu = inv_Laa[np.ix_(iList_r, iList_r)] - 1 / inv_Laa[i, i] * \
                np.dot(inv_Laa[iList_r, i], inv_Laa[iList_r, i].T)
            tmp = np.dot(
                L[each_index][Lindex] -
                np.dot(
                    np.dot(
                        L[each_index][Uindex_r],
                        inv_Luu
                    ),
                    L[np.ix_(Uindex_r, Lindex)]
                ),
                y_labeled,
            )
            eva = L[each_index][each_index] - \
                det_Laa / L[each_index][each_index] + 2 * np.abs(tmp)
            
            U.append(eva)
            if eva < min_eva:
                query_index = each_index
                min_eva = eva
        
        U = torch.tensor(U)
        ask_id = self.Uindex[U.sort()[1][:num]]
        return ask_id
