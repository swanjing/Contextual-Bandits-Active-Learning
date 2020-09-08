""" Representative Sampling

"""
import numpy as np

from libact.base.interfaces import QueryStrategy, ContinuousModel, \
    ProbabilisticModel
from libact.utils import inherit_docstring_from, zip
from libact.base.dataset import Dataset, DatasetBiome

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

class RepresentativeSampling(QueryStrategy):

    """RepresentativeSampling Sampling

    This class implements Representative Sampling active learning algorithm [1]_.

    Parameters
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` or :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The base model used for training.

    method: {'lc', 'sm', 'entropy'}, optional (default='lc')
        least confidence (lc), it queries the instance whose posterior
        probability of being positive is nearest 0.5 (for binary
        classification);
        smallest margin (sm), it queries the instance whose posterior
        probability gap between the most and the second probable labels is
        minimal;
        entropy, requires :py:class:`libact.base.interfaces.ProbabilisticModel`
        to be passed in as model parameter;


    Attributes
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` or :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The model trained in last query.


    Examples
    --------
    Here is an example of declaring a UncertaintySampling query_strategy
    object:

    .. code-block:: python

       from libact.query_strategies import UncertaintySampling
       from libact.models import LogisticRegression

       qs = UncertaintySampling(
                dataset, # Dataset object
                model=LogisticRegression(C=0.1)
            )

    Note that the model given in the :code:`model` parameter must be a
    :py:class:`ContinuousModel` which supports predict_real method.


    References
    ----------

    .. [1] Xu, Zhao, Kai Yu, Volker Tresp, Xiaowei Xu, and Jizhi Wang. 
            "Representative Sampling for Text Classification Using Support Vector Machines." 
           European conference on information retrieval, pp. 393-407.
    """

    def __init__(self, dataset, args, **kwargs):
        super(RepresentativeSampling, self).__init__(dataset, args, **kwargs)

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )

        self.method = kwargs.pop('method', 'lc')

        self.T = kwargs.pop('T', None)
        if self.T is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'T'"
            )

        self.W = [1]
        self.queried_hist_ = []

        self.raw_rw = 0

    def calc_reward_fn(self):
        """Calculate the reward value"""
        reward = self.raw_rw
        reward /= len(self.dataset)
        reward /= self.T
        
        return reward
        

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        """Updating for IW-ACC reward function"""
        self.queried_hist_.append(entry_id)

        self.model.eval()
        with torch.no_grad():
            ex = torch.tensor(self.dataset.data[self.queried_hist_[-1]][0]).to(self.device)
            self.raw_rw += np.sum(
                self.W[-1] * (
                    self.model(ex)[0].max(1)[1].cpu().numpy() == self.dataset.data[self.queried_hist_[-1]][1]))

    def make_query(self, num=1, return_score=False):
        """Return the index of the sample to be queried and labeled and
        selection score of each sample. Read-only.

        No modification to the internal states.

        Returns
        -------
        ask_id : int
            The index of the next unlabeled sample to be queried and labeled.

        """
        self.nquery=num
        dataset = self.dataset
        # unlabeled_entry_ids, _ = dataset.get_unlabeled_entries()

        unlabeled_Xs, unlabeled_entry_ids = dataset.get_unlabeled_entries()
        all_ids = np.arange(len(unlabeled_Xs))
        unlabeled_dataset = self._handler(unlabeled_Xs, all_ids, transform=self._args['transform'])

        loader_te = DataLoader(unlabeled_dataset,
                            shuffle=False, **self.loader_te_args)
        self.model.eval()

        embeds = torch.tensor([])
        with torch.no_grad():
            for batch_idxs, (x,y) in enumerate(loader_te):
                x = torch.tensor(x).to(self.device)
                out, e1 = self.model(x)
                embeds = torch.cat((embeds, e1.cpu()))

        cluster_learner = KMeans(n_clusters=num)
        cluster_learner.fit(embeds)
        cluster_idxs = cluster_learner.predict(embeds)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeds - centers)**2
        dis = dis.sum(axis=1)
        q_idxs = np.array([np.arange(embeds.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(num)])

        ask_id = unlabeled_entry_ids[q_idxs]

        return ask_id
