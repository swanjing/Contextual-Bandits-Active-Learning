""" Uncertainty Sampling

This module contains a class that implements two of the most well-known
uncertainty sampling query strategies: the least confidence method and the
smallest margin method (margin sampling).

"""
import numpy as np

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, zip
from libact.base.dataset import Dataset, DatasetBiome

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class UncertaintySampling(QueryStrategy):

    """Uncertainty Sampling

    This class implements Uncertainty Sampling active learning algorithm [1]_.

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

    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.
    """

    def __init__(self, dataset, args, **kwargs):
        super(UncertaintySampling, self).__init__(dataset, args, **kwargs)

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
        
    def _get_scores(self):
        dataset = self.dataset
        
        unlabeled_entry_Xs, unlabeled_entry_ids = dataset.get_unlabeled_entries()
        unlabeled_dataset = self._handler(unlabeled_entry_Xs, unlabeled_entry_ids, transform=self._args['transform'])

        loader_te = DataLoader(unlabeled_dataset,
                            shuffle=False, **self.loader_te_args)

        self.model.eval()

        probs = torch.tensor([])
        with torch.no_grad():
            for batch_idxs, (x,y) in enumerate(loader_te):
                x = x.to(self.device)
                out, e1 = self.model(x)
                prob = F.softmax(out, dim=1)
                probs = torch.cat((probs, prob.cpu()))

        if self.method == 'lc':  # least confident
            # score = -np.max(dvalue, axis=1)
            U = probs.max(1)[0]

        elif self.method == 'sm':  # smallest margin
            # if np.shape(dvalue)[1] > 2:
            #     # Find 2 largest decision values
            #     dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])
            # score = -np.abs(dvalue[:, 0] - dvalue[:, 1])
            probs_sorted, idxs = probs.sort(descending=True)
            U = probs_sorted[:,0] - probs_sorted[:,1]

        elif self.method == 'entropy':
            # score = np.sum(-dvalue * np.log(dvalue), axis=1)
            log_probs = torch.log(probs)
            U = (probs*log_probs).sum(1)

        return unlabeled_entry_ids, U

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
        dataset = self.dataset
        # unlabeled_entry_ids, _ = dataset.get_unlabeled_entries()

        unlabeled_entry_ids, U = self._get_scores()
        ask_id = unlabeled_entry_ids[U.sort()[1][:num]]
        # print(ask_id)

        if return_score:
            return ask_id, \
                   list(zip(unlabeled_entry_ids, U))
        else:
            return ask_id
