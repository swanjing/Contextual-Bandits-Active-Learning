"""Uniform Bandits Active Learning (UBAL)

"""
from __future__ import division

import copy

import numpy as np

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state, zip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class UniformBanditsActiveLearning(QueryStrategy):

    """Uniform Bandits Active Learning (UBAL) query strategy.

    UBAL is an ensemble-based active learning algorithm that randomly choose among existing
    query strategies to decide which data to make query.

    Parameters
    ----------
    T : integer
        Query budget, the maximal number of queries to be made.

    query_strategies : list of :py:mod:`libact.query_strategies`\
    object instance
        The active learning algorithms used in ALBL, which will be both the
        the arms in the multi-armed bandit algorithm Exp4.P.
        Note that these query_strategies should share the same dataset
        instance with ActiveLearningByLearning instance.

    model : :py:mod:`libact.models` object instance
        The learning model used for the task.

    reward_fn : reward function, optional (default = IW-ACC)

    Attributes
    ----------
    query_strategies\_ : list of :py:mod:`libact.query_strategies` object instance
        The active learning algorithm instances.

    queried_hist\_ : list of integer
        A list of entry_id of the dataset which is queried in the past.

    Examples
    --------
    Here is an example of how to declare a ActiveLearningByLearning
    query_strategy object:

    .. code-block:: python

       from libact.query_strategies import ActiveLearningByLearning
       from libact.query_strategies import HintSVM
       from libact.query_strategies import UncertaintySampling
       from libact.models import LogisticRegression

        qs = UniformBanditsActiveLearning(trn_ds, args,
                    query_strategies=[
                        RandomSampling(trn_ds, args, T=quota, model=model_ubal),
                        UncertaintySampling(trn_ds, args, T=quota, model=model_ubal, method='lc'),
                        UncertaintySampling(trn_ds, args, T=quota, model=model_ubal, method='sm'),
                        UncertaintySampling(trn_ds, args, T=quota, model=model_ubal, method='entropy'),
                        RepresentativeSampling(trn_ds, args, T=quota, model=model_ubal)
                    ],
                    T=quota,
                    uniform_sampler=False,
                    model=model_ubal,
                    reward_fn = custom_rw
                )

    """

    def __init__(self, dataset, args, **kwargs):
        super(UniformBanditsActiveLearning, self).__init__(dataset, args, **kwargs)
        self.query_strategies_ = kwargs.pop('query_strategies', None)
        if self.query_strategies_ is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: "
                "'query_strategies'"
            )
        elif not self.query_strategies_:
            raise ValueError("query_strategies list is empty")

        # check if query_strategies share the same dataset with albl
        for qs in self.query_strategies_:
            if qs.dataset != self.dataset:
                raise ValueError("query_strategies should share the same"
                                 "dataset instance with albl")

        self.T = kwargs.pop('T', None)
        if self.T is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'T'"
            )

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )

        self.reward_fn = kwargs.pop('reward_fn', None)

        self.rw_hist_ = np.array([])

        self.W = [1]
        self.queried_hist_ = []

        self.raw_rw = 0

    def calc_reward_fn(self):
        """Calculate the reward value"""
        if self.reward_fn == None:
            reward = self.raw_rw
            reward /= len(self.dataset)
            reward /= self.T
        else:
            reward = self.rw

        return reward

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        """Updating for IW-ACC reward function"""
        self.queried_hist_.append(entry_id)

        model = copy.copy(self.model)
        model.eval()
        with torch.no_grad():
            ex = torch.tensor(self.dataset.data[self.queried_hist_[-1]][0]).to(self.device)
            self.raw_rw += np.sum(
                self.W[-1] * (
                    model(ex)[0].max(1)[1].cpu().numpy() == self.dataset.data[self.queried_hist_[-1]][1]))

    @inherit_docstring_from(QueryStrategy)
    def make_query(self,num=1):
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
        self.dataset.get_unlabeled_entries()
        try:
            _, unlabeled_entry_ids = dataset.get_unlabeled_entries()
        except ValueError:
            # might be no more unlabeled data left
            return

        self.strat_idx = np.random.choice(len(self.query_strategies_))
        print(
            'Strategy Chosen:', type(self.query_strategies_[self.strat_idx]).__name__
        )
        ask_id = self.query_strategies_[self.strat_idx].make_query(self.nquery)

        return ask_id
