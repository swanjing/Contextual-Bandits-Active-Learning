"""Random Sampling
"""
import numpy as np

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state, zip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class RandomSampling(QueryStrategy):

    """Random sampling

    This class implements the random query strategy. A random entry from the
    unlabeled pool is returned for each query.

    Parameters
    ----------
    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------
    random_states\_ : np.random.RandomState instance
        The random number generator using.

    Examples
    --------
    Here is an example of declaring a RandomSampling query_strategy object:

    .. code-block:: python

       from libact.query_strategies import RandomSampling

       qs = RandomSampling(
                dataset, # Dataset object
            )
    """

    def __init__(self, dataset, args, **kwargs):
        super(RandomSampling, self).__init__(dataset, args, **kwargs)

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

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

    @inherit_docstring_from(QueryStrategy)
    def make_query(self, num=1):
        """Return the index of the sample to be queried and labeled and
        selection score of each sample. Read-only.

        No modification to the internal states.

        Returns
        -------
        ask_id : int
            The index of the next unlabeled sample to be queried and labeled.
            
        """
        dataset = self.dataset
        _, unlabeled_entry_ids = dataset.get_unlabeled_entries()
        entry_id = unlabeled_entry_ids[self.random_state_.choice(len(unlabeled_entry_ids), size=num, replace=False)]
        return entry_id
