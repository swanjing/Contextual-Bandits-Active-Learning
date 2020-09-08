"""Multi Bandits Active Learning (UBAL)

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

class MultiBanditsActiveLearning(QueryStrategy):

    """Multi Bandits Active Learning (MBAL) query strategy.

    MBAL is an ensemble-based active learning algorithm that adaptively choose among existing
    query strategies to decide which data to make query. It makes use of the EXP4.P algorithm
    and updates its probability vector according to the ALBL algorithm.


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

    delta : float, optional (default=0.1)
        Parameter for Exp4.P.

    uniform_sampler : {True, False}, optional (default=True)
        Determining whether to include uniform random sample as one of arms.

    pmin : float, 0<pmin< :math:`\frac{1}{len(query\_strategies)}`,\
                  optional (default= :math:`\frac{\sqrt{\log{N}}}{KT}`)
        Parameter for Exp4.P. The minimal probability for random selection of
        the arms (aka the underlying active learning algorithms). N = K =
        number of query_strategies, T is the number of query budgets.

    model : :py:mod:`libact.models` object instance
        The learning model used for the task.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.
        
    reward_fn : reward function, optional (default = IW-ACC)

    Attributes
    ----------
    query_strategies\_ : list of :py:mod:`libact.query_strategies` object instance
        The active learning algorithm instances.

    exp4p\_ : instance of Exp4P object
        The multi-armed bandit instance.

    queried_hist\_ : list of integer
        A list of entry_id of the dataset which is queried in the past.

    random_states\_ : np.random.RandomState instance
        The random number generator using.

    Examples
    --------
    Here is an example of how to declare a ActiveLearningByLearning
    query_strategy object:

    .. code-block:: python

       from libact.query_strategies import ActiveLearningByLearning
       from libact.query_strategies import HintSVM
       from libact.query_strategies import UncertaintySampling
       from libact.models import LogisticRegression

        qs = MultiBanditsActiveLearning(trn_ds, args,
                    query_strategies=[
                        RandomSampling(trn_ds, args, T=quota, model=model_mbal),
                        UncertaintySampling(trn_ds, args, T=quota, model=model_mbal, method='lc'),
                        UncertaintySampling(trn_ds, args, T=quota, model=model_mbal, method='sm'),
                        UncertaintySampling(trn_ds, args, T=quota, model=model_mbal, method='entropy'),
                        RepresentativeSampling(trn_ds, args, T=quota, model=model_mbal)
                    ],
                    T=quota,
                    uniform_sampler=False,
                    model=model_mbal,
                    reward_fn = custom_rw
                )

    References
    ----------
    .. [1] Wei-Ning Hsu, and Hsuan-Tien Lin. "Active Learning by Learning."
           Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.

    """

    def __init__(self, dataset, args, **kwargs):
        super(MultiBanditsActiveLearning, self).__init__(dataset, args, **kwargs)
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

        # parameters for Exp4.p
        self.delta = kwargs.pop('delta', 0.1)

        # query budget
        self.T = kwargs.pop('T', None)
        if self.T is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'T'"
            )

        _, self.unlabeled_entry_ids = self.dataset.get_unlabeled_entries()
        self.unlabeled_invert_id_idx = {}
        for i, idx in enumerate(self.dataset.get_unlabeled_entries()[1]):
            self.unlabeled_invert_id_idx[idx] = i

        self.uniform_sampler = kwargs.pop('uniform_sampler', True)
        if not isinstance(self.uniform_sampler, bool):
            raise ValueError("'uniform_sampler' should be {True, False}")

        self.pmin = kwargs.pop('pmin', None)
        n_algorithms = (len(self.query_strategies_) + self.uniform_sampler)
        if self.pmin and (self.pmin > (1. / n_algorithms) or self.pmin < 0):
            raise ValueError("'pmin' should be 0 < pmin < "
                             "1/len(n_active_algorithm)")

        self.reward_fn = kwargs.pop('reward_fn', None)

        self.exp4p_ = Exp4P(
            query_strategies=self.query_strategies_,
            T=self.T,
            delta=self.delta,
            pmin=self.pmin,
            unlabeled_invert_id_idx=self.unlabeled_invert_id_idx,
            uniform_sampler=self.uniform_sampler
        )

        # classifier instance
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

        self.query_dist = None

        self.W = []
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

    def calc_query(self):
        """Calculate the sampling query distribution"""
        # initial query
        if self.query_dist is None:
            self.query_dist = self.exp4p_.next(-1, None, None, self.nquery)
        else:
            self.query_dist = self.exp4p_.next(
                self.calc_reward_fn(),
                self.queried_hist_[-1],
                self.dataset.data[self.queried_hist_[-1]][1],
                self.nquery
            )
        return

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        """Updating for IW-ACC reward function"""
        for strategy in self.query_strategies_:
            strategy.update(entry_id, label)

        try:
            ask_idx = self.unlabeled_invert_id_idx[int(entry_id)]
        except TypeError:
            ask_idx = [self.unlabeled_invert_id_idx[int(entry)] for entry in entry_id]
        self.W.append(1. / np.sum(self.query_dist[ask_idx]))
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
        try:
            _, unlabeled_entry_ids = dataset.get_unlabeled_entries()
        except ValueError:
            # might be no more unlabeled data left
            return

        self.calc_query()
        try:
            ask_idx = self.random_state_.choice(
                np.arange(len(self.unlabeled_invert_id_idx)),
                size=num,
                p=self.query_dist
            )
        except ValueError:
            self.query_dist /= np.sum(self.query_dist)
            ask_idx = self.random_state_.choice(
                np.arange(len(self.unlabeled_invert_id_idx)),
                size=num,
                p=self.query_dist
            )
        ask_id = self.unlabeled_entry_ids[ask_idx]

        is_unlabeled = [id in unlabeled_entry_ids for id in ask_id]
        if all(is_unlabeled):
            return ask_id
        else:
            print("Error in updating unlabeled entries")
            # self.update(ask_id[~np.array(is_unlabeled)], dataset.data[ask_id[~np.array(is_unlabeled)]][1])

        raise ValueError("Out of query budget")


class Exp4P(object):

    r"""A multi-armed bandit algorithm Exp4.P.

    For the Exp4.P used in ALBL, the number of arms (actions) and number of
    experts are equal to the number of active learning algorithms wanted to
    use. The arms (actions) are the active learning algorithms, where is
    inputed from parameter 'query_strategies'. There is no need for the input
    of experts, the advice of the kth expert are always equal e_k, where e_k is
    the kth column of the identity matrix.

    Parameters
    ----------
    query_strategies : QueryStrategy instances
        The active learning algorithms wanted to use, it is equivalent to
        actions or arms in original Exp4.P.

    unlabeled_invert_id_idx : dict
        A look up table for the correspondance of entry_id to the index of the
        unlabeled data.

    delta : float, >0, optional (default=0.1)
        A parameter.

    pmin : float, 0<pmin<1/len(query_strategies), optional (default= :math:`\frac{\sqrt{log(N)}}{KT}`)
        The minimal probability for random selection of the arms (aka the
        unlabeled data), N = K = number of query_strategies, T is the maximum
        number of rounds.

    T : int, optional (default=100)
        The maximum number of rounds.

    uniform_sampler : {True, False}, optional (default=Truee)
        Determining whether to include uniform random sampler as one of the
        underlying active learning algorithms.

    Attributes
    ----------
    t : int
        The current round this instance is at.

    N : int
        The number of arms (actions) in this exp4.p instance.

    query_models\_ : list of :py:mod:`libact.query_strategies` object instance
        The underlying active learning algorithm instances.

    References
    ----------
    .. [1] Beygelzimer, Alina, et al. "Contextual bandit algorithms with
           supervised learning guarantees." In Proceedings on the International
           Conference on Artificial Intelligence and Statistics (AISTATS),
           2011u.

    """

    def __init__(self, *args, **kwargs):
        """ """
        # QueryStrategy class object instances
        self.query_strategies_ = kwargs.pop('query_strategies', None)
        if self.query_strategies_ is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: "
                "'query_strategies'"
            )
        elif not self.query_strategies_:
            raise ValueError("query_strategies list is empty")

        # whether to include uniform random sampler as one of underlying active
        # learning algorithms
        self.uniform_sampler = kwargs.pop('uniform_sampler', True)

        # n_armss
        if self.uniform_sampler:
            self.N = len(self.query_strategies_) + 1
        else:
            self.N = len(self.query_strategies_)

        # weight vector to each query_strategies, shape = (N, )
        self.w = np.array([1. for _ in range(self.N)])

        # max iters
        self.T = kwargs.pop('T', 100)

        # delta > 0
        self.delta = kwargs.pop('delta', 0.1)

        # n_arms = n_models (n_query_algorithms) in ALBL
        self.K = self.N

        # p_min in [0, 1/n_arms]
        self.pmin = kwargs.pop('pmin', None)
        if self.pmin is None:
            self.pmin = np.sqrt(np.log(self.N) / self.K / self.T)

        self.exp4p_gen = self.exp4p()

        self.unlabeled_invert_id_idx = kwargs.pop('unlabeled_invert_id_idx')
        if not self.unlabeled_invert_id_idx:
            raise TypeError(
                "__init__() missing required keyword-only argument:"
                "'unlabeled_invert_id_idx'"
            )

    def __next__(self, reward, ask_id, lbl, num):
        """For Python3 compatibility of generator."""
        return self.next(reward, ask_id, lbl, num)

    def next(self, reward, ask_id, lbl, num):
        """Taking the label and the reward value of last question and returns
        the next question to ask."""
        # first run don't have reward, TODO exception on reward == -1 only once
        self.nquery=num
        if reward == -1:
            return next(self.exp4p_gen)
        else:
            # TODO exception on reward in [0, 1]
            return self.exp4p_gen.send((reward, ask_id, lbl))

    def exp4p(self):
        """The generator which implements the main part of Exp4.P.

        Parameters
        ----------
        reward: float
            The reward value calculated from ALBL.

        ask_id: integer
            The entry_id of the sample point ALBL asked.

        lbl: integer
            The answer received from asking the entry_id ask_id.

        Yields
        ------
        q: array-like, shape = [K]
            The query vector which tells ALBL what kind of distribution if
            should sample from the unlabeled pool.

        """
        while True:
            # TODO probabilistic active learning algorithm
            # len(self.unlabeled_invert_id_idx) is the number of unlabeled data
            query = np.zeros(len(self.unlabeled_invert_id_idx))
            W = np.sum(self.w)
            p = (1 - self.K * self.pmin) * self.w / W + self.pmin
            strategy_idx = int(np.random.choice(
                np.arange(self.K),
                p=p
            ))
            print(
            'Strategy Chosen:', strategy_idx, type(self.query_strategies_[strategy_idx]).__name__
            )

            try:
                query[self.unlabeled_invert_id_idx[int(self.query_strategies_[strategy_idx].make_query(self.nquery))]] = 1
            except TypeError:
                query[[self.unlabeled_invert_id_idx[int(qr)] for qr in self.query_strategies_[strategy_idx].make_query(self.nquery)]] = 1

            # query vector, shape= = (self.n_unlabeled, )
            query_vector = query

            reward, ask_id, _ = yield query_vector
            rhat = reward / p[strategy_idx]           
            # The original advice vector in Exp4.P in ALBL is a identity matrix
            yhat = rhat
            vhat = 1 / p[strategy_idx]
            self.w[strategy_idx] = self.w[strategy_idx] * np.exp(
                self.pmin / 2 * (
                    yhat + vhat * np.sqrt(
                        np.log(self.N / self.delta) / self.K / self.T
                    )
                )
            )

        raise StopIteration
