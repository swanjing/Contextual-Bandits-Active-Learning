"""
Contextual Bandit strategies.
"""
from __future__ import absolute_import

from .uniform_sampling import UniformSampling
from .fixed_policy_sampling import FixedPolicySampling
# from .posterior_bnn_sampling import PosteriorBNNSampling
from .neural_linear_sampling import NeuralLinearPosteriorSampling
from .linear_full_posterior_sampling import LinearFullPosteriorSampling
from .bootstrapped_bnn_sampling import BootstrappedBNNSampling
from .parameter_noise_sampling import ParameterNoiseSampling

__all__ = [
    'UniformSampling',
    'FixedPolicySampling',
    # 'PosteriorBNNSampling',
    'NeuralLinearPosteriorSampling',
    'LinearFullPosteriorSampling',
    'BootstrappedBNNSampling',
    'ParameterNoiseSampling',
]
