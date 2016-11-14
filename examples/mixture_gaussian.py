#!/usr/bin/env python
"""Mixture of Gaussians.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.models import Categorical, Dirichlet, Empirical, InverseGamma, Normal
from scipy.stats import norm


def build_toy_dataset(N):
  pi = np.array([0.4, 0.6])
  mus = [[1, 1], [-1, -1]]
  stds = [[0.1, 0.1], [0.1, 0.1]]
  x = np.zeros((N, 2), dtype=np.float32)
  for n in range(N):
    k = np.argmax(np.random.multinomial(1, pi))
    x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

  return x


N = 500  # number of data points
K = 2  # number of components
D = 2  # dimensionality of data
ed.set_seed(42)

# DATA
x_data = build_toy_dataset(N)

# MODEL
pi = Dirichlet(alpha=tf.constant([1.0]*K))
mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
c = Categorical(logits=ed.tile(ed.logit(pi), [N, 1]))
x = Normal(mu=tf.gather(mu, c), sigma=tf.gather(sigma, c))

# INFERENCE
qpi_alpha = tf.nn.softplus(tf.Variable(tf.random_normal([K])))
qmu_mu = tf.Variable(tf.random_normal([K, D]))
qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([K, D])))
qsigma_alpha = tf.nn.softplus(tf.Variable(tf.random_normal([K, D])))
qsigma_beta = tf.nn.softplus(tf.Variable(tf.random_normal([K, D])))
qc_logits = tf.Variable(tf.random_normal([N, K])) # TODO technically only need [N, K-1]

qpi = Dirichlet(alpha=qpi_alpha)
qmu = Normal(mu=qmu_mu, sigma=qmu_sigma)
qsigma = InverseGamma(alpha=qsigma_alpha, beta=qsigma_beta)
qc = Categorical(logits=qc_logits)

data = {x: x_data}
inference = ed.MFVI({pi: qpi, mu: qmu, sigma: qsigma, c: qc}, data)
inference.initialize(n_samples=5)

sess = ed.get_session()
init = tf.initialize_all_variables()
init.run()

for _ in range(500):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  t = info_dict['t']
  if t == 1 or t % inference.n_print == 0:
    loss = info_dict['loss']
    print("Inferred membership probabilities:")
    print(sess.run(qpi.mean()))
    print("Inferred cluster means:")
    print(sess.run(qmu.mean()))
    print("Inferred cluster standard deviations:")
    print(sess.run(qmu.std()))
