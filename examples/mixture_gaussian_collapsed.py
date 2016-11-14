#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf

from edward.models import Categorical, InverseGamma, Mixture, \
    MultivariateNormalDiag, Normal
from scipy.stats import norm

plt.style.use('ggplot')


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
# K = 2  # number of components
K = 1  # number of components
D = 2  # dimensionality of data
ed.set_seed(42)

# DATA
x_data = build_toy_dataset(N)
# plt.scatter(x_data[:, 0], x_data[:, 1])
# plt.axis([-3, 3, -3, 3])
# plt.title("Simulated dataset")
# plt.show()

# MODEL
mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
cat = Categorical(logits=tf.zeros([N, K]))
components = [
    MultivariateNormalDiag(mu=tf.ones([N, 1]) * tf.gather(mu, k),
                           diag_stdev=tf.ones([N, 1]) * tf.gather(sigma, k))
    for k in range(K)]
# components = [
#     MultivariateNormalDiag(mu=tf.ones([N, 1]) * mu.value()[k],
#                            diag_stdev=tf.ones([N, 1]) * sigma.value()[k])
#     for k in range(K)]
# components = [
#     MultivariateNormalDiag(mu=-tf.ones([N, D]),
#                            diag_stdev=0.1 * tf.ones([N, D])),
#     MultivariateNormalDiag(mu=-1.0 * tf.ones([N, D]),
#                            diag_stdev=0.1 * tf.ones([N, D]))]
x = Mixture(cat=cat, components=components)
# sess = ed.get_session()
# x_data = x.value().eval()
# plt.scatter(x_data[:, 0], x_data[:, 1])
# plt.axis([-3, 3, -3, 3])
# plt.title("Simulated dataset")
# plt.show()

# INFERENCE
# TODO the gradients don't seem to be propagated correctly;
# not sure if it's tf.gather, Mixture, Categorical,...?
from edward.models import PointMass
qmu = PointMass(params=tf.Variable(tf.random_normal([K, D])))
qsigma = PointMass(params=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))

data = {x: x_data}
inference = ed.MAP({mu: qmu, sigma: qsigma}, data)
inference.initialize(n_iter=5000)
# qmu = Normal(
#   mu=tf.Variable(tf.random_normal([K, D])),
#   sigma=tf.nn.softplus(tf.Variable(tf.zeros([K, D]))))
# qsigma = InverseGamma(
#   alpha=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))),
#   beta=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))

# data = {x: x_data}
# inference = ed.KLqp({mu: qmu, sigma: qsigma}, data)
# inference.initialize(n_samples=20, n_iter=5000)
# T = int(1e4)
# from edward.models import Empirical
# qmu = Empirical(params=tf.Variable(tf.zeros([T, K, D])))
# qsigma = Empirical(params=tf.Variable(tf.zeros([T, K, D])))

# data = {x: x_data}
# inference = ed.HMC({mu: qmu, sigma: qsigma}, data)
# inference.initialize()

sess = ed.get_session()
init = tf.initialize_all_variables()
init.run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  t = info_dict['t']
  if t % inference.n_print == 0:
    print("Inferred cluster means:")
    print(sess.run(qmu.value()))
  #   print(sess.run(qmu.mean()))
  #   print("Inferred cluster standard deviations:")
  #   print(sess.run(qmu.std()))

# # Average per-cluster and per-data point likelihood over many posterior samples.
# log_liks = []
# for s in range(100):
#   zrep = {'pi': qpi.sample(()),
#           'mu': qmu.sample(()),
#           'sigma': qsigma.sample(())}
#   log_liks += [model.predict(data, zrep)]

# log_liks = tf.reduce_mean(log_liks, 0)

# # Choose the cluster with the highest likelihood for each data point.
# clusters = tf.argmax(log_liks, 0).eval()
# plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
# plt.axis([-3, 3, -3, 3])
# plt.title("Predicted cluster assignments")
# plt.show()
