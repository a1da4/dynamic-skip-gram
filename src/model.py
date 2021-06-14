import logging

import numpy as np

from ioutils import *
from functions import *


class SkipGramSmoothing:
    def __init__(self, vocab, T, dim, D, taus, positive, negative):
        """initialization
        :param vocab: list(str), vocab
        :param T: int, time steps
        :param dim: int, dimension
        :param D: float, global diffusion
        :param taus: list(int), times observed (e.g. [1800, 1802, 1804, 1807])
        :param positive, negative: path, positive/negative samples counted in preproecss.py
        """
        self.vocab = vocab
        self.V = len(self.vocab)
        self.T = T
        self.D = dim
        self.mean_target = np.zeros([self.V, dim, T])
        self.mean_context = np.zeros([self.V, dim, T])
        self.val0 = 1
        self.vals = [D * (taus[t + 1] - taus[t]) for t in range(T - 1)]
        # cov: tridiagonal matrix
        self.cov = np.zeros([T, T])
        for i in range(self.T):
            for j in range(self.T):
                if i == j:
                    if i == 0:
                        v = self.val0 ** (-1) + self.vals[i] ** (-1)
                    elif i == self.T - 1:
                        v = self.val0 ** (-1) + self.vals[i - 1] ** (-1)
                    else:
                        v = (
                            self.val0 ** (-1)
                            + self.vals[i - 1] ** (-1)
                            + self.vals[i] ** (-1)
                        )
                    self.cov[i][j] = v
                if i + 1 == j or i == j + 1:
                    v = -self.vals[min(i, j)] ** (-1)
                    self.cov[i][j] = v
        # cov = B^T@B, B: bidiagonal matrix
        # v, w: variational parameters
        # e.g. T=4
        # B = [[v_u,1                  ]
        #      [w_u,1 v_u,2            ]
        #      [      w_u,2 v_u,3      ]
        #      [            w_u,3 v_u,4]]
        B = np.linalg.cholesky(self.cov)
        v = np.diag(B)
        w = np.diag(B, -1)
        self.v_target = np.tile(v, (self.V, dim, 1))
        self.w_target = np.tile(w, (self.V, dim, 1))
        self.v_context = np.tile(v, (self.V, dim, 1))
        self.w_context = np.tile(w, (self.V, dim, 1))
        # load positive and negative samples
        self.positives = load_3d_matrix(
            positive, z_size=self.T, x_size=self.V, y_size=self.V
        )
        self.negatives = load_3d_matrix(
            negative, z_size=self.T, x_size=self.V, y_size=self.V
        )

    def _B_eachdim(self, v, w, word_id, dim_id):
        """reconstruct matrix B from v and w
        :param v, w: float, variational parameters of B, B^T@B = cov
        :param word_id, dim_id: int, word an dimension index
        :return: matrix B for word_id, dim_id overtime (T, T)
        """
        return np.diag(v[word_id][dim_id]) + np.diag(w[word_id][dim_id], 1)

    def _sample_vec_eachdim(self, v, w, mean, word_id, dim_id):
        """sample single vector (Eq. S8 and S9)
        :param v, w: float, variational parameters of B, B^T@B = cov
        :param mean: float, mean
        :param word_id, dim_id: int, word and dimension index
        :return: sampled_vector
        """
        e = np.random.normal(0, 1, self.T)
        B_inv = np.linalg.inv(self._B_eachdim(v, w, word_id, dim_id))
        x = B_inv @ e
        sampled_vec = mean[word_id][dim_id] + x
        return sampled_vec, x

    def _gamma(self, i, target_vec, sampled_context_ids, sampled_v):
        """gamma function in each time (Eq. S12)
        :param i: int, index of sampled_target_ids
        :param target_vec: vec(D, T)
        :param sampled_context_ids: list(int), context ids
        :param sampled_v: vec(Vsampled, D, T), sampled context vectors
        :return: gamma(D, T)
        """
        # TODO 時期で並列化
        gamma = np.zeros([self.D, self.T])
        target_vec = target_vec.T
        logging.debug(f" [_gamma] target_vec (size): {target_vec.shape}")
        for t in range(self.T):
            # gamma_eachtime: (D)
            gamma_eachtime = np.zeros([self.D])
            # TODO 各文脈語で同時に計算できないか。アダマール積 * などを使う？
            for sampled_context_id in range(len(sampled_context_ids)):
                global_context_id = sampled_context_ids[sampled_context_id]
                n_p = self.positives[t][i][global_context_id]
                n_n = self.negatives[t][i][global_context_id]
                context_vec = sampled_v[sampled_context_id].T[t]
                gamma_eachtime += (
                    (n_p + n_n) * sigmoid(target_vec[t].T @ context_vec) - n_n
                ) * context_vec
            gamma.T[t] += gamma_eachtime
            logging.debug(f" [_gamma] gamma (nonzero): {np.count_nonzero(gamma)}")
        return gamma

    def _estimate_gradient(
        self, i, rate, target_vec, sampled_context_ids, sampled_v, sampled_x_target
    ):
        """estimate gradient for each word
        :param i: int, target word index
        :param rate: float, % of vocab
        :param target_vec: sampled target vector (D, T)
        :param sampled_context_ids: list(int), sampled context indices (V*rate)
        :param sampled_v: sampled context vectors (V*rate, D, T)
        :param sampled_x_target: sampled gaussian noise (V*rate, D, T)
        :return: mean_grad (D, T), v_grad (D, T), w_grad (D, T-1)
        """
        gamma = self._gamma(i, target_vec, sampled_context_ids, sampled_v)
        # mean_grad: (D, T)
        mean_grad = (rate * gamma.T - self.cov @ target_vec.T).T
        logging.debug(f" [_estimate_gradient] mean_grad (shape): {mean_grad.shape}")
        v_grad = np.zeros([self.D, self.T])
        w_grad = np.zeros([self.D, self.T - 1])
        for d in range(self.D):
            B_T_inv = np.linalg.inv(
                self._B_eachdim(self.v_target, self.w_target, i, d).T
            )
            logging.debug(f" [_estimate_gradient] B_T_inv (shape): {B_T_inv.shape}")
            # y_eachdim: (T)
            y_eachdim = B_T_inv @ mean_grad[d]
            v_grad[d] += -y_eachdim * sampled_x_target[i][d] - 1 / self.v_target[i][d]
            w_grad[d] += -y_eachdim[:-1] * sampled_x_target[i][d][1:]
        logging.debug(f" [_estimate_gradient] v_grad (shape): {v_grad.shape}")
        logging.debug(f" [_estimate_gradient] w_grad (shape): {w_grad.shape}")
        return mean_grad, v_grad, w_grad

    def pretrain(self, iter, alpha=0.01, beta1=0.9, beta2=0.999, eta=1e-8, rate=0.1):
        """pre-train using minibatch (10% of vocab)
        :param iter: int, iteration
        :param alpha: float, learning rate of Adam
        :param beta1, beta2: float, decay rate of 1st/2nd moment estimate of Adam
        :param eta: float, regularizer of Adam
        :param rate: float, % of vocab
        """

        def sample_minibatch(V, rate=0.1):
            """sampling vocab
            :param V: int, num of vocab
            :param rate: float, % of vocab
            :return: sampled_vocab
            """
            sampled_vocab = np.sort(np.random.choice(V, int(V * rate)))
            return sampled_vocab

        # optimize with Adam

        for step in range(iter):
            logging.debug(f" [pretrain] {step}th iteration")
            logging.info(" [pretrain] sampling minibatch...")
            sampled_target_ids = sample_minibatch(len(self.vocab))
            sampled_context_ids = sample_minibatch(len(self.vocab))
            logging.debug(
                f" [pretrain] minibatch(target): {len(sampled_target_ids)} words"
            )
            logging.debug(
                f" [pretrain] minibatch(context): {len(sampled_context_ids)} words"
            )

            sampled_u = np.zeros([len(sampled_target_ids), self.D, self.T])
            sampled_x_target = np.zeros(sampled_u.shape)
            sampled_v = np.zeros([len(sampled_context_ids), self.D, self.T])
            sampled_x_context = np.zeros(sampled_v.shape)

            logging.info(" [pretrain] sampling vectors...")
            for i in range(len(sampled_target_ids)):
                # TODO 各次元でまとめて最適化
                for d in range(self.D):
                    sampled_u_eachdim, x_target = self._sample_vec_eachdim(
                        self.v_target,
                        self.w_target,
                        self.mean_target,
                        sampled_target_ids[i],
                        d,
                    )
                    sampled_v_eachdim, x_context = self._sample_vec_eachdim(
                        self.v_context,
                        self.w_context,
                        self.mean_context,
                        sampled_context_ids[i],
                        d,
                    )
                    sampled_u[i][d] += sampled_u_eachdim
                    sampled_x_target[i][d] += x_target
                    sampled_v[i][d] += sampled_v_eachdim
                    sampled_x_context[i][d] += x_context

            logging.debug(f" [pretrain] sampled_u (size): {sampled_u.shape}")
            logging.debug(f" [pretrain] sampled_v (size): {sampled_v.shape}")
            
            # estimate gradient
            # TODO まずは target のパラメータを更新。 context も同様に行う
            logging.info(" [pretrain] estimate gradient and update...")
            for i in range(len(sampled_target_ids)):
                # target_vec: (D, T)
                target_vec = sampled_u[i]
                mean_grad, v_grad, w_grad = self._estimate_gradient(
                    i,
                    rate,
                    target_vec,
                    sampled_context_ids,
                    sampled_v,
                    sampled_x_target,
                )

    def train(self, iter, lr):
        """train using full vocab
        :param iter: int, iteration
        :param lr: float, learning rate
        """
