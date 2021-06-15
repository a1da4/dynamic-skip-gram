import numpy as np
import logging

class Adam():
    def __init__(self, alpha, beta1, beta2, eta, params):
        """ initialization
        :param alpha: float, learning rate
        :param beta1, beta2: float, decay rate of 1st/2nd moment estimate
        :param eta: float, regularizer
        :param params: {key: np.array.shape}, target parameter(s)
        """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.name2id = {key: i for i, key in enumerate(params.keys())}
        self.m1 = [np.zeros(param) for param in params.values()]
        self.m2 = [np.zeros(param) for param in params.values()]

    def step(self, t, i, param_name, param, grad):
        """ obtain update steps (t -> t+1)
        :param t: int, timestep
        :param i: int, index part of param
        :param param_name: str, name of param defined in __init__
        :param param: np.array, target parameter
        :param grad: np.array, gradient
        :return: d (update step)
        """
        param_id = self.name2id[param_name]
        self.m1[param_id][i] += self.beta1 * self.m1[param_id][i] + (1 - self.beta1) * grad
        self.m2[param_id][i] += self.beta2 * self.m2[param_id][i] + (1 - self.beta2) * (grad**2)
        bias_corrected_m1 = self.m1[param_id][i] / (1 - self.beta1**t)
        bias_corrected_m2 = self.m2[param_id][i] / (1 - self.beta2**t)
        d = self.alpha * bias_corrected_m1 / (np.sqrt(bias_corrected_m2) + self.eta)
        return d
    
    def update_enforce_positive(self, param, d):
        """ update with enforcing positive value (mirror ascent)
        :param param: np.array, target parameter
        :param d: np.array, update step obtained from Adam.step()
        :return: updated_param (np.array)
        """
        tmp = 1/2 * param*d
        return tmp + np.sqrt(tmp**2 + param**2)
