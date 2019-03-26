import numpy as np

import core.general.stat_lib as slib


class GMMEM:
    def __init__(self, config):
        # DMM config
        self.dmm_config = config['general']
        self.input_dim = self.dmm_config[0]
        self.num_mixture = self.dmm_config[1]
        self.num_dynamic_dim = self.dmm_config[2]
        # Mixture modeling
        self.gmm_config = [self.num_mixture, self.input_dim, self.num_dynamic_dim]
        self.gmm = slib.GaussianMixtureModeling(self.gmm_config)

    def model(self, x, tolerance):
        phi, mixture_mean, mixture_dev, mixture_cov = self.gmm.em_learning(x, tolerance)
        return phi, mixture_mean, mixture_dev, mixture_cov

    def test(self, x, phi, mixture_mean, mixture_dev, mixture_cov):
        # Augmenting input: [batch_size, num_mixture, num_dim]
        x_t = np.reshape(x, [-1, 1, self.input_dim])
        x_t = np.tile(x_t, (1, self.num_mixture, 1))

        likelihood, _ = self.gmm.em_evaluate(x_t, phi, mixture_mean, mixture_dev, mixture_cov)
        return likelihood
