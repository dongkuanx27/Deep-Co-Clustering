import core.general.param_init as pini
import tensorflow as tf

import core.general.stat_lib as slib


class GMMEstimationNetNvi:
    def __init__(self, config):
        # DMM config
        self.dmm_config = config['general']  # [40, num_clus_r, 40]
        self.input_dim = self.dmm_config[0]  # 40
        self.num_mixture = self.dmm_config[1]  # num_clus_r
        self.num_dynamic_dim = self.dmm_config[2]  # 40
        # Layer 1
        layer_1_config = config['layer_1']  # 80
        self.output_d_1 = layer_1_config[0] # 80
        self.w1 = pini.weight_variable([self.input_dim, self.output_d_1])
        self.b1 = pini.bias_variable([self.output_d_1])
        # Layer 2
        # layer_2_config = config['layer_2']
        # self.output_d_2 = layer_2_config[0]
        self.w2 = pini.weight_variable([self.output_d_1, self.num_mixture])
        self.b2 = pini.bias_variable([self.num_mixture])
        # Mixture modeling
        self.gmm_config = [self.num_mixture, self.input_dim, self.num_dynamic_dim]
        self.kmm = slib.SoftKMeansMixtureModeling(self.gmm_config)
        self.gmm = slib.GaussianMixtureModeling(self.gmm_config)
        self.var_list = [self.w1, self.b1, self.w2, self.b2]

    def run(self, x, keep_prob):
        # Mixture estimation network
        # Layer 1
        z1 = tf.nn.softsign(tf.matmul(x, self.w1) + self.b1)  #softsign, tanh
        # Layer 2
        z1_drop = tf.nn.dropout(z1, keep_prob)
        p = tf.nn.softmax(tf.matmul(z1_drop, self.w2) + self.b2)
        
        xx = tf.matmul(z1_drop, self.w2) + self.b2 # 'xx' is the representation on the last layer of estimation NN

        # Log likelihood
        gmm_energy, pen_dev, likelihood, phi, x_t, p_t, z_p, z_t, mixture_mean, mixture_dev, mixture_cov, mixture_dev_det = self.gmm.vi_learning(xx, p)
        # k_dist = self.kmm.eval(x, p)
        # mixture_dev_0 = mixture_dev[:, 0]
        # _, prior_energy = self.inverse_gamma.eval(mixture_dev, phi)
        # train
        # energy = gmm_energy
        loss = gmm_energy
        # loss = k_dist
        # reg = 0
        # for w in self.var_list:
        #     reg = reg + tf.nn.l2_loss(w)
        # reg = tf.reduce_sum(phi * tf.log(phi+1e-12)) + 0.1*tf.reduce_mean(tf.reduce_sum(- p * tf.log(p + 1e-12), 1))
        return loss, pen_dev, likelihood, p, self.var_list, x_t, p_t, z_p, z_t, mixture_mean, mixture_dev, mixture_cov, mixture_dev_det
        # return loss, reg

    def model(self, x, keep_prob):
        # Mixture estimation network
        # Layer 1
        z1 = tf.nn.softsign(tf.matmul(x, self.w1) + self.b1)
        # Layer 2
        z1_drop = tf.nn.dropout(z1, keep_prob)
        p = tf.nn.softmax(tf.matmul(z1_drop, self.w2) + self.b2)
        # Log likelihood
        _, _, _, phi, mixture_mean, mixture_dev, mixture_cov = self.gmm.eval(x, p)
        return phi, mixture_mean, mixture_dev, mixture_cov

    def test(self, x, phi, mixture_mean, mixture_dev, mixture_cov):
        likelihood = self.gmm.test(x, phi, mixture_mean, mixture_dev, mixture_cov)
        return likelihood
