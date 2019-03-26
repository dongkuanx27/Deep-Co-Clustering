import tensorflow as tf

import core.general.param_init as pini


class PretrainAutoencoder:
    def __init__(self, config, num_drop_out):
        self.num_dim = config
        self.code_layer = len(config)
        self.num_dropout_layer = num_drop_out
        # Parameters in layers
        self.wi = []
        self.bi = []
        # Encode
        for i in range(0, len(self.num_dim)-1):
            w = pini.weight_variable([self.num_dim[i], self.num_dim[i+1]])
            b = pini.bias_variable([self.num_dim[i+1]])
            self.wi.append(w)
            self.bi.append(b)
        # Decode
        for i in range(1, len(self.num_dim)):
            j = len(self.num_dim)-i
            w = pini.weight_variable([self.num_dim[j], self.num_dim[j-1]])
            b = pini.bias_variable([self.num_dim[j-1]])
            self.wi.append(w)
            self.bi.append(b)

    def run(self, x, keep_prob):
        vision_coef = 1.0
        error = []
        var_list = []
        reg = []
        l2_reg = 0

        # Encode
        zi = x
        for i in range(int(len(self.wi)/2)):
            if i < len(self.wi) / 2 - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            if i < self.num_dropout_layer:
                zj = tf.nn.dropout(zj, keep_prob=keep_prob)
            ni = len(self.wi) - i - 1
            if i == 0:
                z_r = tf.matmul(zj, self.wi[len(self.wi) - 1]) + self.bi[len(self.bi) - 1]
            else:
                z_r = tf.nn.tanh(tf.matmul(zj, self.wi[ni]) + self.bi[ni])
            error_l = tf.reduce_mean(tf.norm(zi - z_r, ord=2, axis=1, keep_dims=True))
            error.append(error_l * vision_coef)
            zi = zj
            reg.append(tf.nn.l2_loss(self.wi[i]) + tf.nn.l2_loss(self.wi[ni]))
            l2_reg = l2_reg + tf.nn.l2_loss(self.wi[i])
            var_list.append([self.wi[i], self.bi[i], self.wi[ni], self.bi[ni]])
        zc = zi
        
        # Decode
        for i in range(int(len(self.wi)/2), len(self.wi)):
            if i < len(self.wi) - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
                if i >= len(self.wi) - 1 - self.num_dropout_layer:
                    zj = tf.nn.dropout(zj, keep_prob=keep_prob)
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            l2_reg = l2_reg + tf.nn.l2_loss(self.wi[i])
            zi = zj
        zo = zi

        # Cosine similarity
        normalize_x = tf.nn.l2_normalize(x, 1)
        normalize_zo = tf.nn.l2_normalize(zo, 1)
        cos_sim = tf.reduce_sum(tf.multiply(normalize_x, normalize_zo), 1, keep_dims=True)
        loss = tf.norm(x - zo, ord=2, axis=1, keep_dims=True)
        dist = tf.norm(x - zo, ord=2, axis=1, keep_dims=True)
        relative_dist = dist / tf.norm(x, ord=2, axis=1, keep_dims=True)
        # self.dist_min = tf.reduce_min(dist)
        # self.dist_max = tf.reduce_max(dist)
        # dist_norm = (dist - self.dist_min) / (self.dist_max - self.dist_min + 1e-12)
        
        xo = tf.concat([zc], 1)
        
        error_all = tf.reduce_mean(loss)
        error.append(error_all)
        # var_list.append([self.w6, self.b6, self.w9, self.b9])
        # error = error_all + error_1 + error_2 + error_3 + error_4 + error_5 + error_6 + error_7
        return xo, error, var_list, l2_reg, reg

    def test(self, x):
        # Encode
        zi = x
        for i in range(int(len(self.wi) / 2)):
            if i < len(self.wi) / 2 - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            zi = zj
        zc = zi
        # Decode
        for i in range(int(len(self.wi) / 20), len(self.wi)):
            if i < len(self.wi) - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            zi = zj
        zo = zi

        # Cosine similarity
        normalize_x = tf.nn.l2_normalize(x, 1)
        normalize_zo = tf.nn.l2_normalize(zo, 1)
        cos_sim = tf.reduce_sum(tf.multiply(normalize_x, normalize_zo), 1, keep_dims=True)
        dist = tf.norm(x - zo, ord=2, axis=1, keep_dims=True)
        # relative_dist = dist / tf.norm(x, ord=2, axis=1, keep_dims=True)
        # dist_norm = (dist - self.dist_min) / (self.dist_max - self.dist_min + 1e-12)
        # xo = tf.concat([zc, relative_dist, cos_sim], 1)
        # xo = tf.concat([zc, relative_dist], 1)
        return dist

    def dcn_run(self, x, keep_prob):
        vision_coef = 1.0
        error = []
        var_list = []
        reg = []
        l2_reg = 0
        # Encode
        zi = x
        for i in range(int(len(self.wi) / 2)):
            if i < len(self.wi) / 2 - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            if i < self.num_dropout_layer:
                zj = tf.nn.dropout(zj, keep_prob=keep_prob)
            ni = len(self.wi) - i - 1
            if i == 0:
                z_r = tf.matmul(zj, self.wi[len(self.wi) - 1]) + self.bi[len(self.bi) - 1]
            else:
                z_r = tf.nn.tanh(tf.matmul(zj, self.wi[ni]) + self.bi[ni])
            error_l = tf.reduce_mean(tf.norm(zi - z_r, ord=2, axis=1, keep_dims=True))
            error.append(error_l * vision_coef)
            zi = zj
            reg.append(tf.nn.l2_loss(self.wi[i]) + tf.nn.l2_loss(self.wi[ni]))
            l2_reg = l2_reg + tf.nn.l2_loss(self.wi[i])
            var_list.append([self.wi[i], self.bi[i], self.wi[ni], self.bi[ni]])
        zc = zi
        # Decode
        for i in range(int(len(self.wi) / 2), len(self.wi)):
            if i < len(self.wi) - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
                if i >= len(self.wi) - 1 - self.num_dropout_layer:
                    zj = tf.nn.dropout(zj, keep_prob=keep_prob)
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            l2_reg = l2_reg + tf.nn.l2_loss(self.wi[i])
            zi = zj
        zo = zi

        # Cosine similarity
        normalize_x = tf.nn.l2_normalize(x, 1)
        normalize_zo = tf.nn.l2_normalize(zo, 1)
        cos_sim = tf.reduce_sum(tf.multiply(normalize_x, normalize_zo), 1, keep_dims=True)
        loss = tf.norm(x - zo, ord=2, axis=1, keep_dims=True)
        dist = tf.norm(x - zo, ord=2, axis=1, keep_dims=True)
        relative_dist = dist / tf.norm(x, ord=2, axis=1, keep_dims=True)
        # self.dist_min = tf.reduce_min(dist)
        # self.dist_max = tf.reduce_max(dist)
        # dist_norm = (dist - self.dist_min) / (self.dist_max - self.dist_min + 1e-12)
        xo = zc
        # xo = tf.concat([zc, relative_dist], 1)
        error_all = tf.reduce_mean(loss)
        error.append(error_all)
        # var_list.append([self.w6, self.b6, self.w9, self.b9])
        # error = error_all + error_1 + error_2 + error_3 + error_4 + error_5 + error_6 + error_7
        return xo, error, var_list, l2_reg, reg


if __name__ == '__main__':
    # file_name = '../dd/Attack.2.feature.csv'
    print('test autoencoder')

