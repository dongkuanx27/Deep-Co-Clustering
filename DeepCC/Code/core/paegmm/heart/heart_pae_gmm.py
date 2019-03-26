import numpy as np
import tensorflow as tf

import core.data.heart.heart as heart
import core.general.gmm_estimation_net as dgmmb
import core.general.pretrain_autoencoder as ae


'''
DMM for kddcup10, using batch training
'''


def get_key(item):
    return item[0]


class HeartPaeGmm:
    def __init__(self, ae_config, gmm_config, num_dropout):
        self.num_input_dim = ae_config[0]
        self.autoencoder = ae.PretrainAutoencoder(ae_config, num_dropout)
        self.e_net = dgmmb.GMMEstimationNet(gmm_config)

    @staticmethod
    def get_data(input_file):
        data = heart.HeartData(input_file)
        data.get_clean_training_testing_data(0.5)
        return data

    @staticmethod
    def gaussian_normalization(train_x):
        mu = np.mean(train_x, axis=0)
        dev = np.std(train_x, axis=0)
        norm_x = (train_x - mu) / (dev + 1e-12)
        # print norm_x
        return norm_x

    @staticmethod
    def minmax_normalization(x, base):
        min_val = np.min(base, axis=0)
        max_val = np.max(base, axis=0)
        norm_x = (x - min_val) / (max_val - min_val + 1e-12)
        # print norm_x
        return norm_x

    @staticmethod
    def output_code(train_out, num_test_points):
        fout = open('../result/heart_output_batch_1.csv', 'w')
        for i in range(num_test_points):
            fout.write(str(train_out[i, 0]) + ',' + str(train_out[i, 1]) + ',' + str(train_out[i, 2]) + ','
                       + str(train_out[i, 3]) + '\n')
            # fout.write(str(train_out[i, 0]) + ',' + str(train_out[i, 1]) + ',' + str(train_out[i, 2]) + '\n')
        fout.close()

    def correctness(self, predict, truth, num_test_points):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(num_test_points):
            if predict[i, 0] > 0.5:
                if truth[i, 0] > 0.5:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if truth[i, 0] > 0.5:
                    fn = fn + 1
                else:
                    tn = tn + 1
        return tp, fp, tn, fn

    def accuracy(self, predict_lh, num_test_points, test_y, outlier_ratio):
        tmp = []
        for i in range(num_test_points):
            p = (predict_lh[i], i)
            tmp.append(p)
        tmp.sort(key=get_key)
        predict = np.zeros([num_test_points, 1])
        num_tag = int(num_test_points * outlier_ratio)
        for i in range(num_tag):
            p = tmp[i]
            idx = p[1]
            # print p[0], test_y[idx, 0]
            predict[idx, 0] = 1
        tp, fp, tn, fn = self.correctness(predict, test_y, num_test_points)
        print tp, fp, tn, fn
        # print test_y
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
        print precision, recall, f1
        return precision, recall, f1

    def run(self, data, train_epochs):
        sess = tf.InteractiveSession()
        # Data
        train_x = data.train_data
        test_x = data.test_data
        test_y = data.test_label
        base = np.concatenate([train_x, test_x], 0)
        # num_train_points = data.num_train_points
        num_train_points = data.num_train_points
        num_test_points = data.num_test_points
        # num_dirty_points = data.num_dirty_points

        train_norm_x = self.minmax_normalization(train_x, base)
        test_norm_x = self.minmax_normalization(test_x, base)
        # train_norm_x = train_x
        # test_norm_x = test_x

        # Setup
        train_x_v = tf.placeholder(dtype=tf.float64, shape=[None, self.num_input_dim])
        test_x_v = tf.placeholder(dtype=tf.float64, shape=[None, self.num_input_dim])
        y = tf.placeholder(dtype=tf.float64, shape=[None, 1])
        keep_prob = tf.placeholder(tf.float64)
        batch_size = 128
        x_b = tf.train.shuffle_batch(
            [train_norm_x],
            batch_size=batch_size,
            num_threads=4,
            capacity=3000,
            enqueue_many=True,
            min_after_dequeue=1000)
        # Autoencoder
        z_b, error_b, var_list_b, l2_reg_b, reg_b = self.autoencoder.run(x_b, keep_prob)
        train_z, train_error, train_var_list, train_l2_reg, train_reg = self.autoencoder.run(train_x_v, keep_prob)
        test_z, test_error, test_var_list, test_l2_reg, test_reg = self.autoencoder.run(test_x_v, keep_prob)
        test_error = self.autoencoder.test(test_x_v)
        # Pretraining
        pretrain_step = []
        pretrain_obj = []
        for i in range(len(var_list_b)):
            obj_i = error_b[i] + reg_b[i] * 0.1
            obj_oa = train_error[i] + train_reg[i] * 0.1
            train_step_i = tf.train.AdamOptimizer(1e-4).minimize(obj_i, var_list=var_list_b[i])
            pretrain_step.append(train_step_i)
            pretrain_obj.append(obj_oa)

        # Joint fine training
        error_b_oa = 0
        for error_k in error_b:
            error_b_oa = error_b_oa + error_k
        error_oa = 0
        for error_k in train_error:
            error_oa = error_oa + error_k
        reconstruction_error = train_error[len(train_error) - 1]
        # GMM Membership estimation
        loss_b, pen_dev_b, likelihood_b = self.e_net.run(z_b, keep_prob)
        loss, pen_dev, likelihood = self.e_net.run(train_z, keep_prob)
        model_phi, model_mean, model_dev, model_cov = self.e_net.model(train_z, keep_prob)
        # testing
        test_likelihood = self.e_net.test(test_z, model_phi, model_mean, model_dev, model_cov)
        z_o = tf.concat([test_z, y], 1)
        # Train step
        # obj_b = error_oa_batch
        obj = error_b_oa + 0.1 * l2_reg_b
        # obj = error_b_oa + loss_b * 0.1 + 0.1 * reg_b
        obj_oa = error_oa + 0.1 * train_l2_reg
        # obj_oa = error_oa + loss * 0.1 + 0.1 * reg
        # obj = error_oa
        train_step = tf.train.AdamOptimizer(1e-4).minimize(obj)
        # train_step_1 = tf.train.AdamOptimizer(1e-4).minimize(obj)
        # train_step_2 = tf.train.AdamOptimizer(1e-5).minimize(obj)
        # train_step_3 = tf.train.AdamOptimizer(1e-5).minimize(obj)
        # GMM training
        obj_gmm_b = loss_b + pen_dev_b * 0.05
        obj_gmm = loss * 0.1 + pen_dev * 0.05

        train_step_gmm = tf.train.AdamOptimizer(1e-4).minimize(obj_gmm, var_list=self.e_net.var_list)

        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        epoch_tot = train_epochs
        num_step = num_train_points / batch_size + 1
        for k in range(len(pretrain_step)):
            train_step_pre_k = pretrain_step[k]
            obj_k = pretrain_obj[k]
            for i in range(epoch_tot):
                for j in range(num_step):
                    train_step_pre_k.run(feed_dict={keep_prob: 0.5})
                if (i + 1) % 100 == 0:
                    train_obj = obj_k.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                    print("Pre-training %g Epoch %d: error %g" % (k, i + 1, train_obj))

        for k in range(epoch_tot):
            for i in range(num_step):
                train_step.run(feed_dict={keep_prob: 0.5})
                # elif k < 20000:
                #     train_step_1.run(feed_dict={keep_prob: 0.5})
                # else:
                #     train_step_2.run(feed_dict={keep_prob: 0.5})
            if (k+1) % 100 == 0:
                train_obj = obj_oa.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                train_err = reconstruction_error.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                print("Epoch %d: objective %g; error %g" % (k + 1, train_obj, train_err))

        for k in range(epoch_tot):
            train_step_gmm.run(feed_dict={keep_prob: 0.5, train_x_v: train_norm_x})
            # elif k < 20000:
            #     train_step_1.run(feed_dict={keep_prob: 0.5})
            # else:
            #     train_step_2.run(feed_dict={keep_prob: 0.5})
            if (k+1) % 100 == 0:
                train_obj = obj_gmm.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                print("Epoch %d at gmm: objective %g" % (k + 1, train_obj))

        train_out = z_o.eval(feed_dict={test_x_v: test_norm_x, keep_prob: 1.0, y: test_y})
        predict_lh = test_likelihood.eval(feed_dict={train_x_v: train_norm_x, test_x_v: test_norm_x, keep_prob: 1.0})
        precision, recall, f1 = self.accuracy(predict_lh, num_test_points, test_y, 0.15)
        # print train_out
        self.output_code(train_out, num_test_points)
        coord.request_stop()
        coord.join(threads)
        sess.close()
        return precision, recall, f1

