from __future__ import division
import numpy as np
import tensorflow as tf
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import scipy.io as sio

import core.data.kddcup10.kddcup10 as kddcup
import core.general.gmm_estimation_net as dgmmb
import core.general.gmm_estimation_net_raw as dgmmb_multi
import core.general.pretrain_autoencoder as ae


'''
DMM for kddcup10, using batch training
'''


def get_key(item):
    return item[0]


class KddcupPaeGmm:
    def __init__(self, num_input_dim, nclu_row, nclu_col, ae_config, ae_col_config, gmm_config, num_dropout):
        self.num_input_dim = num_input_dim
        self.autoencoder = ae.PretrainAutoencoder(ae_config, num_dropout)
        self.autoencoder_col = ae.PretrainAutoencoder(ae_col_config, num_dropout)
        #self.e_net = dgmmb.GMMEstimationNet(gmm_config)
        self.e_net = dgmmb_multi.GMMEstimationNetRaw(gmm_config)
        self.e_net_col = dgmmb_multi.GMMEstimationNetRaw(gmm_config)

        self.nclu_row = nclu_row # the cluster num of rows
        self.nclu_col = nclu_col

    @staticmethod
    def get_data(input_file):
        data = kddcup.KddcupDataP10(input_file)
        data.get_clean_training_testing_data(1) # the ratio of training vs whole
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
        fout = open('../result/kddcup_output_batch_1.csv', 'w')
        for i in range(num_test_points):
            fout.write(str(train_out[i, 0]) + ',' + str(train_out[i, 1]) + ',' + str(train_out[i, 2]) + ','
                       + str(train_out[i, 3]) + '\n')
            # fout.write(str(train_out[i, 0]) + ',' + str(train_out[i, 1]) + ',' + str(train_out[i, 2]) + '\n')
        fout.close()

    def eval(self, tru, pre):
        # true label: numpy, vector in col
        # pred lable: numpy, vector in row

        num_labels = tru.shape[0]
        # nmi
        nmi = normalized_mutual_info_score(tru.reshape(num_labels), pre)

        # accuracy
        tru = np.reshape(tru, (num_labels))
        #set_tru = set(tru.tolist())
        set_pre = set(pre.tolist())
        #nclu_tru = len(set_tru) # in case that nclu_tru != the preset cluster num
        nclu_pre = len(set_pre)
        
        correct = 0
        for i in range(nclu_pre):
            flag = list(set_pre)[i]
            idx = np.argwhere(pre == flag)
            correct += max(np.bincount(tru[idx].T[0].astype(np.int64)))
        acc = correct / num_labels

        return acc, nmi

    def Reduce_Table(self, T, V1, V2):
        # T: a table (instances of the same cluster are adjacent)
        # V1: the cluster assignment vector for rows
        # V2: the cluster assignment vector for cols
        # return the reduced table

        # first reduce by rows 
        for i in range(self.nclu_row):
            idx = tf.where(tf.equal(V1, i))
            if i == 0:
                T_r_data = tf.gather(T, tf.transpose(idx))[0]
                T_r = tf.reduce_sum(T_r_data, 0)
                T_r = tf.reshape(T_r, (1, V2.get_shape().as_list()[0]))
            else:
                T_r_data = tf.gather(T, tf.transpose(idx))[0]
                temp = tf.reshape(tf.reduce_sum(T_r_data, 0),(1, V2.get_shape().as_list()[0]))
                T_r = tf.concat([T_r, temp],0)

        # second reduce by cols
        for i in range(self.nclu_col):
            idx = tf.where(tf.equal(V2, i))
            if i == 0:
                T_rr_data = tf.transpose(tf.gather(tf.transpose(T_r), tf.transpose(idx))[0])
                T_rr = tf.reduce_sum(T_rr_data, 1)
                T_rr = tf.reshape(T_rr, (self.nclu_row, 1))
            else:
                T_rr_data = tf.transpose(tf.gather(tf.transpose(T_r), tf.transpose(idx))[0])
                temp = tf.reshape(tf.reduce_sum(T_rr_data, 1), (self.nclu_row, 1))
                T_rr = tf.concat([T_rr, temp], 1)    

        return T_rr

    def MI_Table(self, T):
        # T: a probablity table; numpy array
        # return the mutual information between rows and cols of table T

        P_x, P_y = tf.reduce_sum(T, 1), tf.reduce_sum(T, 0)
        nx, ny = tf.shape(T)[0], tf.shape(T)[1]
        T_xy = tf.matmul(tf.reshape(P_x, (nx,1)), tf.reshape(P_y, (1, ny)))

        MI_temp = tf.log((T+0.1**15)/(T_xy+0.1**15)) / tf.log(tf.constant(2, tf.float64))
        MI_T = tf.reduce_sum(tf.multiply(T, MI_temp))

        return MI_T


    def MI_loss(self, Ur, Uc):
        # Ur: the cluster assignment matrix for rows, N_ins * N_Row_clus
        # Uc: the cluster assignment matrix for cols, N_att * N_Col_clus
        # return the loss of mutual information between the original data matrix and the reduced data matrix

        N_ins, N_Row_clus = Ur.get_shape().as_list()
        N_att, N_Col_clus = Uc.get_shape().as_list()
        #T_pro_org = tf.Variable(tf.zeros((N_ins, N_att))) # original table for the joint probability
        #T_pro_red = tf.Variable(tf.zeros((N_Row_clus, N_Col_clus))) # reduced table for the joint probability

        Ur_max = tf.reshape(tf.reduce_max(Ur, 1), (N_ins, 1)) # the max value for each row; column vector
        Uc_max = tf.reshape(tf.reduce_max(Uc, 1), (N_att, 1))

        Ur_max_idx = tf.argmax(Ur, 1) # the index of max value for rows, vector in row
        Uc_max_idx = tf.argmax(Uc, 1)

        T_pro_org = tf.matmul(Ur_max, tf.transpose(Uc_max)) # original table for the joint probability
        T_pro_org = T_pro_org / tf.reduce_sum(T_pro_org) # normalization: make sum equal 1

        T_pro_red = self.Reduce_Table(T_pro_org, Ur_max_idx, Uc_max_idx) # reduced table
        
        # calculate MI of orginal and reduced tables
        MI_org = self.MI_Table(T_pro_org)
        MI_red = self.MI_Table(T_pro_red)
        
        #loss = tf.abs((1 - MI_red/MI_org)*(MI_org - MI_red)) # alternative calculation way
        loss = tf.abs(1 - MI_red/MI_org)
        loss = tf.log(1+loss)


        self.T_pro_org = T_pro_org
        self.sum_T_pro_org = tf.reduce_sum(T_pro_org)
        self.T_pro_red = T_pro_red
        self.sum_T_pro_red = tf.reduce_sum(T_pro_red)
        self.Ur = Ur
        self.Uc = Uc
        self.Ur_max = Ur_max
        self.Uc_max = Uc_max
        self.Ur_max_idx = Ur_max_idx
        self.Uc_max_idx = Uc_max_idx
        self.Ur_max_idx_sum = tf.reduce_sum(Ur_max_idx)
        self.Uc_max_idx_sum = tf.reduce_sum(Uc_max_idx)
        self.MI_org = MI_org
        self.MI_red = MI_red

        return loss

    def run(self, data, train_epochs):
        sess = tf.InteractiveSession()
        # Data
        train_x = data.train_data
        train_y = data.train_label 
        train_x_col = data.train_data_col

        # train_norm_x = self.minmax_normalization(train_x, train_x)
        # train_norm_x_col = self.minmax_normalization(train_x_col, train_x_col)

        train_norm_x = self.gaussian_normalization(train_x)
        train_norm_x_col = self.gaussian_normalization(train_x_col)

        # Setup
        train_x_v = tf.placeholder(dtype=tf.float64, shape=[train_norm_x.shape[0], train_norm_x.shape[1]])
        train_x_v_col = tf.placeholder(dtype=tf.float64, shape=[train_norm_x_col.shape[0], train_norm_x_col.shape[1]])
        keep_prob = tf.placeholder(tf.float64)
        
        # Autoencoder
        train_z, train_error, train_var_list, train_l2_reg, train_reg = self.autoencoder.run(train_x_v, keep_prob)
        train_z_col, train_error_col, train_var_list_col, train_l2_reg_col, train_reg_col = self.autoencoder_col.run(train_x_v_col, keep_prob)

        '''
        z_b, error_b, var_list_b, l2_reg_b, reg_b = self.autoencoder.run(x_b, keep_prob)
        train_z, train_error, train_var_list, train_l2_reg, train_reg = self.autoencoder.run(train_x_v, keep_prob)
        '''

        # Pretraining
        pretrain_step = []
        pretrain_obj = []
        for i in range(len(train_var_list)):
            obj_oa_pretrain = train_error[i] * 1e1 + train_reg[i] * 1e-2
            train_step_i = tf.train.AdamOptimizer(1e-4).minimize(obj_oa_pretrain, var_list=train_var_list[i])
            pretrain_step.append(train_step_i)
            pretrain_obj.append(obj_oa_pretrain)

        pretrain_col_step = []
        pretrain_col_obj = []
        for i in range(len(train_var_list_col)):
            obj_oa_col_pretrain = train_error_col[i] * 1e1 + train_reg_col[i] * 1e-2
            train_col_step_i = tf.train.AdamOptimizer(1e-4).minimize(obj_oa_col_pretrain, var_list=train_var_list_col[i])
            pretrain_col_step.append(train_col_step_i)
            pretrain_col_obj.append(obj_oa_col_pretrain)

        # Joint fine training
        error_oa = 0
        for error_k in train_error:
            error_oa = error_oa + error_k
        # error_oa = train_error[len(train_error) - 1]
        # reconstruction_error = train_error[len(train_error) - 1]

        error_oa_col = 0
        for error_col_k in train_error_col:
            error_oa_col = error_oa_col + error_col_k

        # GMM Membership estimation
        # loss, pen_dev, likelihood = self.e_net.run(train_z, keep_prob)
        loss, pen_dev, likelihood, p_z, x_t, p_t, z_p, z_t, mixture_mean, mixture_dev, mixture_cov, mixture_dev_det = self.e_net.run(train_z, keep_prob)

        loss_col, pen_dev_col, likelihood_col, p_z_col, x_t_col, p_t_col, z_p_col, z_t_col, mixture_mean_col, mixture_dev_col, mixture_cov_col, mixture_dev_det_col = self.e_net_col.run(train_z_col, keep_prob)

        # Train step
        # obj_oa = error_oa + loss * 0.1 + 0.1 * reg
        # obj_oa = error_oa + 0.1 * train_l2_reg
        # obj_oa = error_oa * 1e1 + train_l2_reg * 1e-2 + loss * 1e1 + pen_dev * 1e-2
        obj_oa = error_oa * 1e1 + train_l2_reg * 1e-2 + loss * 1e1 + pen_dev * 1e-2 + \
                error_oa_col * 1e1 + train_l2_reg_col * 1e-2 + loss_col * 1e1 + pen_dev_col * 1e-2
        
        train_step = tf.train.AdamOptimizer(1e-4).minimize(obj_oa)
        # train_step_1 = tf.train.AdamOptimizer(1e-4).minimize(obj)
        # train_step_2 = tf.train.AdamOptimizer(1e-5).minimize(obj)
        # train_step_3 = tf.train.AdamOptimizer(1e-5).minimize(obj)
        
        # GMM training
        # obj_gmm = loss + pen_dev * 0.05

        # train_step_gmm = tf.train.AdamOptimizer(1e-4).minimize(obj_gmm_b, var_list=self.e_net.var_list)

        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        RR_obj_oa = []
        RR_error_oa = []
        RR_train_l2_reg = []
        RR_loss = []
        RR_pen_dev = []

        RR_acc = []
        RR_nmi = []
        RR_label_size = []

        # pretrain for AE of row
        epoch_tot = train_epochs
        # num_step = num_train_points / batch_size + 1
        for k in range(len(pretrain_step)):
            train_step_pre_k = pretrain_step[k]
            obj_k = pretrain_obj[k]
            for i in range(epoch_tot):
                train_step_pre_k.run(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                if (i + 1) % 1 == 0:
                    train_obj = obj_k.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                    print("Pre-training row %g Epoch %d: error %g" % (k, i + 1, train_obj))

        # pretrain for AE of col
        epoch_tot_col = train_epochs
        for k in range(len(pretrain_col_step)):
            train_step_pre_col_k = pretrain_col_step[k]
            obj_col_k = pretrain_col_obj[k]
            for i in range(epoch_tot_col):
                train_step_pre_col_k.run(feed_dict={keep_prob: 1.0, train_x_v_col: train_norm_x_col})
                if (i + 1) % 1 == 0:
                    train_col_obj = obj_col_k.eval(feed_dict={keep_prob: 1.0, train_x_v_col: train_norm_x_col})
                    print("Pre-training col %g Epoch %d: error %g" % (k, i + 1, train_col_obj))

        for k in range(epoch_tot):
            train_step.run(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x, train_x_v_col: train_norm_x_col})
            # elif k < 20000:
            #     train_step_1.run(feed_dict={keep_prob: 0.5})
            # else:
            #     train_step_2.run(feed_dict={keep_prob: 0.5})
            '''
            if (k+1) % 10 == 0:
                train_obj = obj_oa.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                train_err = reconstruction_error.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                print("Epoch %d: objective %g; error %g" % (k + 1, train_obj, train_err))
            '''
            fetch = {'obj_oa': obj_oa, \
                        'error_oa':error_oa, 'train_l2_reg':train_l2_reg, 'loss':loss, 'pen_dev':pen_dev, \
                        'error_oa_col':error_oa_col, 'train_l2_reg_col':train_l2_reg_col, 'loss_col':loss_col, 'pen_dev_col':pen_dev_col, \
                        'train_z':train_z, 'p_z':p_z, 'train_z_col':train_z_col, 'p_z_col':p_z_col,\
                        'mixture_dev':mixture_dev, 'mixture_cov':mixture_cov, 'mixture_dev_det':mixture_dev_det, \
                        'mixture_dev_col':mixture_dev_col, 'mixture_cov_col':mixture_cov_col, 'mixture_dev_det_col':mixture_dev_det_col, \
                        'x_t':x_t, 'p_t':p_t, 'z_p':z_p, 'z_t':z_t, \
                        'x_t_col':x_t_col, 'p_t_col':p_t_col, 'z_p_col':z_p_col, 'z_t_col':z_t_col}
            
            RR = sess.run(fetch,feed_dict={keep_prob: 1.0, train_x_v: train_norm_x, train_x_v_col: train_norm_x_col})

            '''
            print 'AE output'
            print(RR['train_z'])
            '''
            # print 'p_z '
            # print(RR['p_z'])
            '''
            print 'x_t'
            print(x_t)
            print(RR['x_t'][:,0,:])
            print 'p_t'
            print(p_t)
            print 'z_p'
            print(z_p)
            print 'z_t'
            print(z_t)

            print'********'

            print 'mixture_mean'
            print(mixture_mean)
            print 'mixture_cov'
            print(RR['mixture_cov'].shape)
            print 'mixture_dev'
            print(RR['mixture_dev'].shape)
            print 'mixture_dev_det'
            print(RR['mixture_dev_det'].shape)
            '''

            # print 'inverse of cov'
            # print (np.linalg.inv(RR['mixture_cov']))

            # row
            print("Epoch %d: obj_oa %g; error_oa %g; train_l2_reg %g; loss %g; pen_dev %g;"
                  % (k + 1, RR['obj_oa'], RR['error_oa'], RR['train_l2_reg'], RR['loss'], RR['pen_dev']))
            # col
            print("Epoch %d: obj_oa_col %g; error_oa_col %g; train_l2_reg_col %g; loss_col %g; pen_dev_col %g;"
                  % (k + 1, RR['obj_oa'], RR['error_oa_col'], RR['train_l2_reg_col'], RR['loss_col'], RR['pen_dev_col']))

            # calculate accuracy and NMI
            pred_label = np.argmax(RR['p_z'], 1) # vertor in row
            #pred_label = KMeans(n_clusters=20, random_state=0).fit(RR['train_z']).labels_
            set_pre = set(pred_label.tolist())
            print 'Predicted label set'
            print set_pre
            true_label = train_y # numpy
            acc, NMI = self.eval(true_label, pred_label)
            print 'acc:' + str(acc) + ',' + 'NMI:' + str(NMI)

            RR_obj_oa = np.append(RR_obj_oa, RR['obj_oa'])
            RR_error_oa = np.append(RR_error_oa, RR['error_oa'])
            RR_train_l2_reg = np.append(RR_train_l2_reg, RR['train_l2_reg'])
            RR_loss = np.append(RR_loss, RR['loss'])
            RR_pen_dev = np.append(RR_pen_dev, RR['pen_dev'])
            
            RR_acc = np.append(RR_acc, acc)
            RR_nmi = np.append(RR_nmi, NMI)
            RR_label_size = np.append(RR_label_size, len(set_pre))

        '''
        # GMM training
        for k in range(2000):
            train_step_gmm.run(feed_dict={keep_prob: 0.5})
            # elif k < 20000:
            #     train_step_1.run(feed_dict={keep_prob: 0.5})
            # else:
            #     train_step_2.run(feed_dict={keep_prob: 0.5})
            if (k+1) % 10 == 0:
                train_obj = obj_gmm.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                print("Epoch %d at gmm: objective %g" % (k + 1, train_obj))
        '''

        coord.request_stop()
        coord.join(threads)

        # calculate accuracy and NMI
        #pred_label = np.argmax(RR['p_z'], 1) # vertor in row
        pred_label = KMeans(n_clusters=20, random_state=0).fit(RR['train_z']).labels_
        set_pre = set(pred_label.tolist())
        print 'Predicted label'
        print pred_label
        print 'Predicted label set'
        print set_pre
        true_label = train_y # numpy
        acc, NMI = self.eval(true_label, pred_label)

        sio.savemat('/home/dux19/Research/Working/Deep_co_clustering/DCC_with_KL_divergence_loss_pretrain/Coil20/RR_result.mat', {'RR_obj_oa':RR_obj_oa, 'RR_error_oa':RR_error_oa, 'RR_train_l2_reg':RR_train_l2_reg, \
            'RR_loss':RR_loss, 'RR_pen_dev':RR_pen_dev, 'RR_acc':RR_acc, 'RR_nmi':RR_nmi, 'RR_label_size':RR_label_size})

        sess.close()

        return acc, NMI

