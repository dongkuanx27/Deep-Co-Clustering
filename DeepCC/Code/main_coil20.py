from __future__ import division
import os
import tensorflow as tf
import core.paegmm.kddcup10.kddcup10_pae_gmm as paegmm
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

if __name__ == '__main__':
    
    # data_path
    Coil20 = ['.../Data/coil20.mat', 20, 'coil20']
    
    filename   = Coil20[0]
    num_clus_r = Coil20[1]
    num_clus_c = Coil20[1]

    ae_config = [1024, 500, 200, 100, 40]

    ae_col_config = [1440, 500, 200, 100, 40]

    gmm_config = [[num_clus_r, 5], 40, 160, 80, 40, num_clus_r]

    accuracy     = []
    NMI          = []

    rounds = 1
    epochs = 2
    epochs_pretrain = 2
    for k in range(rounds):
        tf.reset_default_graph()
        machine = paegmm.KddcupPaeGmm(1024, num_clus_r, num_clus_c, ae_config, ae_col_config, gmm_config, 0)
        data = machine.get_data(filename)
        acc, nmi = machine.run(data, epochs, epochs_pretrain)

        accuracy      = np.append(accuracy, acc)
        NMI           = np.append(NMI, nmi)
