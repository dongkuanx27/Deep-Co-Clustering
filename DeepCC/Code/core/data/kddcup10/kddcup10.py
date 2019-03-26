import numpy as np
import random
import scipy.io as sio
# import gzip as gz


class KddcupDataP10:
    def __init__(self, filename):
        self.num_dim = 0  #120
        self.num_points = 0  #494021
        self.xy = []
        #self.num_assign_points = 0
        self.num_train_points = 0
        #self.num_test_points = 0
        self.train_data = 0
        #self.test_data = 0
        #self.test_label = 0
        self.train_label = 0

        # for columns
        #self.xy_col = []
        #self.num_assign_points_col = 0
        self.num_train_points_col = 0
        #self.num_test_points_col = 0
        self.train_data_col = 0
        #self.test_data_col = 0

        '''
        # fin = gz.open(filename, 'r')
        fin = open(filename, 'r')
        line = fin.readline()
        outlier = 0
        while line != '':
            line = line.strip()
            field = line.split(',')
            p = []
            for i in range(0, len(field)):
                p.append(float(field[i]))
            outlier = outlier + p[len(p) - 1]
            line = fin.readline()
            self.xy.append(p)
        fin.close()
        ratio = outlier / len(self.xy)
        print 'all: ' + str(len(self.xy)) + ', outlier: ' + str(int(outlier)) + ', ratio: ' + str(ratio)
        '''
        self.num_points = sio.loadmat(filename)['fea'].shape[0]
        self.num_dim = sio.loadmat(filename)['fea'].shape[1]
        
        self.xy = np.column_stack((sio.loadmat(filename)['fea'], sio.loadmat(filename)['gnd'] - 1))


    def get_clean_training_testing_data(self, ratio):
        '''
        self.num_assign_points = int(self.num_points * ratio)
        self.num_test_points = self.num_points - self.num_assign_points
        random.shuffle(self.xy)
        self.num_train_points = 0
        for i in range(self.num_assign_points):
            pl = float(self.xy[i][120])
            if pl < 1:
                self.num_train_points = self.num_train_points + 1
        self.train_data = np.zeros([self.num_train_points, 120], dtype=np.float64)
        self.test_data = np.zeros([self.num_test_points, 120], dtype=np.float64)
        self.test_label = np.zeros([self.num_test_points, 1], dtype=np.float64)
        head = 0
        for i in range(self.num_assign_points):
            pl = float(self.xy[i][120])
            if pl < 1:
                for j in range(120):
                    self.train_data[head, j] = float(self.xy[i][j])
                head = head + 1
        for i in range(self.num_test_points):
            ni = self.num_assign_points + i
            for j in range(120):
                self.test_data[i, j] = float(self.xy[ni][j])
            self.test_label[i, 0] = float(self.xy[ni][120])
        return self.train_data, self.test_data, self.test_label, self.num_test_points
        '''

        self.num_train_points = int(self.num_points * ratio)
        #self.num_assign_points = self.num_train_points
        #self.num_test_points = self.num_points - self.num_train_points
        #np.random.shuffle(self.xy)

        self.train_data = self.xy[0:self.num_train_points, 0:self.num_dim]
        self.train_label = self.xy[0:self.num_train_points, self.num_dim]
        #self.test_data = self.xy[self.num_train_points:self.num_points, 0:self.num_dim]
        #self.test_label = self.xy[self.num_train_points:self.num_points, self.num_dim]
        self.train_data_col = self.train_data.T

        print('Data loaded successfully')

        return self.train_data, self.train_label, self.train_data_col

'''
    def get_contaminated_training_testing_data(self, ratio, contamination):
        adjusted_ratio = contamination / (1.0 - contamination)
        self.num_assign_points = int(self.num_points * ratio)
        self.num_test_points = self.num_points - self.num_assign_points
        random.shuffle(self.xy)
        self.num_train_points = 0
        for i in range(self.num_assign_points):
            pl = float(self.xy[i][120])
            if pl < 1:
                self.num_train_points = self.num_train_points + 1
        num_contaminated_points = int(self.num_train_points * adjusted_ratio)
        self.num_train_points = num_contaminated_points + self.num_train_points
        self.train_data = np.zeros([self.num_train_points, 120], dtype=np.float64)
        self.test_data = np.zeros([self.num_test_points, 120], dtype=np.float64)
        self.test_label = np.zeros([self.num_test_points, 1], dtype=np.float64)
        head = 0
        contaminated_cnt = 0
        for i in range(self.num_assign_points):
            pl = float(self.xy[i][120])
            if pl < 1:
                for j in range(120):
                    self.train_data[head, j] = float(self.xy[i][j])
                head = head + 1
            elif contaminated_cnt < num_contaminated_points:
                for j in range(120):
                    self.train_data[head, j] = float(self.xy[i][j])
                head = head + 1
                contaminated_cnt = contaminated_cnt + 1
        print contaminated_cnt, num_contaminated_points, head, self.num_train_points
        for i in range(self.num_test_points):
            ni = self.num_assign_points + i
            for j in range(120):
                self.test_data[i, j] = float(self.xy[ni][j])
            self.test_label[i, 0] = float(self.xy[ni][120])
        return self.train_data, self.test_data, self.test_label, self.num_test_points
'''

if __name__ == '__main__':
    kddcup10 = KddcupDataP10('/home/dsi/dxu/Backups/Research_Server/Working/Deep_co_clustering/Data/coil20.mat')
    # train_data, test_data, test_label, num_test_points, num_dirty_points = kddcup10.get_clean_training_testing_data(0.5)
    # print np.shape(train_data), np.shape(test_data), num_test_points, num_dirty_points
