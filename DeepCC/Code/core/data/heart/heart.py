import numpy as np
import random


class HeartData:
    def __init__(self, filename):
        self.num_dim = 274
        self.num_points = 452
        self.xy = []
        self.num_train_points = 0
        self.num_assign_points = 0
        self.num_test_points = 0
        self.train_data = 0
        self.test_data = 0
        self.test_label = 0

        fin = open(filename, 'r')
        line = fin.readline()
        outlier = 0
        label_dict = {}
        while line != '':
            line = line.strip()
            field = line.split(',')
            p = []
            for i in range(0, len(field)):
                p.append(float(field[i]))
            label = field[len(field)-1]
            if label not in label_dict.keys():
                label_dict[label] = 0
            label_dict[label] = label_dict[label] + 1
            if p[len(p) - 1] > 0:
                outlier = outlier + 1.0
            line = fin.readline()
            self.xy.append(p)
        fin.close()
        ratio = outlier / len(self.xy)
        print 'all: ' + str(len(self.xy)) + ', outlier: ' + str(int(outlier)) + ', ratio: ' + str(ratio)
        print label_dict
        print '# dimensions: ' + str(len(self.xy[0]) - 1)

    def get_clean_training_testing_data(self, ratio):
        self.num_assign_points = int(self.num_points * ratio)
        self.num_test_points = self.num_points - self.num_assign_points
        random.shuffle(self.xy)
        self.num_train_points = 0
        for i in range(self.num_assign_points):
            pl = float(self.xy[i][274])
            if pl < 1:
                self.num_train_points = self.num_train_points + 1
        self.train_data = np.zeros([self.num_train_points, 274], dtype=np.float64)
        self.test_data = np.zeros([self.num_test_points, 274], dtype=np.float64)
        self.test_label = np.zeros([self.num_test_points, 1], dtype=np.float64)
        head = 0
        for i in range(self.num_assign_points):
            pl = float(self.xy[i][274])
            if pl < 1:
                for j in range(274):
                    self.train_data[head, j] = float(self.xy[i][j])
                head = head + 1
        for i in range(self.num_test_points):
            ni = self.num_assign_points + i
            for j in range(274):
                self.test_data[i, j] = float(self.xy[ni][j])
            self.test_label[i, 0] = float(self.xy[ni][274])
        print self.num_test_points, self.num_train_points
        return self.train_data, self.test_data, self.test_label, self.num_test_points

if __name__ == '__main__':
    heart = HeartData('../../../../dataset/arrhythmia/heart.pp.csv')
    # usenet.tfidf_process()
    # train_data, test_data, test_label, num_test_points, num_dirty_points = kddcup10.get_clean_training_testing_data(0.5)
    # print np.shape(train_data), np.shape(test_data), num_test_points, num_dirty_points
