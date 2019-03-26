import numpy as np
# import gzip as gz


class KddcupData:
    def __init__(self, filename):
        self.num_dim = 120
        self.num_points = 985940
        self.data = {}
        x = np.zeros([self.num_points, self.num_dim], dtype=np.float32)
        y = np.zeros([self.num_points, 1], dtype=np.float32)

        # fin = gz.open(filename, 'r')
        fin = open(filename, 'r')
        head = 0
        line = fin.readline()
        while line != '':
            line = line.strip()
            field = line.split(',')
            for i in range(0, len(field)-1):
                x[head, i] = float(field[i])
            y[head, 0] = float(field[len(field)-1])
            line = fin.readline()
            head = head + 1
        fin.close()
        self.data['feature'] = x
        self.data['label'] = y

if __name__ == '__main__':
    kddcup = KddcupData('../../../dataset/kddcup99/kddcup10.data.pp.csv')
    print np.shape(kddcup.data['feature'])
    print np.shape(kddcup.data['label'])
