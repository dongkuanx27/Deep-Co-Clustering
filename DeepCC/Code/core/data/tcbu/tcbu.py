import numpy as np


class TcbuData:
    def __init__(self, filename):
        self.num_dim = 6
        self.data = {}
        fin = open(filename, 'r')
        line = fin.readline()
        field = line.split(',')
        field_name = []
        for i in range(1, len(field)-1):
            field_name.append(field[i])
        self.data['feature_name'] = field_name
        x = np.zeros([180, 6])
        y = np.zeros([180, 1])
        ts = []
        head = 0
        line = fin.readline()
        while line != '':
            field = line.split(',')
            ts.append(field[0])
            # print field[0], field[1]
            for i in range(1, len(field)-1):
                x[head, i-1] = float(field[i])
            y[head, 0] = float(field[len(field)-1])
            line = fin.readline()
            head = head + 1
        fin.close()
        self.data['time'] = ts
        self.data['feature'] = x
        self.data['label'] = y
        # print x

if __name__ == '__main__':
    TcbuData('../../dataset/tcbu_spoof/attack.2.feature.csv')