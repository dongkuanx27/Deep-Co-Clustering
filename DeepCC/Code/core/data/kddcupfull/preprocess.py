import numpy as np


def preprocess():
    filename = '../../../dataset/kddcup99/kddcup10.data.raw.csv'
    fin = open(filename, 'r')
    line = fin.readline()
    protocol_type = {}
    service_type = {}
    flag_type = {}
    label = {'normal': 0, 'attack': 0}
    cnt = 0
    prev = -1
    data = np.zeros([985940, 121])
    shift1 = 1
    shift2 = 1 + 3
    shift3 = 1 + 3 + 68
    shift4 = 1 + 3 + 68 + 11
    while line != '':
        line = line.strip()
        field = line.split(',')
        if field[2] != 'ecr_i' and field[2] != 'private':
            data[cnt, 0] = float(field[0])

            if prev == -1:
                prev = len(field)
            elif prev != len(field):
                print 'error'

            key1 = field[1]
            if key1 not in protocol_type.keys():
                idx = len(protocol_type)
                protocol_type[key1] = idx

            data[cnt, shift1+protocol_type[key1]] = 1.0
            # protocol_type[key1] = protocol_type[key1] + 1

            key2 = field[2]
            if key2 not in service_type.keys():
                idx = len(service_type)
                service_type[key2] = idx
            # service_type[key2] = service_type[key2] + 1
            data[cnt, shift2+service_type[key2]] = 1.0

            key3 = field[3]
            if key3 not in flag_type.keys():
                idx = len(flag_type)
                flag_type[key3] = idx
            # flag_type[key3] = flag_type[key3] + 1
            data[cnt, shift3+flag_type[key3]] = 1.0

            for i in range(4, len(field)-1):
                j = i - 4
                data[cnt, shift4+j] = float(field[i])

            key4 = field[prev - 1]
            if key4 == 'normal.':
                label['normal'] = label['normal'] + 1
                data[cnt, 120] = 0.0
            else:
                label['attack'] = label['attack'] + 1
                data[cnt, 120] = 1.0
            cnt = cnt + 1
        line = fin.readline()
    fin.close()
    dim = prev - 3 + len(protocol_type) + len(service_type) + len(flag_type)
    print cnt, dim
    print len(protocol_type), protocol_type
    print len(service_type), service_type
    print len(flag_type), flag_type
    print len(label), label
    np.savetxt('../../../dataset/kddcup99/kddcup10.data.pp.csv.gz', data, delimiter=',', fmt='%.2f')


def load_data():
    fin = open('../../../dataset/kddcup99/kddcup10.data.pp.csv', 'r')
    data = np.zeros([985940, 121])
    line = fin.readline()
    cnt = 0
    while line != '':
        line = line.strip()
        field = line.split(',')
        for i in range(len(field)):
            data[cnt, i] = float(field[i])
        cnt = cnt + 1
        line = fin.readline()
    attack = data[:, 120]
    attack_num = np.sum(attack)
    print attack_num, float(attack_num) / 985940


if __name__ == '__main__':
    # preprocess()
    load_data()
