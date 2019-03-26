def preprocess():
    fin = open('../../../../dataset/thyroid/thyroid.x.csv', 'r')
    line = fin.readline()
    data = []
    while line != '':
        line = line.strip()
        field = line.split(',')
        p = []
        for i in range(0, len(field)):
            p.append(float(field[i]))
        data.append(p)
        line = fin.readline()
    fin.close()
    fin = open('../../../../dataset/thyroid/thyroid.y.csv', 'r')
    line = fin.readline()
    head = 0
    while line != '':
        line = line.strip()
        label = float(line)
        data[head].append(label)
        head = head + 1
        line = fin.readline()
    fin.close()
    label_dict = {}
    outlier = 0
    fout = open('../../../../dataset/thyroid/thyroid.pp.csv', 'w')
    num_dim = len(data[0]) - 1
    for i in range(len(data)):
        fout.write(str(data[i][0]))
        for j in range(1, len(data[i])):
            fout.write(',' + str(data[i][j]))
        label = data[i][num_dim]
        outlier = outlier + label
        if label not in label_dict.keys():
            label_dict[label] = 0
        label_dict[label] = label_dict[label] + 1
        fout.write('\n')
    fout.close()
    lcnt = len(data)
    ratio = outlier / lcnt
    print 'all: ' + str(lcnt) + ', outlier: ' + str(int(outlier)) + ', ratio: ' + str(ratio)
    print label_dict
    print num_dim


if __name__ == '__main__':
    preprocess()
