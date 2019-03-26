from datetime import datetime


def parse_raw_into_csv(file_name):
    epoch = datetime.utcfromtimestamp(0)
    fin = open(file_name, 'r')
    output = file_name + '.csv'
    fout = open(output, 'wt')
    fout.write('Time,Type,GT,MSIS,IMSI\n')
    line = fin.readline()
    key_name = ['Type', 'GT', 'MSIS', 'IMSI']
    while line != '':
        line = line.rstrip()
        field = line.split(' ')
        ts = field[0] + ' ' + field[1]
        ts = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f')
        ts = int((ts - epoch).total_seconds() * 1000)
        oline = str(ts)
        feature = {}
        for i in range(2, len(field)):
            kv = field[i].split('=')
            if kv[0] not in key_name:
                print kv[0]
                print 'Unexpected feature!'
            feature[kv[0]] = kv[1].rstrip()
        for key in key_name:
            if key in feature.keys():
                oline = oline + ',' + feature[key]
            else:
                oline = oline + ',None'
        fout.write(oline + '\n')
        line = fin.readline()
    fin.close()
    fout.close()


if __name__ == '__main__':
    file_name = '../dd/NoisyData.txt'
    parse_raw_into_csv(file_name)
