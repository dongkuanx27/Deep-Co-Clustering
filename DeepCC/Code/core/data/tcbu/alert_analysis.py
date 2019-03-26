import json
from datetime import datetime


def type9_stats():
    input_file = ['./result/type9-10.30',
                  './result/type9-30.30',
                  './result/type9-50.30',
                  './result/type9-70.30']
    for filename in input_file:
        output_file = filename + '.stat.csv'
        fin = open(filename, 'rt')
        fout = open(output_file, 'wt')
        anomaly = {}
        init_epoch = datetime.utcfromtimestamp(0)
        line = fin.readline()
        while line != '':
            ins = json.loads(line)
            ts = ins['@timestamp']
            ts = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')
            ts = int((ts - init_epoch).total_seconds() * 1000)
            # ts = ts - ts % 500
            if ts not in anomaly.keys():
                anomaly[ts] = 0
            anomaly[ts] = anomaly[ts] + 1
            line = fin.readline()
        fin.close()
        ts_list = anomaly.keys()
        ts_list.sort()
        for ts in ts_list:
            fout.write(str(datetime.utcfromtimestamp(ts/1000.0)) + ',' + str(ts) + ',' + str(anomaly[ts]) + '\n')
        fout.close()


def type8_stats():
    input_file = ['./result/tcbu_10.30_type8',
                  './result/tcbu_30.30_type8',
                  ]

if __name__ == '__main__':
    type9_stats()
