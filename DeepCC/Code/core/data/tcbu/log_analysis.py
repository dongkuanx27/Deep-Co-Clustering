import requests as rq
import json
from datetime import datetime


def search(uri):
    query = json.dumps({
        "from": 0,
        "size": 10000,
        "query": {
            "match_all": {
            }
        }
    })
    resp = rq.get(uri, data=query)
    res = resp.json()
    return res


def process(input_data, output_filename):
    fout = open(output_filename, 'wt')
    epoch = datetime.utcfromtimestamp(0)
    output_data = {}
    for obj in input_data:
        info = obj['_source']
        ts = info['@timestamp']
        if len(ts.split('-')[0]) == 4:
            ts = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')
            ts = int((ts - epoch).total_seconds() * 1000)
            anomaly_type = info['anomalyType']
            if ts not in output_data.keys():
                output_data[ts] = {1: 0, 5: 0, 9: 0}
            output_data[ts][anomaly_type] = output_data[ts][anomaly_type] + 1
        else:
            print ts
    ts_list = output_data.keys()
    ts_list.sort()
    fout.write('Time,T1,T5,T9,Total\n')
    key_list = [1, 5, 9]
    for key in ts_list:
        fout.write(str(key))
        total = 0
        for t in key_list:
            fout.write(',' + str(output_data[key][t]))
            total = total + output_data[key][t]
        fout.write(',' + str(total) + '\n')
    fout.close()


if __name__ == '__main__':
    uri = ['http://138.15.170.105:9200/alerts-spoof/TCBU.10.30.test/_search',
           'http://138.15.170.105:9200/alerts-spoof/TCBU.30.30.test/_search',
           'http://138.15.170.105:9200/alerts-spoof/TCBU.50.30.test/_search',
           'http://138.15.170.105:9200/alerts-spoof/TCBU.70.30.test/_search']
    filename = ['./dd/10-30.csv',
                './dd/30-30.csv',
                './dd/50-30.csv',
                './dd/70-30.csv']
    for i in range(len(uri)):
        print uri[i]
        results = search(uri[i])
        data = results['hits']['hits']
        process(data, filename[i])
