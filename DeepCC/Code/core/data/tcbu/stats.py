import pandas as pd


def simple_stat(file_name, tr):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.read_csv(file_name)
    print 'Time coverage: ' + str(df['Time'].max() - df['Time'].min())
    sample = {}
    for x in df.values:
        key = x[0] - x[0] % tr
        tp = x[1]
        if key not in sample.keys():
            sample[key] = {'InvokeSendAuthenticationInfo':0,
                           'InvokeUpdateLocation': 0,
                           'InvokeInsertSubscriberData': 0,
                           'InvokeUpdateGprsLocation': 0,
                           'InvokeProvideSubscriberInfo': 0,
                           'InvokeProvideRoamingNumber': 0,
                           'InvokeMoForwardSM': 0,
                           'InvokeSendRoutingInfoForSM': 0,
                           'InvokeMtForwardSM': 0,
                           'InvokeReset': 0,
                           'InvokeAlertServiceCentre': 0,
                           'InvokePurgeMs': 0,
                           'InvokeCancelLocation': 0}
        sample[key][tp] = sample[key][tp] + 1
    feature_list = ['InvokeSendAuthenticationInfo', 'InvokeUpdateLocation', 'InvokeInsertSubscriberData',
                    'InvokeUpdateGprsLocation', 'InvokeProvideSubscriberInfo', 'InvokeProvideRoamingNumber',
                    'InvokeMoForwardSM', 'InvokeSendRoutingInfoForSM', 'InvokeMtForwardSM',
                    'InvokeReset', 'InvokeAlertServiceCentre', 'InvokePurgeMs', 'InvokeCancelLocation']
    output = file_name + '-' + str(tr) + '.feature.csv'
    fout = open(output, 'wt')
    oline = feature_list[0]
    for i in range(1, len(feature_list)):
        oline = oline + ',' + feature_list[i]
    fout.write(oline + '\n')
    for key in sample.keys():
        oline = str(sample[key][feature_list[0]])
        for i in range(1, len(feature_list)):
            oline = oline + ',' + str(sample[key][feature_list[i]])
        fout.write(oline + '\n')
    fout.close()


def to_tseq(file_name):
    fin = open(file_name, 'r')
    fin.readline()
    feature_list = ['InvokeSendAuthenticationInfo', 'InvokeUpdateLocation', 'InvokeInsertSubscriberData',
                    'InvokeUpdateGprsLocation', 'InvokeProvideSubscriberInfo', 'InvokeProvideRoamingNumber',
                    'InvokeMoForwardSM', 'InvokeSendRoutingInfoForSM', 'InvokeMtForwardSM',
                    'InvokeReset', 'InvokeAlertServiceCentre', 'InvokePurgeMs', 'InvokeCancelLocation',
                    'delta_t']
    output = file_name + '-tseq.feature.csv'
    fout = open(output, 'wt')
    oline = feature_list[0]
    for i in range(1, len(feature_list)):
        oline = oline + ',' + feature_list[i]
    fout.write(oline + '\n')
    v = dict(InvokeSendAuthenticationInfo=0, InvokeUpdateLocation=0, InvokeInsertSubscriberData=0,
             InvokeUpdateGprsLocation=0, InvokeProvideSubscriberInfo=0, InvokeProvideRoamingNumber=0,
             InvokeMoForwardSM=0, InvokeSendRoutingInfoForSM=0, InvokeMtForwardSM=0, InvokeReset=0,
             InvokeAlertServiceCentre=0, InvokePurgeMs=0, InvokeCancelLocation=0, delta_t=0)
    line = fin.readline()
    i = 0
    prev = -1
    while line != '':
        x = line.split(',')
        ts = int(x[0])
        tp = x[1]
        v[tp] = 1
        if prev > 0:
            dt = ts-prev
            v['delta_t'] = dt
        oline = str(v[feature_list[0]])
        for j in range(1, len(feature_list)):
            oline = oline + ',' + str(v[feature_list[j]])
        fout.write(oline + '\n')
        v[tp] = 0
        v['delta_t'] = 0
        i = i+1
        print i
        prev = ts
        line = fin.readline()
    fout.close()
    fin.close()


def attack_pattern_1(file_name, tr):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    fin = open(file_name, 'rt')
    feature_list = ['InvokeSendAuthenticationInfo', 'InvokeUpdateLocation', 'InvokeUpdateGprsLocation']
    output = 'ap-1-' + str(tr) + '.feature.csv'
    fout = open(output, 'wt')
    oline = feature_list[0]
    for i in range(1, len(feature_list)):
        oline = oline + ',' + feature_list[i]
    fout.write(oline + '\n')
    sample = {'InvokeSendAuthenticationInfo': 0,
              'InvokeUpdateLocation': 0,
              'InvokeUpdateGprsLocation': 0}
    line = fin.readline()
    line = fin.readline()
    current = 'None'
    while line != '':
        x = line.split(',')
        ts = int(x[0])
        key = ts - ts % tr
        tp = x[1]
        if key != current:
            if current != 'None':
                oline = str(sample[feature_list[0]])
                for i in range(1, len(feature_list)):
                    oline = oline + ',' + str(sample[feature_list[i]])
                fout.write(oline + '\n')
                sample = {'InvokeSendAuthenticationInfo': 0,
                          'InvokeUpdateLocation': 0,
                          'InvokeUpdateGprsLocation': 0}
            current = key
        if tp in sample.keys():
            sample[tp] = sample[tp] + 1
        line = fin.readline()
    oline = str(sample[feature_list[0]])
    for i in range(1, len(feature_list)):
        oline = oline + ',' + str(sample[feature_list[i]])
    fout.write(oline + '\n')
    fin.close()
    fout.close()


def attack_pattern_2(file_name, tr):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.read_csv(file_name)
    print 'Time coverage: ' + str(df['Time'].max() - df['Time'].min())
    sample = {}
    for x in df.values:
        key = x[0] - x[0] % tr
        tp = x[1]
        if key not in sample.keys():
            sample[key] = {'InvokePurgeMs': 0,
                           'InvokeSendAuthenticationInfo': 0,
                           'InvokeUpdateLocation': 0}
        if tp in sample[key].keys():
            sample[key][tp] = sample[key][tp] + 1
    feature_list = ['InvokePurgeMs', 'InvokeSendAuthenticationInfo', 'InvokeUpdateLocation']
    output = 'ap-2-' + str(tr) + '.feature.csv'
    fout = open(output, 'wt')
    oline = feature_list[0]
    for i in range(1, len(feature_list)):
        oline = oline + ',' + feature_list[i]
    fout.write(oline + '\n')
    for key in sample.keys():
        oline = str(sample[key][feature_list[0]])
        for i in range(1, len(feature_list)):
            oline = oline + ',' + str(sample[key][feature_list[i]])
        fout.write(oline + '\n')
    fout.close()


def attack_pattern_3(file_name, tr):
    fin = open(file_name, 'rt')
    feature_list = ['InvokeMtForwardSM']
    output = './result/ap-3-' + str(tr) + '.feature.csv'
    fout = open(output, 'wt')
    oline = feature_list[0]
    for i in range(1, len(feature_list)):
        oline = oline + ',' + feature_list[i]
    fout.write(oline + '\n')
    sample = {'InvokeMtForwardSM': 0}
    line = fin.readline()
    line = fin.readline()
    current = 'None'
    while line != '':
        x = line.split(',')
        ts = int(x[0])
        key = ts - ts % tr
        tp = x[1]
        if key != current:
            if current != 'None':
                oline = str(sample[feature_list[0]])
                for i in range(1, len(feature_list)):
                    oline = oline + ',' + str(sample[feature_list[i]])
                fout.write(oline + '\n')
                sample = {'InvokeMtForwardSM': 0}
            current = key
        if tp in sample.keys():
            sample[tp] = sample[tp] + 1
        line = fin.readline()
    oline = str(sample[feature_list[0]])
    for i in range(1, len(feature_list)):
        oline = oline + ',' + str(sample[feature_list[i]])
    fout.write(oline + '\n')
    fin.close()
    fout.close()


def anomaly_1(input_name, anomaly_name, output_file):
    fin1 = open(input_name, 'r')
    fin2 = open(anomaly_name, 'r')
    fin1.readline()
    fout = open(output_file, 'w')
    line = fin1.readline()
    cnt = 1
    base = 1462788000000
    while line != '':
        line2 = fin2.readline()
        flag = int(line2)
        if flag == -1:
            fout.write(str(base + (cnt-1)*500) + ',' + line)
        cnt = cnt + 1
        line = fin1.readline()
    fout.close()
    fin1.close()
    fin2.close()


if __name__ == '__main__':
    file_name = './dd/NoisyData.csv'
    # attack_pattern_1(file_name, 500)
    # anomaly_1('./result/ap-1-500.feature.csv',
    #           './result/ap-1-500-train+test-result-rbf00005',
    #           './result/anomaly_1.csv')
    # attack_pattern_2(file_name, 500)
    attack_pattern_3(file_name, 60000)
    # simple_stat(file_name, 500)
    # to_tseq(file_name)
