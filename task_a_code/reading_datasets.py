import pickle
import json
import csv

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def read_task(location, split = 'train'):
    filename = location + split + '.tsv'

    data = []
    with open(filename) as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for i, row in enumerate(tsv_reader):
            if i > 0:
                #tweet_id = row[0]
                sentence = row[0].strip()
                label = row[1]
                data.append((sentence, label))

    return data

def read_test(location, split = 'test'):
    filename = location + split + '.tsv'

    data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if i > 0:
                #tweet_id = row[0]
                sentence = row[0].strip()
                data.append((sentence))

    return data


if __name__ == '__main__':
    location = '../homophobia/datasets/task_a'
    split = 'train'
    
    data = read_task5(location, split)
    print(len(data))

    data = read_task5(location, 'dev')
    print(len(data))
    
    data = read_task5(location, 'test')
    print(len(data))

    




