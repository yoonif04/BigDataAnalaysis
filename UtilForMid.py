
import numpy as np
from random import shuffle, seed
import pandas as pd

def getxfold(data, turn, fold):
    tot_length = len(data)
    each = int(tot_length / fold)
    mask = np.array([True if each * turn <= i < each * (turn + 1) else False for i in list(range(tot_length))])
    return data[~mask], data[mask]

# runCV 함수에 shuffle, fold 추가 + 맞게 예측한 인덱스 정보
def runCVshuffle(clf, data, labels, fold=10):
    seed(0)
    numbers = list(range(len(data)))
    shuffle(numbers)
    shuffled_data = data[numbers]
    shuffled_labels = labels[numbers]

    accuracies = []
    c_index = []
    for i in range(fold):
        data_tr, data_te = getxfold(shuffled_data, i, fold)
        labels_tr, labels_te = getxfold(shuffled_labels, i, fold)

        clf = clf.fit(data_tr, labels_tr)
        pred = clf.predict(data_te)
        correct = pred == labels_te
        for x in range(len(labels_te)):
            if pred[x, 0] == labels_te[x, 0] & labels_te[x, 0] == 1:
                c_index.append(x)

        acc = sum(correct) / len(correct)
        accuracies.append(acc)
    return accuracies, c_index


def runCV(clf, data, labels, fold=10):
    seed(0)
    numbers = list(range(len(data)))
    shuffle(numbers)
    shuffled_data = data[numbers]
    shuffled_labels = labels[numbers]

    accuracies = []
    c_index = []
    for i in range(fold):
        data_tr, data_te = getxfold(shuffled_data, i, fold)
        labels_tr, labels_te = getxfold(shuffled_labels, i, fold)

        clf = clf.fit(data_tr, labels_tr)
        pred = clf.predict(data_te)
        correct = pred == labels_te
        for x in range(len(labels_te)):
            if pred[x] == labels_te[x] & labels_te[x] == 1:
                c_index.append(x)

        acc = sum(correct) / len(correct)
        accuracies.append(acc)
    return accuracies, c_index




def featurenum(findcol, c_index, kind):
    c_data = findcol[c_index]    # 찾고자 하는 특징의 데이터에서 correct 데이터만 모음
    feature_num = []
    for x in range(kind):
        feature_num.append(sum(c_data[:, x]))

    return feature_num, c_data
