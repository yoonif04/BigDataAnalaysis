import numpy as np
from random import shuffle, seed


def getxfold(data, turn, fold):
    tot_length = len(data)
    each = int(tot_length / fold)
    mask = np.array([True if each * turn <= i < each * (turn + 1) else False for i in list(range(tot_length))])
    return data[~mask], data[mask]


# runCV 함수에 shuffle, fold, isAcc 추가하여 수정
def runCVshuffle(clf, data, labels, fold=10, isAcc=True):
    from sklearn.metrics import precision_recall_fscore_support
    seed(0)
    numbers = list(range(len(data)))
    shuffle(numbers)
    shuffled_data = data[numbers]
    shuffled_labels = labels[numbers]

    results = []
    for i in range(fold):
        data_tr, data_te = getxfold(shuffled_data, i, fold)
        labels_tr, labels_te = getxfold(shuffled_labels, i, fold)

        clf = clf.fit(data_tr, labels_tr)
        pred = clf.predict(data_te)
        correct = pred == labels_te
        if isAcc:
            acc = sum(correct) / len(correct)
            results.append(acc)
        else:
            results.append(precision_recall_fscore_support(pred, labels_te))
    return results


def load_mnist(path):
    import numpy as np
    f = open(path, 'r')
    f.readline()

    l_digit = []
    l_label = []
    for line in f.readlines():
        splitted = line.replace("\n", "").split(",")
        label = int(splitted[0])
        digit = np.array(splitted[1:], dtype=np.float32)
        l_label.append(label)
        l_digit.append(digit)

    digits = np.array(l_digit)
    norm_digits = digits / 255
    d_labels = np.array(l_label)

    return norm_digits, d_labels
