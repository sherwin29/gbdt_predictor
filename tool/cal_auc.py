import numpy
import time
import random
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, metrics
from sklearn.preprocessing import StandardScaler

def cal_auc(file_name):
    files = open(file_name)
    result = []
    line = files.readline()
    negative_cnt, positive_cnt = line.split(",")
    line = files.readline()
    while line:
        line = line[:len(line) - 1]
        result.append(float(line))
        line = files.readline()
    # print negative_cnt, positive_cnt
    negative_cnt = int(negative_cnt)
    positive_cnt = int(positive_cnt[:len(positive_cnt) - 1])
    # print positive_cnt, negative_cnt
    labels = numpy.zeros(positive_cnt + negative_cnt)
    # print len(labels)
    for i in range(positive_cnt):
        labels[negative_cnt + i] = 1
    print metrics.roc_auc_score(labels, result)


if __name__ == "__main__":
    cal_auc("test_result.txt")
    cal_auc("train_result.txt")
    label = numpy.zeros(10000)
    val = numpy.zeros(10000)
    for i in range(5000):
        label[i + 5000] = 1
        val[i] = random.uniform(0.5, 1)
        val[i + 5000] = random.uniform(0.69, 1)
        # print metrics.roc_auc_score(label, val)


