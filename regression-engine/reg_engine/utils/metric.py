
import numpy as np

# for test


def mean_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = np.abs(y_true - y_pred)
    # correct_kpts.append(eu)
    # # print(correct_kpts[0])
    # for threshold in range(0.1, 1, 0.1):

    #     acc_list = []

    #     for i in eu:
    #         if i < threshold:
    #             acc_list.append(i)

    #     acc = len(acc_list)
    #     accuracy = acc / (len(correct_kpts[0]) + 1e-6)
    acc_list = []
    threshold_list = []
    for threshold in range(5, 55, 5):
        threshold /= 100
        threshold_list.append(threshold)
        correct = (diff < threshold).sum()
        total = len(diff)
        acc = correct / (total + 1e-6)
        acc_list.append(acc)

    return acc_list, threshold_list


# for train
def mean_accuracy_log(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = np.abs(y_true - y_pred)
    # correct_kpts.append(eu)
    # # print(correct_kpts[0])
    # for threshold in range(0.1, 1, 0.1):

    #     acc_list = []

    #     for i in eu:
    #         if i < threshold:
    #             acc_list.append(i)

    #     acc = len(acc_list)
    #     accuracy = acc / (len(correct_kpts[0]) + 1e-6)
    acc_list = []
    threshold_list = []
    for threshold in range(5, 55, 5):
        threshold /= 100
        threshold_list.append(threshold)
        correct = (diff < threshold).sum()
        total = len(diff)
        acc = correct / (total + 1e-6)
        acc_list.append(acc)
    return np.average(acc_list)
