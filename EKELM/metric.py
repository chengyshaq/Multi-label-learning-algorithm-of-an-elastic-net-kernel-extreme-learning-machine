# HL，RL，CV，AP by sklearn and PGSmall（只会使用库函数）
# OE by 钱坤（大神）
from sklearn.metrics import hamming_loss, label_ranking_loss, coverage_error
from sklearn.metrics import average_precision_score
import numpy as np

def metrics(Outputs,test_target):
    test_target = np.maximum(test_target,0)

    Pre_Labels = np.sign(Outputs)
    Pre_Labels = np.maximum(Pre_Labels,0)
    num_instance,num_class = np.shape(test_target)

    sumhl=0.0
    for i in range(num_class):
        y_true = test_target[:,i]
        y_pred = Pre_Labels[:,i]

        hammingloss = hamming_loss(y_true, y_pred)
        sumhl = sumhl + hammingloss

    HL = sumhl/float(num_class)
    RL = label_ranking_loss(test_target,Outputs)
    CV = coverage_error(test_target,Outputs)-1
    AP = average_precision_score(test_target,Outputs,average='samples')

    return HL,RL,CV,AP


def OneError(outPuts, test_targets):
    num_class, num_instance = np.mat(outPuts).shape
    tempOutputs = outPuts
    tempTestTargets = test_targets
    for i in range(num_instance):
        temp = test_targets[:, i]
        if ((sum(temp) == num_class) | (sum(temp) == -num_class)):  # 将 & ---> |
            tempOutputs = np.delete(tempOutputs, i, axis=1)  # 赋值给原矩阵
            tempTestTargets = np.delete(tempTestTargets, i, axis=1)
    after_num_class, after_num_instance = np.mat(tempOutputs).shape
    tempTestTargets = np.mat(tempTestTargets)

    label = []
    notLabel = []
    labelSize = np.zeros((1, after_num_instance))
    for i in range(after_num_instance):
        temp = tempTestTargets[:, i]
        labelSize[0, i] = sum((temp == np.ones((after_num_class, 1))))
        tempLabel = []
        tempNotLabel = []
        for j in range(after_num_class):
            if (temp[j] == 1):
                tempLabel.append(j)
            else:
                tempNotLabel.append(j)
        label.append(tempLabel)
        notLabel.append(tempNotLabel)

    oneErr = 0
    for i in range(after_num_instance):
        indicator = 0
        temp = tempOutputs[:, i]
        maximum = max(temp)
        for j in range(after_num_class):
            if (temp[j] == maximum):
                if (j in label[i]):
                    indicator = 1
                    break
        if (indicator == 0):
            oneErr = oneErr + 1
    OneErr = oneErr / after_num_instance
    return OneErr
