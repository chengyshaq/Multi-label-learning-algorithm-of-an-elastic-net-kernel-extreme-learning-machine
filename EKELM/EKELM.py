# 本代码为：弹性网络核极限学习机的多标记学习算法（EKELM）
import numpy as np
import scipy.io as sio
from sklearn.linear_model import MultiTaskElasticNet
from metric import metrics,OneError
from kernel_mapping import kernel_matrix
data = sio.loadmat('Arts.mat')

train_data = data['train_data']
train_target = data['train_target']
test_data = data['test_data']
test_target = data['test_target']

# kernel_matrix为RBF核映射
omega_train = kernel_matrix(train_data)
omega_test = kernel_matrix(train_data,test_data,flag=2)
train_target = train_target.T
test_target = test_target.T

# 多任务弹性网络
# 参数设置需要调，这里给出大致的。
elnet = MultiTaskElasticNet(alpha=0.0001,l1_ratio=0.01,tol=0.4,max_iter=1000)
model_elnet = elnet.fit(omega_train.T, train_target)


Outputs_elnet = model_elnet.predict(omega_test.T)

HL,RL,CV,AP = metrics(Outputs_elnet,test_target)
OE = OneError(Outputs_elnet.T,test_target.T)
print(HL,RL,OE,CV,AP)