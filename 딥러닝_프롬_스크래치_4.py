import sys, os
#sys.path.append(os.pardir)
#print("os.pardir : ", os.pardir)
import numpy as np
from dataset.mnist import load_mnist

def cross_entropy_error(y, t) :
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
#####################################################
# 4_2_3
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

print(x_train.shape)        # (60000, 784)
print(t_train.shape)        # (60000, 10)

print(x_test.shape)     # (10000, 784)
print(t_test.shape)     # (10000, 10)

#for i in range(0, len(t_train)) :
#    print(t_train[i])
#for i in range(0, len(t_test)) :
#    print(t_test[i])

train_size = x_train.shape[0]       # (60000, 784) 중 0번째 인덱스 값인 60000
# print(x_train.shape[0])       # 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)       # 0 ~ 60000 사이의 무작위 값 batch_size 만큼(10개) 고른다

#for i in range(0, len(batch_mask)) :
#    print(batch_mask[i])
#    print(t_train[batch_mask[i]])

# x.size : 형상 인덱스들의 곱
print(x_train.size)
print(x_test.size)

print(x_train.ndim)     # 2, ndim 은 차원 수를 알려준다.

#####################################################
# 4_2_4
def cross_entropy_error_2(y, t) :
    if y.ndim == 1 :
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
#####################################################
# 4_2_4
def cross_entropy_error
