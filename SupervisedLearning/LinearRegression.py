#Init setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

ds_train = pd.read_csv('F:/MachineLearning/DataSets/random-linear-regression/train.csv')
ds_test = pd.read_csv('F:/MachineLearning/DataSets/random-linear-regression/test.csv')

X_train = ds_train['x']
y_train = ds_train['y']
x_test = ds_test['x']
y_test = ds_test['y']

# X_train =np.random.rand(9,1)
# y_train =np.random.rand(9,1)
# x_test = np.random.rand(9,1)
# y_test = np.random.rand(9,1)

X_train = np.array(X_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

X_train = X_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
#print(y_train)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
#removing nan
X_train = np.nan_to_num(X_train)
x_test = np.nan_to_num(x_test)
y_train=np.nan_to_num(y_train)
y_test= np.nan_to_num(y_test)
# print('Ytrain')
# print(y_train)

n=np.size(x_test)
learningRate = 0.0001
a_0 = np.zeros((n,1))
a_1 = np.zeros((n,1))

# y_predict1=a_0+a_1*X_train
# print('prediction :')
# print(y_predict1)
# print('Y_Train')
# print(y_train)
# error1 = y_predict1-y_train
# print('Error :')
# print(error1)
# mse1 = np.sum(error1**2)
# print('MSE :')
# print(mse1)
epochs= 0
while epochs<1000 :

    y_predict=a_0+a_1*x_test

    error = y_predict-y_test
    mse = np.sum(error**2)
    mse = mse/n
    a_0 = a_0 - (learningRate*2*np.sum(error))/n
    a_1 = a_1 -(learningRate*2*np.sum(error*x_test))/n
    epochs+=1
    if epochs%10 == 0:
       print(mse)


y_prediction = a_0 + a_1 * x_test
print('a-0',a_0)
print('a_1',a_1)
print('R2 Score:',r2_score(y_test,y_prediction))

y_plot = []
for i in range(100):
    y_plot.append(a_0 + a_1 * i)
print('Y_Plot',np.shape(y_plot))
print('Y_Test',np.shape(y_test))

print(range(len(y_plot)))
plt.figure(figsize=(10,10))
plt.scatter(x_test,y_test,color='red',label='GT')
#plt.plot(,y_plot,color='black',label = 'pred')
plt.legend()
plt.show()