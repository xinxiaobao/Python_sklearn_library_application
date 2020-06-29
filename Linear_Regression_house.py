import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np 

dataset_x = []
dataset_y = []

fr = open('../课程数据/回归/prices.txt', 'r')
lines = fr.readlines()
for line in lines: 
    items = line.strip().split(',')
    dataset_x.append(int(items[0]))
    dataset_y.append(int(items[1]))
length = len(dataset_x)
dataset_x = np.array(dataset_x).reshape([length, 1])
dataset_y = np.array(dataset_y)

minX = min(dataset_x)
maxX = max(dataset_x)
x = np.arange(minX, maxX).reshape([-1, 1])

linear = linear_model.LinearRegression()
linear.fit(dataset_x, dataset_y)

print('Coefficients:', linear.coef_)
print('intercept:', linear.intercept_)

plt.scatter(dataset_x, dataset_y, color='red')
plt.plot(x, linear.predict(x), color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()