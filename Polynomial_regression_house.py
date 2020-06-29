import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np 
from sklearn.preprocessing import PolynomialFeatures

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

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(dataset_x)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(x_poly, dataset_y)


plt.scatter(dataset_x, dataset_y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()