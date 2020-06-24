import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
y = data.target
x = data.data
n_components = 2
pca = PCA(n_components= n_components)
reduced_x = pca.fit_transform(x)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    if y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    if y[i] == 2:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

plt.scatter(red_x, red_y, c='r', marker='X')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()