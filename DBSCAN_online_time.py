import numpy as np 
from sklearn.cluster import DBSCAN
import sklearn.cluster as skc 
from sklearn import metrics
import matplotlib.pyplot as plt 

data_path = '../课程数据/聚类/学生月上网时间分布-TestData.txt'
mac2id = {}
onlinetimes = []
f = open(data_path, encoding='utf-8')
for line in f:
    # print(line)
    mac = line.split(',')[2]
    onlinetime = int(line.split(',')[6])
    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])
    if mac not in mac2id:
        mac2id[mac] = len(onlinetimes)
        onlinetimes.append((starttime, onlinetime))
    else:
        onlinetimes[mac2id[mac]] = [(starttime, onlinetime)]
real_x = np.array(onlinetimes).reshape((-1, 2))

X = real_x[:, 0:1]
db = DBSCAN(eps=0.01, min_samples=20).fit(X)
labels = db.labels_

print('Labels:')
print(labels)
ratio = len(labels[labels[:] == -1]) / len(labels)
print('Noise ratio:', format(ratio, '.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print('Silhouette Coefficient: %.3f' % metrics.silhouette_score(X, labels))

for i in range(n_clusters_):
    print('Cluster', i, ':')
    print(list(X[labels == i].flatten()))

# plt.hist(X, 24)
# plt.show()