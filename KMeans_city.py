import numpy as np 
from sklearn.cluster import KMeans


def loadData(filePath):
    fr = open(filePath, 'r', 1, 'gbk')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(',')
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName



data_path = '../课程数据/聚类/31省市居民家庭消费水平-city.txt'
data, cityName = loadData(data_path)
n_clusters = int(input('请输入需要聚类数量：'))
km = KMeans(n_clusters)

label = km.fit_predict(data)
expenses = np.sum(km.cluster_centers_, axis=1)

CityCluster = ()
for i in range(n_clusters):
    CityCluster += ([],)


for i in range(len(cityName)):
    CityCluster[label[i]].append(cityName[i])

for i in range(len(CityCluster)):
    print('Expenses: %.2f'%expenses[i])
    print(CityCluster[i])
