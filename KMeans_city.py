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



# data_path = './city.txt'
data_path = '..//课程数据/聚类/31省市居民家庭消费水平-city.txt'
data, cityName = loadData(data_path)
print(len(data))