import numpy as np 
from os import listdir
from sklearn import neighbors

def img2vector(fileName):
    retMat = np.zeros([1024], int)
    fr = open(fileName)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i*32+j] = lines[i][j]
    return retMat

def readDataSet(path):
    fileList = listdir(path)
    numFiles = len(fileList)
    dataSet = np.zeros([numFiles, 1024], int)
    hwLabels = np.zeros([numFiles])
    for i in range(numFiles):
        filePath = fileList[i]
        digit = int(filePath.split('_')[0])
        hwLabels[i] = digit
        dataSet[i] = img2vector(path+'/'+filePath)
    return dataSet, hwLabels

train_dataSet, train_hwLabels = readDataSet('../课程数据/手写数字/digits/trainingDigits')

knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
print('Start Training ..')
knn.fit(train_dataSet, train_hwLabels)


test_dataSet, test_hwLabels = readDataSet('../课程数据/手写数字/digits/testDigits')

res = knn.predict(test_dataSet)
error_num = np.sum(res != test_hwLabels)
num = len(test_dataSet)

print('Total num:', num, 'wrong num:', error_num, \
    'wrong Rate:', error_num/num)