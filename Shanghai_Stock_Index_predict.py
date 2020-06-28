import pandas as pd  
import numpy as np 
from sklearn import svm 
from sklearn import model_selection

data_path = '../课程数据/分类/stock/000777.csv'
data = pd.read_csv('./000777.csv', encoding='gbk', parse_dates=[0], index_col=0)
data.sort_index(0, ascending=True, inplace=True)

day_feature = 150
feature_num = 5 * day_feature

x = np.zeros((data.shape[0]-day_feature, feature_num + 1))
y = np.zeros((data.shape[0]-day_feature))

for i in range(0, data.shape[0]-day_feature):
    x[i, 0:feature_num] = np.array(data[i: i+day_feature][['收盘价', '最高价', '最低价', '最低价', '成交量']]).reshape((1, feature_num))
    x[i, feature_num] = data.ix[i+day_feature]['开盘价']

for i in range(0, data.shape[0]-day_feature):
    if data.ix[i+day_feature]['收盘价'] >= data.ix[i+day_feature]['开盘价']:
        y[i] = 1
    else:
        y[i] = 0

clf = svm.SVC(kernel='rbf')
result = []
for i in range(5):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2)
    clf.fit(x_train, y_train)
    result.append(np.mean(y_test == clf.predict(x_test)))

print('svm classifier accuracy:')
print(result)