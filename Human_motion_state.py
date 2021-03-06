import numpy as np 
import pandas as pd 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def load_dataset(feature_paths, label_paths):
    feature = np.ndarray(shape=(0, 41))
    label = np.ndarray(shape=(0, 1))

    for file in feature_paths:
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature, df))
    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))

    label = np.ravel(label)
    return feature, label

feature_paths = ['../课程数据/分类/dataset/A/A.feature', '../课程数据/分类/dataset/B/B.feature',
                '../课程数据/分类/dataset/C/C.feature', '../课程数据/分类/dataset/D/D.feature',
                '../课程数据/分类/dataset/E/E.feature']

label_paths = ['../课程数据/分类/dataset/A/A.label', '../课程数据/分类/dataset/B/B.label', 
                '../课程数据/分类/dataset/C/C.label', '../课程数据/分类/dataset/D/D.label',
                '../课程数据/分类/dataset/E/E.label']

x_train, y_train = load_dataset(feature_paths[:4], label_paths[:4])
x_test, y_test = load_dataset(feature_paths[4:], label_paths[4:])

# x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size=0.0)
x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size=0.01)

print('Start training KNN')
knn = KNeighborsClassifier().fit(x_train, y_train)
print('Training Done!')
answer_knn = knn.predict(x_test)
print('Predict Done!')

print('Start training DT')
dt = DecisionTreeClassifier().fit(x_train, y_train)
print('Training Done!')
answer_dt = dt.predict(x_test)
print('Predict Done!')

print('Start training Bays')
gnb = GaussianNB().fit(x_train, y_train)
print('Training Done!')
answer_gnb = gnb.predict(x_test)
print('Predict Done!')

print('\n\nThe classification report for knn:')
print(classification_report(y_test, answer_knn))

print('\n\nThe classification report for dt:')
print(classification_report(y_test, answer_dt))

print('\n\nThe classification report for gnb:')
print(classification_report(y_test, answer_gnb))