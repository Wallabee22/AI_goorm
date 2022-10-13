import time
import pickle
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

start_time = time.time()

# 01 데이터 불러오기
df = sns.load_dataset('titanic')
print(df.shape)
print(df.columns)
print(df.head())

df['embarked'].dropna(axis=0, inplace=True)
df['fare'].dropna(axis=0, inplace=True)

# 02 라벨 인코딩(Label Encoding)
le = LabelEncoder()
df['sex_num'] = le.fit_transform(df['sex'])
df['alone_num'] = le.fit_transform(df['alone'])
df['embarked_num'] = le.fit_transform(df['embarked'])

print(df['age'].isnull())

sel = ['pclass', 'sex_num', 'alone_num', 'embarked_num', 'fare']


# 03 데이터 나누기 - 입력, 출력
X = df[sel]
y = df['survived']

X_train, X_test, y_train, y_test =  train_test_split(X, y,
                                                     test_size=0.2, random_state=77)
print(X_train.shape, y_train.shape)

# 04 모델 구축 및 학습, 모델 저장
knn = KNeighborsClassifier().fit(X_train,y_train)
print(knn.score(X_test, y_test))
pickle.dump(knn, open('./model/titanic3_knn.pkl', 'wb'))

print('시간 (초): ', time.time() - start_time)

print(df['fare'].head())