import time
import pickle
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

start_time = time.time()

# 01 데이터 불러오기
df = sns.load_dataset('iris')
print(df.shape)
print(df.columns)
print(df.head())

sel = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# 02 라벨 인코딩(Label Encoding)
le = LabelEncoder()
y_lbl = le.fit_transform(df['species'])

# 03 데이터 나누기 - 입력, 출력
X = df[sel]
y = y_lbl

X_train, X_test, y_train, y_test =  train_test_split(X, y,
                                                     test_size=0.2, random_state=77)

# 04 모델 구축 및 학습, 모델 저장
tree = DecisionTreeClassifier(max_depth=8).fit(X_train,y_train)
print(tree.score(X_test, y_test))
pickle.dump(tree, open('./model/iris_tree.pkl', 'wb'))

print('시간 (초): ', time.time() - start_time)