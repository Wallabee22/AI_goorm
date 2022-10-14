#%% 01. 라이브러리 임포트
import tensorflow as tf
import keras
import sys
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
# 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping

# 데이터셋 생성하기
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# figure, axes = plt.subplots(nrows=3, ncols=5)
# figure.set_size_inches(18, 12)
#
# plt.gray()
# print('label={}'.format(y_train[0:15]))
# col = 0
# for row in range(0,3):
#     col = row * 5
#     axes[row][0].imshow(X_train[col])
#     axes[row][1].imshow(X_train[col + 1])
#     axes[row][2].imshow(X_train[col + 2])
#     axes[row][3].imshow(X_train[col + 3])
#     axes[row][4].imshow(X_train[col + 4])

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 입력층의 값을 0~1사이로 만든다.
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
# print(X_train[0])

# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# 4. 모델 학습시키기
# validation_data : 모델을 학습할때는 기본 데이터를 이용하고, 평가시에 사용할 데이터를 지정
early_stopping = EarlyStopping(patience = 30)
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3000, batch_size=32, callbacks=[early_stopping])

## 06. 학습한 결과 확인하기
# plt.plot(hist.history['loss'], label="loss")
# plt.plot(hist.history['accuracy'], label="accuracy")
# plt.legend()
# plt.show()

loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)  # 최종 데이터 loss와 정확도(accuracy)

# loss: 0.0953 - accuracy: 0.9741 - val_loss: 0.1143 - val_accuracy: 0.9686