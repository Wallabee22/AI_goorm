from flask import Flask, render_template, request
import pickle
import numpy as np


# 01 학습된 모델 가져오기
model = pickle.load(open('./model/titanic3_knn.pkl', 'rb'))


# 02 플라스크 사용을 위한 준비
app = Flask(__name__)


# 03 플라스크 앱의 루트 디렉터리를 초기화
@app.route('/')
def main():
    return render_template('start_t.html')


# 04 초기 웹 페이지에서 submit 했을 때 실행
# request.form['']을 사용하여 HTML 페이지에서 입력한 데이터를 가져온다.
# model.predict()를 통해 클래스를 예측한다.
# 예측값에 따라 어떤 텍스트와 이미지를 보낼지, after.html에 설정.
@app.route('/predict', methods=['POST'])
def home():
    val1 = request.form['a']
    val2 = request.form['b']
    val3 = request.form['c']
    val4 = request.form['d']
    val5 = request.form['e']
    val1 = int(val1); val2 = int(val2); val3 = int(val3)
    val4 = int(val4); val5 = int(val5);
    arr = np.array([[val1, val2, val3, val4, val5]])
    pred = model.predict(arr)


    # 렌더링할 html 파일명, 전달할 변수
    return render_template('after_t.html', data=pred)


# 05 직접 실행된 경우, 앱을 디버그 on 모드로 실행
if __name__ == "__main__":
    app.run(debug=True)