from flask import Flask, render_template, request
from flask import jsonify
import pickle
import numpy as np
import warnings
import csv
import _json
app = Flask(__name__)
warnings.filterwarnings("ignore")
with open('model_pickle', 'rb') as f:
    mod = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


data = []


@app.route('/result1', methods=['POST', 'GET'])
def result1():
    if request.method == 'POST':
        f = request.form['csvfile']
        with open(f) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data.append(row)
        k = len(data)
        x = 0
        y = 0
        for i in range(1, k):
            x += int(data[i][0])
            y += int(data[i][1])
        x /= k-1
        y /= k-1
        x = round(x, 5)
        y = round(y, 5)
        return render_template('index.html', dt1=x, dt2=y)
@app.route('/result2', methods=['POST', 'GET'])
def result2():
    int_values = [x for x in request.form.values()]
    arr = np.array(int_values)
    result = mod.predict(arr.reshape(1, -1)).flatten()
    n = result[0].round(5)
    if (n < 33):
        y = "Great! You don't seem very stressed"
    elif (n < 66):
        x = "You have a moderate level of stress"
        y = "Practice relaxation techniques, Eat a healthy diet,Take breaks"
    else:
        x = "You have a moderate high of stress"
        y = "Seek Professional help,Practice deep breathing,get enough Sleep"
    return render_template('result.html', data1=n, data2=x, data3=y)

@app.route('/index')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
