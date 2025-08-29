from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model1 = pickle.load(open(r'/home/aditya/JS/Project/model.pkl', 'rb'))

@app.route('/')

def home():
    return render_template('index.html') # The form to input 

@app.route('/predict', methods=['POST'])

def predict():

    val1 = float(request.form['feature1'])
    val2 = float(request.form['feature2'])
    val3 = float(request.form['feature3'])
    val4 = float(request.form['feature4'])
 
    input_data = [[val1, val2], [val3, val4]] # Two users' 
    predictions = model1.predict(input_data)
    return render_template('result.html', prediction=predictions)

if __name__ == '__main__':
    app.run(debug=True)