from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open(r"/home/aditya/JS/Project/taxi_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
            float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4'])
        ]
        
        features = np.array(input_data).reshape(1, -1)

        prediction = model.predict(features)[0]

        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)