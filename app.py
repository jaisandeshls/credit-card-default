from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("final_model.pkl", "rb"))

def predict_default(features):

    features = np.array(features).astype(np.float64).reshape(1,-1)

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return prediction, probability

@app.route('/', methods = ['GET','POST'])
def home():
     return render_template("index.html")

@app.route('/result', methods = ['POST'])
def prediction():
    try:
        if request.method == 'POST':

            features = request.form.to_dict()

            actual_feature_names = ['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'BILL_AMT1', 'BILL_AMT2', 
                                    'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 
                                    'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            feature_values = [features[i] for i in actual_feature_names]

            prediction, probability = predict_default(feature_values)
            if prediction[0] == 1:
                predicts = "This account will be defaulted with a probability of {}%.".format(round(np.max(probability)*100, 2))
            else:
                predicts = "This account will not be defaulted with a probability of {}%.".format(round(np.max(probability)*100, 2))
    except:
        predicts = 'Irrelevant information.'
    return render_template("result.html", predicts = predicts)

if __name__ == '__main__':
    app.run(debug = True)