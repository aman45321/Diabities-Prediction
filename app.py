from flask import Flask, render_template, request,jsonify
from flask_cors import cross_origin
import pickle

app = Flask(__name__) # initializing a flask app
cross_origin()

@app.route('/',methods=['GET'])  # route to display the home page

def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI

def predict():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user

            Pregnancies = float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])

            filename = 'modelforprediction.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            scaler = pickle.load(open('StandardScaler.sav', 'rb'))
            prediction=loaded_model.predict(scaler.transform([[Pregnancies,Glucose,BloodPressure,
                                                                SkinThickness,
                                                                Insulin,BMI,DiabetesPedigreeFunction,
                                                                Age]]))
            print("Prediction is:",prediction)
            if prediction == 0:
                return render_template('0.html')
            else:
                return render_template('1.html')

            # showing the prediction results in a UI

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
        # return render_template('results.html')
    else:
        return render_template('index.html')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)
