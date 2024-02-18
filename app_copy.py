from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('KNN', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('./index_demo.html')


standard_to = StandardScaler()


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        applicant_inc = float((request.form['applicant_inc']))
        co_applicant_inc = float(request.form['co_applicant_inc'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = float(request.form['credit_history'])

        gender = request.form['gender']
        if gender == 'Male':
            gender = 1
        else:
            gender = 0

        education = request.form['education']
        if education == 'graduate':
            education = 1
        else:
            education = 0

        employed = request.form['employed']
        if employed == 'self_employed':
            employed = 1
        else:
            employed = 0

        married = request.form['married']
        if married == 'married':
            married = 1
        else:
            married = 0

        dependents = request.form['dependents']
        if dependents == 'dependents_0':
            dependents_0 = 1
            dependents_1 = 0
            dependents_2 = 0
            dependents_3 = 0
        elif dependents == 'dependents_1':
            dependents_0 = 0
            dependents_1 = 1
            dependents_2 = 0
            dependents_3 = 0
        elif dependents == 'dependents_2':
            dependents_0 = 0
            dependents_1 = 0
            dependents_2 = 1
            dependents_3 = 0
        else:
            dependents_0 = 0
            dependents_1 = 0
            dependents_2 = 0
            dependents_3 = 1

        property_area = request.form['property_area']
        if property_area == 'urban':
            property_area_urban = 1
            property_area_semi_urban = 0
            property_area_rural = 0
        elif property_area == 'semi_urban':
            property_area_urban = 0
            property_area_semi_urban = 1
            property_area_rural = 0
        else:
            property_area_urban = 0
            property_area_semi_urban = 0
            property_area_rural = 1

        prediction = model.predict([[applicant_inc, co_applicant_inc, loan_amount, loan_amount_term, credit_history,
                                     gender, married, dependents_0, dependents_1, dependents_2, dependents_3, education,
                                     employed, property_area_rural, property_area_semi_urban, property_area_urban]])

        if prediction == 1:
            return render_template('./index.html', prediction_texts="Congratulations! You will receive the loan!")
        else:
            return render_template('./index.html', prediction_text="Sorry! We cannot give you loan.")
    else:
        return render_template('./index.html')


if __name__ == "__main__":
    app.run(debug=True)
