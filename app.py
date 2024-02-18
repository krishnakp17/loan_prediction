import io

from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from reportlab.pdfgen import canvas
from flask import make_response

app = Flask(__name__)
model = pickle.load(open('RF_model_loan', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('index2.html')


standard_to = StandardScaler()


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
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

        if prediction[0] == 1:
            details_entry = {
                'name': request.form['name'],
                'applicant_inc': float(request.form['applicant_inc']),
                'co_applicant_inc': float(request.form['co_applicant_inc']),
                'loan_amount': float(request.form['loan_amount']),
                'loan_amount_term': float(request.form['loan_amount_term']),
                'credit_history': float(request.form['credit_history']),
                'gender': request.form['gender'],
                'married': request.form['married'],
                'dependents': request.form['dependents'],
                'education': request.form['education'],
                'employed': request.form['employed'],
                'property_area': request.form['property_area']
            }
            return render_template('LA.html', details=details_entry)
        else:
            return render_template('LR.html')
    else:
        return render_template('Predict.html')


@app.route("/print_details", methods=['POST'])
def print_details():
    details_entry = request.form['details']
    details = eval(details_entry)

    pdf_filename = "customer_details.pdf"

    # Generate PDF
    response = make_response(pdf_from_details(details))
    response.headers['Content-Disposition'] = f'attachment; filename={pdf_filename}'
    response.headers['Content-Type'] = 'application/pdf'

    return response


def pdf_from_details(details):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)

    # PDF content
    p.drawString(100, 800, "Loan Approved - Customer Details")
    y_position = 780
    for key, value in details.items():
        y_position -= 20
        p.drawString(100, y_position, f"{key}: {value}")

    p.save()
    pdf_value = buffer.getvalue()
    buffer.close()

    return pdf_value


if __name__ == "__main__":
    app.run(debug=True)
