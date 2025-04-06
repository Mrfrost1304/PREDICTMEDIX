
import os
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('insurancemodelf_fullfeatures.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'age': int(request.form['age']),
            'sex': 1 if request.form['sex'] == 'male' else 0,
            'bmi': float(request.form['bmi']),
            'children': int(request.form['children']),
            'smoker': 1 if request.form['smoker'] == 'yes' else 0,
            'region': {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}[request.form['region']]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data, index=[0])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return render_template('index.html', 
                            prediction_text=f'Predicted Insurance Charges: ${prediction:,.2f}')
    
    except Exception as e:
        return render_template('index.html', 
                            prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
