from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('insurancemodelf.pkl', 'rb') as f:
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
            'bmi': float(request.form['bmi']),
            'children': int(request.form['children']),
            'smoker': 1 if request.form['smoker'] == 'yes' else 0
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
    app.run(debug=True)
