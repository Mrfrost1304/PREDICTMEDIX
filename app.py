
import os
from flask import Flask, request, render_template, send_from_directory
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model with error handling
try:
    with open('insurancemodelf_fullfeatures.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure insurancemodelf_fullfeatures.pkl exists.")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/resumes/<filename>')
def download_resume(filename):
    # Create resumes directory if it doesn't exist
    os.makedirs('resumes', exist_ok=True)
    return send_from_directory('resumes', filename, as_attachment=True)


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
