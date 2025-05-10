from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import sqlite3
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Load the model
with open('model/logistic_regression_top_features.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler (same one used during training)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset to extract feature names (for the form)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df_columns = data.feature_names

# These are the top 10 features used (from your feature importance)
top_features = [
    'worst texture',
    'radius error',
    'worst symmetry',
    'mean concave points',
    'worst concavity',
    'area error',
    'worst radius',
    'worst area',
    'mean concavity',
    'worst concave points',
]

# Function to connect to SQLite Database
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize the database (create tables if they don't exist)
def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        patient_name TEXT,
                        features TEXT,
                        prediction INTEGER)''')
    conn.commit()
    conn.close()

# Initialize DB (run this once to create the tables if they don't exist)
init_db()

@app.route('/')
def index():
    # Render the index page with feature names for the form
    return render_template('index.html', features=top_features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get patient name
        patient_name = request.form.get('patient_name')

        # Collect input features
        input_features = [float(request.form[feature]) for feature in top_features]
        input_array = np.array([input_features])

        # Scale the input features (use transform instead of fit_transform)
        scaled_input = scaler.transform(input_array)

        # Get prediction from model
        prediction = model.predict(scaled_input)[0]
        
        # Assign result based on prediction (ensure correct mapping)
        result = "Malignant" if prediction == 1 else "Benign"

        # Save to DB
        conn = get_db_connection()
        conn.execute('''INSERT INTO predictions (date, patient_name, features, prediction) 
                        VALUES (?, ?, ?, ?)''',
                     (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      patient_name,
                      str(input_features),
                      prediction))
        conn.commit()
        conn.close()

        # Render result page
        return render_template('result.html', result=result, prediction=prediction, patient_name=patient_name)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
