from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('rf_ckd_model.pkl')
encoders = joblib.load('label_encoders.pkl')  # this is a dictionary of LabelEncoders

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get input from form
        input_data = [request.form.get(field) for field in request.form]
        input_keys = list(request.form.keys())
        input_df = pd.DataFrame([input_data], columns=input_keys)

        # 2. Standardize strings (lowercase to match training encoders)
        for col in input_df.columns:
            if input_df[col].dtype == object:
                input_df[col] = input_df[col].astype(str).str.strip().str.lower()

        # 3. Apply encoders with unseen-label handling
        for col in input_df.columns:
            if col in encoders:
                encoder = encoders[col]
                known_classes = list(encoder.classes_)

                val = input_df[col].values[0]
                if val in known_classes:
                    input_df[col] = encoder.transform([val])
                else:
                    input_df[col] = -1  # Or handle differently (e.g., most frequent class)

        # 4. Predict
        prediction = model.predict(input_df)[0]
        output = encoders['classification'].inverse_transform([prediction])[0].capitalize()

        return render_template('index.html', prediction=f"Prediction: {output}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True) 