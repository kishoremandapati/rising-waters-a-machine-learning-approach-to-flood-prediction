from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# -------------------------------------------------
# Load Trained Model
# -------------------------------------------------
model_path = os.path.join("models", "flood_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found! Please train the model first.")

model = pickle.load(open(model_path, "rb"))

# -------------------------------------------------
# Home Route
# -------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


# -------------------------------------------------
# Prediction Route
# -------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from form
            annual = float(request.form['annual'])
            jan_feb = float(request.form['jan_feb'])
            mar_may = float(request.form['mar_may'])
            jun_sep = float(request.form['jun_sep'])
            oct_dec = float(request.form['oct_dec'])

            # IMPORTANT: Feature order must match training
            features = np.array([[annual, jan_feb, mar_may, jun_sep, oct_dec]])

            # Prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            percent = round(probability * 100, 2)

            if prediction == 0:
                return render_template('low.html', percent=percent)
            else:
                return render_template('high.html', percent=percent)

        except ValueError:
            return "Invalid input! Please enter numeric values only."
        except Exception as e:
            return f"Unexpected Error: {e}"

    return render_template('predict.html')


# -------------------------------------------------
# Run App
# -------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)