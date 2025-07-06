from flask import Flask, render_template, request
import joblib
import numpy as np

# ✅ Create Flask app
app = Flask(__name__)

# ✅ Load the saved model
model = joblib.load('model.pkl')

# ✅ Home page
@app.route('/')
def home():
    return render_template('input.html')

# ✅ Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input for 10 features
        input_features = [float(x) for x in request.form.values()]

        # ✅ Fill the remaining 29 features with zeros
        missing_features = [0] * (39 - len(input_features))  # 39 total features

        # Combine user input and missing features
        final_features = input_features + missing_features

        # ✅ Ensure the array matches expected shape
        final_features = np.array(final_features).reshape(1, -1)

        # Predict
        prediction = model.predict(final_features)[0]

        # Map prediction
        galaxy_type = "🌌 STARFORMING" if prediction == 1 else "✨ STARBURST"

        return render_template('output.html', prediction=galaxy_type)

    except Exception as e:
        return f"❌ Error: {e}. Please check your input values."

# ✅ Run the app
if __name__ == "__main__":
    app.run(debug=True)
