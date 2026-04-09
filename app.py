from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    print(f"Warning: {MODEL_PATH} not found. Please run build_model.py to train and save the model.")

@app.route('/')
def home():
    return render_template('index.html')

# Valid input ranges based on dataset min/max (with slight buffer)
INPUT_CONSTRAINTS = {
    'N': (0, 150),
    'P': (0, 150),
    'K': (0, 210),
    'temperature': (5, 50),
    'humidity': (10, 100),
    'ph': (0, 14),
    'rainfall': (10, 350),
}

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', top_predictions=[])

    if request.method == 'POST':
        try:
            # Extract features from form submission
            fields = {
                'N': float(request.form['N']),
                'P': float(request.form['P']),
                'K': float(request.form['K']),
                'temperature': float(request.form['temperature']),
                'humidity': float(request.form['humidity']),
                'ph': float(request.form['ph']),
                'rainfall': float(request.form['rainfall']),
            }

            # Validate ranges
            violations = []
            for name, value in fields.items():
                min_val, max_val = INPUT_CONSTRAINTS[name]
                if value < min_val or value > max_val:
                    label = name.capitalize() if name != 'ph' else 'pH'
                    violations.append(f"{label} must be between {min_val} and {max_val} (got {value})")

            if violations:
                return render_template('result.html', top_predictions=[], error="Invalid input values: " + "; ".join(violations))

            # Model prediction
            features = np.array([[fields['N'], fields['P'], fields['K'], fields['temperature'], fields['humidity'], fields['ph'], fields['rainfall']]])
            probabilities = model.predict_proba(features)[0]
            crop_classes = model.classes_

            # Combine crops with their probabilities and sort descending
            crop_probs = sorted(zip(crop_classes, probabilities), key=lambda x: x[1], reverse=True)

            # --- Soil Analysis ---
            soil_bads = []
            if fields['ph'] < 5.5:
                soil_bads.append('Highly Acidic')
            elif fields['ph'] < 6.5:
                soil_bads.append('Slightly Acidic')
            elif fields['ph'] < 7.5:
                soil_bads.append('Neutral')
            elif fields['ph'] < 8.5:
                soil_bads.append('Slightly Alkaline')
            else:
                soil_bads.append('Highly Alkaline')

            # Moisture level
            if fields['humidity'] < 30:
                soil_bads.append('Low Humidity')
            elif fields['humidity'] < 60:
                soil_bads.append('Moderate Humidity')
            else:
                soil_bads.append('High Humidity')

            # Nitrogen level
            if fields['N'] < 30:
                soil_bads.append('Low Nitrogen')
            elif fields['N'] < 70:
                soil_bads.append('Moderate Nitrogen')
            else:
                soil_bads.append('High Nitrogen')

            # Rainfall
            if fields['rainfall'] < 50:
                soil_bads.append('Dry Climate')
            elif fields['rainfall'] < 150:
                soil_bads.append('Moderate Rainfall')
            else:
                soil_bads.append('High Rainfall')

            # Build top 3 result data
            top_predictions = []
            for i in range(min(3, len(crop_probs))):
                crop_name = crop_probs[i][0]
                confidence = crop_probs[i][1] * 100
                top_predictions.append({
                    'crop': crop_name.capitalize(),
                    'confidence': round(confidence, 2),
                    'image': crop_name.lower(),
                    'rank': i + 1
                })

            return render_template('result.html', top_predictions=top_predictions, soil_analysis=soil_bads)

        except ValueError:
            return render_template('result.html', top_predictions=[], error="Please enter valid numeric values for all fields.")
        except Exception as e:
            return render_template('result.html', top_predictions=[], error=str(e))

    return render_template('result.html', top_predictions=[])

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
