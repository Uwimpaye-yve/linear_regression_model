import joblib
import numpy as np

model = joblib.load('best_life_expectancy_model.pkl')
scaler = joblib.load('scaler.pkl')

def get_prediction(data):
    # Data should be a list of features matching your X columns
    scaled = scaler.transform(np.array(data).reshape(1, -1))
    return model.predict(scaled)[0]

# Testing with (example data)
dummy_row = [2015, 1, 62, 0.01, 71.0, 0, 19.1, 83, 6, 8.16, 65, 0.1, 584.25, 33.7, 1.2, 1.3, 0.479, 10.1]
print(f"Predicted Life Expectancy: {get_prediction(dummy_row)}")