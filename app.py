from flask import Flask, request, render_template
import pandas as pd
import joblib

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Load model and pipeline
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        data = {
            "longitude": float(request.form["longitude"]),
            "latitude": float(request.form["latitude"]),
            "housing_median_age": float(request.form["housing_median_age"]),
            "total_rooms": float(request.form["total_rooms"]),
            "total_bedrooms": float(request.form["total_bedrooms"]),
            "population": float(request.form["population"]),
            "households": float(request.form["households"]),
            "median_income": float(request.form["median_income"]),
            "ocean_proximity": request.form["ocean_proximity"]
        }

        df = pd.DataFrame([data])

        # ✅ If pipeline already includes preprocessing + model:
        # prediction = pipeline.predict(df)[0]

        # ✅ If model was trained separately:
        transformed_input = pipeline.transform(df)
        prediction = model.predict(transformed_input)[0]

        return render_template("index.html", prediction=f"{round(prediction, 2)}")
    except Exception as e:
        import logging
        logging.exception("Prediction Error:")
        return render_template("index.html", prediction="Error: Invalid input. Please check your values.")

if __name__ == "__main__":
    app.run(debug=True)
