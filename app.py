from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("final_model.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sleep = float(request.form["sleep_hours"])
        screen = float(request.form["screen_time"])
        input_data = np.array([[sleep, screen]])
        
        prediction = model.predict(input_data)[0]  # This returns "optimal" or "not optimal"
        
        return render_template("result.html", prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
