import pickle
import pandas as pd
from flask import Flask, request, render_template

# Load the pre-trained model
with open('home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_house_price():
    if request.method == 'POST':
        try:
            # Get user input from the form
            total_sqft = float(request.form['total_sqft'])
            bath = float(request.form['bath'])
            bhk = float(request.form['bhk'])

            # Create a DataFrame with the user input
            input_data = pd.DataFrame([[total_sqft, bath, bhk]], columns=['total_sqft', 'bath', 'bhk'])

            # Make prediction using the model
            predicted_price = model.predict(input_data)[0]

            # Render the result on the web page
            return render_template('index.html', predicted_price=predicted_price, total_sqft=total_sqft, bath=bath, bhk=bhk)

        except Exception as e:
            error_message = "Error occurred: {}".format(e)
            return render_template('index.html', error_message=error_message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

