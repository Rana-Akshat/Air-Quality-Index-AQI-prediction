from flask import Flask, request, render_template, jsonify 
import pickle
import pandas as pd
import numpy as np
import joblib
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit_form():
    County = request.form['County']
    State = request.form['State']
    year = request.form['Year']

    with open('encoded_values.pkl', 'rb') as file:
        mapping_dict = pickle.load(file)

    county = mapping_dict['County'][County]
    state = mapping_dict['State'][State]

    with open('fixedData.pkl', 'rb') as file:
        df = pickle.load(file)


    series = df.loc[(df['State'] == State) & (df['County'] == County)]
    
    input_dict = {
        'State': state,
        'County' : county,
        'Year': int(year),
        'Days with AQI': int(series['Days with AQI'].iloc[0]),
        'Good Days': int(series['Good Days'].iloc[0]),
        'Moderate Days': int(series['Moderate Days'].iloc[0]),
        'Unhealthy for Sensitive Groups Days':int(series['Unhealthy for Sensitive Groups Days'].iloc[0]),
        'Unhealthy Days': int(series['Unhealthy Days'].iloc[0]),
        'Very Unhealthy Days': int(series['Very Unhealthy Days'].iloc[0]),
        'Hazardous Days': int(series['Hazardous Days'].iloc[0]),
        'Max AQI': int(series['Max AQI'].iloc[0]),
        'Median AQI': int(series['Median AQI'].iloc[0]),
        'Days CO': int(series['Days CO'].iloc[0]),
        'Days NO2': int(series['Days NO2'].iloc[0]),
        'Days Ozone':int(series['Days Ozone'].iloc[0]),
        'Days PM2.5':int(series['Days PM2.5'].iloc[0]),
        'Days PM10': int(series['Days PM10'].iloc[0])
    }

    input_df = pd.DataFrame([input_dict])

    scaler = joblib.load('scaler.bin')
    input_array = scaler.transform(input_df)

    with open('model_64_64_1.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(input_array)
    prediction = predictions.astype(int)
    output = "The predicted AQI for year "+ year
    output += " is "
    output +=  str(prediction[0][0])
    print(output)

    return jsonify(content=output)

if __name__ == '__main__':
    app.run()