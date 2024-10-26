from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import json

app = Flask(__name__)

def prepare_data():
    df = pd.read_csv('sample_data.csv')
    df.columns = ["Month", "Sales"]
    df.drop([106, 105], axis=0, inplace=True)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    return df

def train_model(df):
    model = SARIMAX(df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    return results

def make_prediction(model, df, months):
    last_date = df.index[-1]
    forecast = model.predict(
        start=len(df),
        end=len(df) + months - 1,
        dynamic=True
    )
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='M')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast.values})
    return forecast_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    months = int(request.form['months'])
    
    df = prepare_data()
    
    model = train_model(df)
    
    forecast_df = make_prediction(model, df, months)
    
    response = {
        'dates': forecast_df['Date'].dt.strftime('%Y-%m').tolist(),
        'forecasts': forecast_df['Forecast'].round(2).tolist(),
        'historical_dates': df.index.strftime('%Y-%m').tolist(),
        'historical_values': df['Sales'].round(2).tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)