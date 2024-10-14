from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the dataset and train model
data = pd.read_csv('Odisha_Power_Consumption_2019-20.csv')

# Convert 'Dt' column to datetime format
data['Dt'] = pd.to_datetime(data['Dt'], format='%d/%m/%Y', errors='coerce')
data = data.dropna(subset=['Dt'])

# Extract features (X) and target variable (y)
X = data[['Dt']].copy()
y = data['Odisha']

# Extract numerical features from the date
X['year'] = X['Dt'].dt.year
X['month'] = X['Dt'].dt.month
X['day'] = X['Dt'].dt.day
X['day_of_week'] = X['Dt'].dt.dayofweek
X['hour'] = X['Dt'].dt.hour
X = X.drop('Dt', axis=1)

# Train the model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestRegressor(random_state=1)
model.fit(X_scaled, y)

# Function to predict power consumption based on today's date
def predict_power(date):
    date_obj = pd.to_datetime(date)
    feature = {
        'year': date_obj.year,
        'month': date_obj.month,
        'day': date_obj.day,
        'day_of_week': date_obj.dayofweek,
        'hour': 0  # Assuming hour doesn't impact much for daily data
    }
    feature_df = pd.DataFrame([feature])
    feature_scaled = scaler.transform(feature_df)
    return model.predict(feature_scaled)[0]

# Generate a plot for estimated power consumption for the year
def plot_power_consumption():
    current_year = datetime.now().year
    dates = pd.date_range(start=f'{current_year}-01-01', end=f'{current_year}-12-31')
    predictions = [predict_power(date) for date in dates]
    
    plt.figure(figsize=(10,6))
    plt.plot(dates, predictions, label='Estimated Power Consumption')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (MU)')
    plt.title(f'Estimated Power Consumption in {current_year}')
    plt.legend()

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f'data:image/png;base64,{plot_url}'

# Home route
@app.route('/')
def home():
    today = datetime.now().strftime('%Y-%m-%d')  # Format the current date
    display_date = datetime.now().strftime('%d/%m/%Y')  # Format as dd/mm/yyyy for display
    predicted_power = round(predict_power(today))
    plot_url = plot_power_consumption()
    return render_template('index.html', predicted_power=predicted_power, plot_url=plot_url, current_date=display_date)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
