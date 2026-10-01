from flask import Flask, render_template, request, flash, redirect
import sqlite3
import pickle
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
from warnings import filterwarnings
import tensorflow as tf
filterwarnings('ignore')

app = Flask(__name__)


    

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

    
# @app.route('/logged')
# def logged():
#     return render_template('fetal.html')

def predict_new(input_data):
   # Load model and scaler
   loaded_model = tf.keras.models.load_model('model/battery_rul_cnn.h5')
   loaded_scaler = joblib.load('model/scaler.save')
   
   # Preprocess input
   scaled_input = loaded_scaler.transform([input_data])
   reshaped_input = scaled_input.reshape((1, scaled_input.shape[1], 1))
   
   # Predict
   prediction = loaded_model.predict(reshaped_input)
   return prediction[0][0]


@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if result:
            return render_template('logged.html')
        else:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')


# @app.route("/fetalPage", methods=['GET', 'POST'])
# def fetalPage():
#     return render_template('fetal.html')


@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        # Get form inputs
        cycle_index = float(request.form['cycle_index'])
        discharge_time = float(request.form['discharge_time'])
        decrement = float(request.form['decrement'])
        max_voltage = float(request.form['max_voltage'])
        min_voltage = float(request.form['min_voltage'])
        time_at_415v = float(request.form['time_at_415v'])
        time_constant_current = float(request.form['time_constant_current'])
        charging_time = float(request.form['charging_time'])
        
        # Create input array for model
        input_data = [
            cycle_index,
            discharge_time,
            decrement,
            max_voltage,
            min_voltage,
            time_at_415v,
            time_constant_current,
            charging_time
        ]
        
        # Make prediction
        predicted_rul = predict_new(input_data)
        
        # Pass all parameters to template for display
        return render_template(
            'result.html',
            pred=predicted_rul,
            cycle_index=cycle_index,
            discharge_time=discharge_time,
            decrement=decrement,
            max_voltage=max_voltage,
            min_voltage=min_voltage,
            time_at_415v=time_at_415v,
            time_constant_current=time_constant_current,
            charging_time=charging_time
        )
    
    return render_template('logged.html')

if __name__ == '__main__':
	app.run(debug = True)