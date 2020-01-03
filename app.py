import pandas as pd 
import streamlit as st
import joblib
import numpy as np

if st.button('Hello'):
    st.write('Have a great day ahead!')
else:
    st.write('Good Morning')

#Web Title
st.title('Sales Forecasting')

#Web Description
st.write('We demonstrate how we can forecast advertising sales based on ad expenditure')

#Read Data
data = pd.read_csv('advertising_regression.csv')

data

st.balloons()

#create sidebar
st.sidebar.subheader('Advertising Costs')

#TV Slider
TV = st.sidebar.slider('TV Advertising Costs', 0, 300, 150)
                       
#Radio Slider
radio = st.sidebar.slider('Radio Advertising Costs', 0, 50, 15)

#Newspaper
newspaper = st.sidebar.slider('Newspaper Advertising Costs', 0, 250, 75)

st.subheader('Radio Advertising Cost Distribution')

# Histogram
hist_values = np.histogram(data.radio, bins=300, range=(0,100))[0]
                           
st.bar_chart(hist_values)

st.subheader('TV Advertising Cost Distribution')

# Histogram
hist_values = np.histogram(data.TV, bins=300, range=(0,100))[0]
                           
st.bar_chart(hist_values)

st.subheader('Newspaper Advertising Cost Distribution')

# Histogram
hist_values = np.histogram(data.newspaper, bins=300, range=(0,100))[0]
                           
st.bar_chart(hist_values)

saved_model = joblib.load('advertising_model.sav')

predicted_sales = saved_model.predict([[TV, radio, newspaper]])[0] * 1000

st.write(f"Predicted sales is {predicted_sales} dollars.")