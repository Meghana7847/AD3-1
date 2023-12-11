# popularity_predictor_app.py

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load the dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('youtube.csv')

# Features and target variable
numeric_features = ['video_views','lowest_yearly_earnings','lowest_monthly_earnings','highest_yearly_earnings','highest_monthly_earnings']
target = 'subscribers'

# Ensure only numeric features are included
df_numeric = df[numeric_features + [target]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_numeric.drop(columns=[target]), df_numeric[target], test_size=0.2, random_state=42)

# Train a Random Forest Regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Function to predict popularity
def predict_subscribers(video_views,lowest_yearly_earnings,lowest_monthly_earnings,highest_yearly_earnings,highest_monthly_earnings):
    input_data = pd.DataFrame({
        'video_views': [video_views],
        'lowest_yearly_earnings': [lowest_yearly_earnings],
        'lowest_monthly_earnings': [lowest_monthly_earnings],
        'highest_yearly_earnings': [highest_yearly_earnings],
        'highest_monthly_earnings': [highest_monthly_earnings],
       
    })

    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
st.title('🎵 Subscriber Predictor')
st.markdown('## Predict the no of subscribers of your youtube channel!!')





# Input form for user to enter feature values
video_views = st.slider('video_views', min_value=0, max_value=100000, step=1)
lowest_yearly_earnings = st.slider('lowest_yearly_earnings', min_value=0, max_value=100000, step=1)
lowest_monthly_earnings = st.slider('lowest_monthly_earnings', min_value=-20, max_value=100000, step=1)
highest_yearly_earnings = st.slider('highest_yearly_earnings', min_value=0, max_value=100000, step=1)
highest_monthly_earnings= st.slider('highest_monthly_earnings', min_value=0, max_value=10000, step=1)


# Predict button
if st.button('Predict Subcribers'):
    prediction = predict_subscribers(video_views,lowest_yearly_earnings,lowest_monthly_earnings,highest_yearly_earnings,highest_monthly_earnings)
    st.success(f'Predicted subscribers: {prediction}')

