# travel_recommendation_app.py

# ===============================
# Travel Package Recommendation System - Streamlit
# ===============================

import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # pyright: ignore[reportMissingModuleSource]
from sklearn.neighbors import NearestNeighbors # pyright: ignore[reportMissingModuleSource]
import warnings
warnings.filterwarnings("ignore")

# ===============================
# Step 1: Load Dataset
# ===============================
st.title("🌴 Travel Package Recommendation System")
st.write("Find the best travel packages based on your preferences!")

df = pd.read_csv("travel_packages_120000.csv")  # Replace with your CSV path

# ===============================
# Step 2: Define Columns
# ===============================
cat_cols = ['From_City', 'Destination', 'Destination_Type', 
            'Budget_Range', 'Accommodation_Type', 'Transport_Mode', 
            'Meal_Plan', 'Activity_Types', 'Season', 
            'Package_Type', 'Recommended_For']

num_cols = ['Trip_Duration_Days', 'Approx_Cost (₹)', 'Activity_Count']

# ===============================
# Step 3: Preprocess Data
# ===============================
# OneHotEncode categorical columns
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_features = ohe.fit_transform(df[cat_cols])

# Scale numeric columns
scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[num_cols])

# Combine numeric + categorical features
cdata = np.hstack([num_features, cat_features])

# Fit NearestNeighbors Model
cosinemodel = NearestNeighbors(n_neighbors=5, metric='cosine')
cosinemodel.fit(cdata)

# ===============================
# Step 4: User Input
# ===============================
st.sidebar.header("Enter Your Travel Preferences")

user_input = {
    'From_City': st.sidebar.selectbox('From City', df['From_City'].unique()),
    'Destination': st.sidebar.selectbox('Destination', df['Destination'].unique()),
    'Destination_Type': st.sidebar.selectbox('Destination Type', df['Destination_Type'].unique()),
    'Trip_Duration_Days': st.sidebar.slider('Trip Duration (Days)', int(df['Trip_Duration_Days'].min()), int(df['Trip_Duration_Days'].max()), 5),
    'Budget_Range': st.sidebar.selectbox('Budget Range', df['Budget_Range'].unique()),
    'Approx_Cost (₹)': st.sidebar.number_input('Approx Cost (₹)', int(df['Approx_Cost (₹)'].min()), int(df['Approx_Cost (₹)'].max()), 30000),
    'Accommodation_Type': st.sidebar.selectbox('Accommodation Type', df['Accommodation_Type'].unique()),
    'Transport_Mode': st.sidebar.selectbox('Transport Mode', df['Transport_Mode'].unique()),
    'Meal_Plan': st.sidebar.selectbox('Meal Plan', df['Meal_Plan'].unique()),
    'Activity_Count': st.sidebar.slider('Number of Activities', int(df['Activity_Count'].min()), int(df['Activity_Count'].max()), 3),
    'Activity_Types': st.sidebar.selectbox('Activity Types', df['Activity_Types'].unique()),
    'Season': st.sidebar.selectbox('Season', df['Season'].unique()),
    'Package_Type': st.sidebar.selectbox('Package Type', df['Package_Type'].unique()),
    'Recommended_For': st.sidebar.selectbox('Recommended For', df['Recommended_For'].unique())
}

user_df = pd.DataFrame([user_input])

# Transform user categorical input
user_cat = ohe.transform(user_df[cat_cols])

# Scale numeric input
user_num = scaler.transform(user_df[num_cols])

# Combine numeric + categorical
user_vector = np.hstack([user_num, user_cat])

# ===============================
# Step 5: Find Nearest Packages
# ===============================
distances, indices = cosinemodel.kneighbors(user_vector)

# Get top recommended packages
top_packages = df.iloc[indices[0]].copy()
top_packages['Similarity_Score'] = 1 - distances.flatten()  # Convert distance to similarity

top_packages_display = top_packages[['Package_ID', 'Destination', 'Trip_Duration_Days',
                                    'Approx_Cost (₹)', 'Accommodation_Type', 
                                    'Package_Type', 'Similarity_Score']]

st.subheader("Top Recommended Packages")
st.dataframe(top_packages_display.reset_index(drop=True))
