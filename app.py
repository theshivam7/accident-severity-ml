import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🚗 Accident Severity Prediction System")
st.markdown("""
This application predicts the severity of road accidents based on various factors such as road type, weather conditions, 
and accident location. The prediction helps in determining the required level of emergency response deployment.
""")

# Load data and model
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('accident_data.csv')
        # Drop any rows with missing values
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_and_save_model(data):
    try:
        # Prepare features
        X = data[['Road_Type', 'Weather', 'Accident_Location', 'Collision_Type']]
        X = pd.get_dummies(X)
        
        # Store feature columns for later use
        feature_columns = X.columns.tolist()
        joblib.dump(feature_columns, 'feature_columns.pkl')
        
        # Prepare target and ensure no NaN values
        severity_map = {'Fatal': 4, 'Grievous Injury': 3, 'Simple Injury': 2, 'Damage Only': 1}
        y = data['Severity'].map(severity_map)
        
        # Drop any rows where target is NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(y) == 0:
            st.error("No valid target values found after cleaning the data.")
            return None, None
        
        # Train model
        model = CatBoostClassifier(iterations=100, depth=10, learning_rate=0.3, random_seed=42, verbose=0)
        model.fit(X, y)
        
        # Save model
        model.save_model('catboost_model.cbm')
        
        return model, feature_columns
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

@st.cache_resource
def load_model():
    try:
        # Check if model and feature columns files exist
        model_exists = os.path.exists('catboost_model.cbm')
        features_exist = os.path.exists('feature_columns.pkl')
        
        if model_exists and features_exist:
            # Load existing model and features
            model = CatBoostClassifier()
            model.load_model('catboost_model.cbm')
            feature_columns = joblib.load('feature_columns.pkl')
            return model, feature_columns
        else:
            # Train new model if either file is missing
            st.info("Training new model...")
            data = load_data()
            if data is None:
                return None, None
            return train_and_save_model(data)
            
    except Exception as e:
        st.error(f"Error loading/training model: {str(e)}")
        return None, None

# Load data and model
data = load_data()
if data is None:
    st.error("Failed to load data. Please check the data file.")
    st.stop()

model, feature_columns = load_model()
if model is None or feature_columns is None:
    st.error("Failed to initialize the model. Please check the error messages above.")
    st.stop()

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Get unique values for each feature
road_types = sorted(data['Road_Type'].unique())
weather_conditions = sorted(data['Weather'].unique())
accident_locations = sorted(data['Accident_Location'].unique())
collision_types = sorted(data['Collision_Type'].unique())

# Create input widgets
road_type = st.sidebar.selectbox("Road Type", road_types)
weather = st.sidebar.selectbox("Weather Condition", weather_conditions)
accident_location = st.sidebar.selectbox("Accident Location", accident_locations)
collision_type = st.sidebar.selectbox("Collision Type", collision_types)

# Prediction function
def predict_severity(input_data):
    try:
        # Create DataFrame with user input
        input_df = pd.DataFrame([input_data])
        
        # One-hot encode the input
        input_encoded = pd.get_dummies(input_df)
        
        # Ensure all columns from training data are present
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_encoded)
        
        # Map prediction to severity level
        severity_map_reverse = {4: 'Fatal', 3: 'Grievous Injury', 2: 'Simple Injury', 1: 'Damage Only'}
        severity = severity_map_reverse[prediction[0]]
        
        return severity, prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Create input data dictionary
input_data = {
    'Road_Type': road_type,
    'Weather': weather,
    'Accident_Location': accident_location,
    'Collision_Type': collision_type
}

# Make prediction when button is clicked
if st.sidebar.button("Predict Severity"):
    severity, level = predict_severity(input_data)
    
    if severity is not None:
        # Display prediction
        st.markdown("### Prediction Results")
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Predicted Severity: {severity}</h3>
            <p>Deployment Level: {level}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display deployment recommendations
        st.markdown("### Deployment Recommendations")
        if level == 4:
            st.warning("🚨 **High Priority Response Required**\n- Full emergency response team\n- Multiple ambulances\n- Police presence\n- Traffic control units")
        elif level == 3:
            st.error("⚠️ **Significant Response Required**\n- Emergency response team\n- At least one ambulance\n- Police presence")
        elif level == 2:
            st.info("ℹ️ **Moderate Response Required**\n- Basic emergency response\n- One ambulance\n- Police presence if available")
        else:
            st.success("✅ **Standard Response Required**\n- Basic emergency response\n- Police presence if available")

# Add information about the model
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About the Model
This application uses a CatBoost Classifier trained on historical accident data to predict accident severity and recommend appropriate emergency response levels.
""")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed with ❤️ for Road Safety</p>
    <p>Version 1.0.0</p>
</div>
""", unsafe_allow_html=True) 