import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(page_title="Lung Cancer Prediction App", layout="wide", initial_sidebar_state="expanded")

# Apply custom CSS for styling
st.markdown("""
    <style>
        /* General Background and Text */
        body {
            background-color: #F9F9F9;
            color: #333333;
            font-family: 'Arial', sans-serif;
        }

        /* Gradient Button Styling */
        .stButton>button {
            background: linear-gradient(90deg, #4CAF50, #2196F3);
            color: white;
            font-size: 54px;
            font-weight: bold;
            border-radius: 10px;
            border: none;
            height: 50px;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #2196F3, #4CAF50);
            transform: scale(1.05);
        }

       /* Input Boxes */
.stSelectbox>div, .stRadio>div, .stNumberInput>div {
    background: linear-gradient(90deg, #2196F3, #4CAF50); /* Gradient background */
    border-radius: 8px; /* Rounded corners */
    padding: 10px; /* Padding inside the box */
    border: 1px solid #DDDDDD; /* Light gray border */
    transition: transform 0.3s ease, box-shadow 0.3s ease; /* Smooth animation for size and shadow */
}

/* Hover Effect */
.stSelectbox>div:hover, .stRadio>div:hover, .stNumberInput>div:hover {
    transform: scale(1.05); /* Slightly increase size */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Add shadow for a 3D effect */
}

        /* Labels */
        .stRadio label, .stSelectbox label, .stNumberInput label {
            color: black;
            font-size: 18px;
            font-weight: bold;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #333333;
        }

        /* Subheaders */
        .stSubheader {
            color: #4CAF50;
        }

        /* Gradient Header */
        .gradient-header {
            background: linear-gradient(90deg, #4CAF50, #2196F3);
            color: white;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            font-size: 44px;
            font-weight: bold;
        }

        /* Divider Line */
        hr {
            border: 5px solid #DDDDDD;
        }
    </style>
    """, unsafe_allow_html=True)


# Load dataset
@st.cache
def load_data():
    data = pd.read_csv("survey_lung_cancer_expanded.csv")  # Replace with your actual file path
    data.columns = data.columns.str.strip()  # Clean column names

    # Preprocess the data
    label_encoder = LabelEncoder()
    data['GENDER'] = label_encoder.fit_transform(data['GENDER'])
    data['LUNG_CANCER'] = label_encoder.fit_transform(data['LUNG_CANCER'])

    binary_features = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                       'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
                       'CHEST PAIN', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                       'SWALLOWING DIFFICULTY']
    data[binary_features] = data[binary_features].replace({2: 1, 1: 0})

    return data

# Train the model
@st.cache
def train_model(data):
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns, accuracy

# Load and preprocess data
data = load_data()

# Train the model
model, feature_columns, model_accuracy = train_model(data)

# Gradient title
st.markdown("""
    <div class="gradient-header">
        💡 Lung Cancer Prediction App
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <h3 style="color:#4CAF50;">Predict the likelihood of lung cancer</h3>
    <p style="color:#555555;">This application helps users predict the possibility of lung cancer based on multiple health attributes. It uses machine learning to provide accurate predictions.</p>
    """, unsafe_allow_html=True)

st.write(f"### Model Accuracy: **{model_accuracy*100:.2f}%**")
st.markdown("<hr>", unsafe_allow_html=True)

# Input form for user data
st.header("📝 Enter Your Details:")

# Organizing the gender and age input into columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender:", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
    age = st.number_input("Age:", min_value=1, max_value=120, step=1)

# Three columns for the rest of the attributes
col3, col4, col5 = st.columns(3)

with col3:
    smoking = st.radio("Smoking:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    yellow_fingers = st.radio("Yellow Fingers:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    anxiety = st.radio("Anxiety:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    peer_pressure = st.radio("Peer Pressure:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])

with col4:
    chronic_disease = st.radio("Chronic Disease:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    fatigue = st.radio("Fatigue:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    allergy = st.radio("Allergy:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    wheezing = st.radio("Wheezing:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    chest_pain = st.radio("Chest Pain:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])

with col5:
    alcohol_consumption = st.radio("Alcohol Consumption:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    coughing = st.radio("Coughing:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    shortness_of_breath = st.radio("Shortness of Breath:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    swallowing_difficulty = st.radio("Swallowing Difficulty:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])

# Create input data for prediction
user_data = [gender[1], age, smoking[1], yellow_fingers[1], anxiety[1], peer_pressure[1], chronic_disease[1], fatigue[1],
             allergy[1], wheezing[1], chest_pain[1], alcohol_consumption[1], coughing[1], shortness_of_breath[1], swallowing_difficulty[1]]

# Predict button
if st.button("🔍 Predict"):
    user_df = pd.DataFrame([user_data], columns=feature_columns)
    prediction = model.predict(user_df)
    result = "YES" if prediction[0] == 1 else "NO"
    st.subheader(f"🔴 Lung Cancer Detected: **{result}**")
    if result == "YES":
        st.error("🛑 Immediate consultation with a healthcare professional is recommended.")
    else:
        st.success("🟢 You're likely safe, but continue to maintain a healthy lifestyle.")
