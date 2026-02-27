import os
import requests
import zipfile

MODEL_URL = "PASTE_YOUR_ZIP_LINK_HERE"
ZIP_PATH = "skin_model.zip"
MODEL_PATH = "skin_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    
    response = requests.get(MODEL_URL)
    
    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)

    print("Extracting model...")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall()

    print("Model ready.")
import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import numpy as np

skin_model = load_model("skin_model.h5")

IMG_SIZE = 128

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Vitamin AI Health Analyzer",
    page_icon="🧬",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    color: #00C896;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}

.card-low {
    background-color: #00c896;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 18px;
    margin-top:10px;
}

.card-high {
    background-color: #ff4b4b;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 18px;
    margin-top:10px;
}

.card-diet {
    background-color: #1f77b4;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 18px;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>Vitamin AI Health Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered Vitamin Deficiency Detection & Diet Recommendation</div>", unsafe_allow_html=True)

st.write("---")

# ---------------- LOAD MODEL ----------------
model = joblib.load("vitamin_model.pkl")

# ---------------- INPUT SECTION ----------------
st.header("Enter your details")

age = st.number_input("Age", 1, 100, 25)

sun = st.number_input("Sun exposure (hours/day)", 0.0, 12.0, 1.0)

fatigue = st.selectbox("Fatigue", [0, 1])

bone = st.selectbox("Bone pain", [0, 1])

veg = st.selectbox("Vegetarian", [0, 1])
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once (put this at top of file, after imports)
skin_model = tf.keras.models.load_model("skin_model.h5")

# ---------------- SKIN IMAGE UPLOAD ----------------

st.write("---")
st.header("Upload Skin Image (Optional)")

uploaded_file = st.file_uploader(
    "Upload a skin image for AI analysis",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Skin Image", width=400)
    st.success("Skin image uploaded successfully")

# ---------------- PREDICT BUTTON ----------------
if st.button("Predict"):

    data = np.array([[age, sun, fatigue, bone, veg]])
    prediction = model.predict(data)

    # get questionnaire risks
    risk_d = prediction[0][0]
    risk_b12 = prediction[0][1]
    risk_c = prediction[0][2]
    # -------- SKIN CNN PREDICTION --------
skin_result = 0   # default

if uploaded_file is not None:

    from tensorflow.keras.models import load_model
    from PIL import Image
    import numpy as np

    skin_model = load_model("skin_model.h5")

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128,128))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction_skin = skin_model.predict(img_array)

    if prediction_skin[0][0] > 0.5:
        skin_result = 1
    else:
        skin_result = 0
    data = np.array([[age, sun, fatigue, bone, veg]])
    # get vitamin risks from questionnaire model
    prediction = model.predict(data)

    risk_d = prediction[0][0]
    risk_b12 = prediction[0][1]
    risk_c = prediction[0][2]
    # Combine questionnaire and skin result
    final_score = 0

# questionnaire risk (use probability threshold)
    if risk_d > 0.5:
       final_score += 1

    if risk_b12 > 0.5:
       final_score += 1

    if risk_c > 0.5:
       final_score += 1

# skin risk
    if uploaded_file is not None and skin_result == 1:
       final_score += 1

    # final decision
    if final_score >= 2:
        st.error("Final AI Result: High risk of vitamin deficiency")
    else:
        st.success("Final AI Result: Low risk of vitamin deficiency")

    # show individual vitamin risks
    if risk_d == 1:
        st.error("Vitamin D deficiency risk: HIGH")
    else:
        st.success("Vitamin D deficiency risk: LOW")

    if risk_b12 == 1:
        st.error("Vitamin B12 deficiency risk: HIGH")
    else:
        st.success("Vitamin B12 deficiency risk: LOW")

    if risk_c == 1:
        st.error("Vitamin C deficiency risk: HIGH")
    else:
        st.success("Vitamin C deficiency risk: LOW")

    st.write("---")

    st.warning("Disclaimer: This is not a medical diagnosis. Consult a doctor.")

# ---------------- FOOTER ----------------
st.write("---")

st.caption("AI Health Analyzer | CSP Project | Built using Python & Streamlit")


