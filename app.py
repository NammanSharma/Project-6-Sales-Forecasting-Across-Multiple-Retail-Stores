import streamlit as st
import pickle
import numpy as np
import tensorflow as tf

# App title
st.set_page_config(page_title="LSTM Sales Forecast", layout="centered")
st.title("📈 LSTM Sales Forecast App")
st.markdown("Enter the **last 3 months' sales** to predict the **next month's sales**.")

# Load model and scaler safelystr
try:
    model = tf.keras.models.load_model("lstm_sales_model.h5")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    st.success("✅ Model and scaler loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model or scaler: {e}")
    st.stop()

# Input from user
input_data = st.text_input("Enter 3 months of sales (comma-separated)", "20000,22000,25000")

if st.button("Predict"):
    try:
        values = [float(x.strip()) for x in input_data.split(",")]
        if len(values) != 3:
            st.warning("⚠️ Please enter exactly 3 values.")
        else:
            input_array = np.array(values).reshape(-1, 1)      # (3,1)
            scaled = scaler.transform(input_array)             # (3,1)
            scaled_input = scaled.reshape(1, 3, 1)              # (1,3,1)

            prediction = model.predict(scaled_input)
            predicted_value = prediction[0][0]

            st.success(f"📊 Predicted Sales for Next Month: ₹{predicted_value:,.2f}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")